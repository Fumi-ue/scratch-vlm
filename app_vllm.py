"""
Small Flask server that exposes an OpenAI compatible vision-chat endpoint,
but uses a *remote vLLM server* as the backend.

前提:
- vLLM サーバーは Docker で起動済み:
  docker run ... -p 8888:8000 vllm-custom:25.09 \
    vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
      --host 0.0.0.0 --port 8000 --max-model-len 32768
"""

from flask import Flask, request, jsonify
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
import json
import re
import requests

try:
    import jsonschema
except Exception:
    jsonschema = None

app = Flask(__name__)

# ==== vLLM サーバー設定 ====
# Docker の -p 8888:8000 に合わせる
VLLM_BASE_URL = "http://localhost:8888/v1/chat/completions"

# vLLM 側の model 名（serve したときのパス）
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"


# ==== JSON フォーマット関連のユーティリティ ====


def build_format_instructions(response_format: Dict[str, Any]) -> str:
    """Return extra system instructions to enforce JSON-only replies."""

    if not isinstance(response_format, dict):
        return ""
    rf_type = response_format.get("type", "text")
    if rf_type == "json_object":
        return (
            "You MUST respond with ONLY a valid minified JSON object. "
            "No markdown, no code fences, no extra text."
        )
    if rf_type == "json_schema":
        schema = response_format.get("json_schema", {})
        schema_dump = json.dumps(schema, ensure_ascii=False)
        return (
            "You MUST respond with ONLY a valid minified JSON object that "
            "conforms to the following JSON Schema strictly. "
            "Do not include any text outside the JSON.\n"
            f"SCHEMA:\n{schema_dump}"
        )
    return ""


def try_extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Heuristically pull the first JSON object from the model output."""
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            # fall-through to regex
            pass

    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            return json.loads(match.group(0))
        except Exception:
            continue

    return None


def validate_with_schema(obj: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate obj against JSON schema if jsonschema is available."""
    if jsonschema is None:
        return True, ""
    try:
        jsonschema.validate(instance=obj, schema=schema)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def format_response(
    raw_text: str, response_format: Dict[str, Any]
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Validate/format raw model text according to response_format rules."""
    rf_type = response_format.get("type", "text")
    if rf_type not in ("json_object", "json_schema"):
        # テキストモードならそのまま返す
        return raw_text, None

    parsed_json = try_extract_json(raw_text)
    if parsed_json is None:
        return None, {
            "error": {
                "message": "Model output was not valid JSON.",
                "type": "invalid_json_output",
            }
        }

    if rf_type == "json_schema":
        schema = response_format.get("json_schema") or {}
        ok, schema_error = validate_with_schema(parsed_json, schema)
        if not ok:
            return None, {
                "error": {
                    "message": "JSON does not conform to the provided schema.",
                    "type": "schema_validation_error",
                    "details": schema_error,
                }
            }

    return json.dumps(parsed_json, ensure_ascii=False), None


# ==== system メッセージに JSON 指示を注入 ====


def inject_system_instructions(
    messages: List[Dict[str, Any]], extra_system: str
) -> List[Dict[str, Any]]:
    """system メッセージに format_instructions を足す（なければ新規追加）"""
    if not extra_system:
        return messages

    new_messages: List[Dict[str, Any]] = []
    system_found = False
    for msg in messages:
        if msg.get("role") == "system" and not system_found:
            content = msg.get("content", "")
            if isinstance(content, str):
                content = (content + "\n" + extra_system).strip()
            # list 形式のときはテキストを追加するなど、必要なら拡張してもよい
            new_messages.append({**msg, "content": content})
            system_found = True
        else:
            new_messages.append(msg)

    if not system_found:
        new_messages.insert(
            0,
            {
                "role": "system",
                "content": extra_system,
            },
        )
    return new_messages


# ==== メインの /v1/chat/completions エンドポイント ====


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completion():
    t_total0 = perf_counter()
    try:
        req = request.get_json(silent=True) or {}

        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 16384)
        requested_model = (req.get("model", "") or "").strip()
        effective_model = requested_model or DEFAULT_MODEL
        temperature = req.get("temperature", 0.5)
        response_format = req.get("response_format", {"type": "text"})

        # JSON Schema などの指示を system に注入
        format_instructions = build_format_instructions(response_format)
        vllm_messages = inject_system_instructions(messages, format_instructions)

        # vLLM に投げる payload
        vllm_payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": vllm_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t_vllm0 = perf_counter()
        vllm_resp = requests.post(VLLM_BASE_URL, json=vllm_payload, timeout=600)
        t_vllm1 = perf_counter()

        if not vllm_resp.ok:
            return jsonify(
                {
                    "error": {
                        "message": f"vLLM backend error: {vllm_resp.status_code} {vllm_resp.text}",
                        "type": "backend_error",
                    }
                }
            ), 500

        vllm_json = vllm_resp.json()
        # OpenAI 互換を前提
        choice = vllm_json["choices"][0]
        raw_text = choice["message"]["content"]

        # JSON Schema / json_object に従って整形
        content_out, error_payload = format_response(raw_text, response_format)
        if error_payload:
            return jsonify(error_payload), 400
        assert content_out is not None

        # usage 情報は vLLM があればそのまま利用、無ければ None
        usage = vllm_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        t_total1 = perf_counter()
        total_ms = int((t_total1 - t_total0) * 1000)
        vllm_ms = int((t_vllm1 - t_vllm0) * 1000)

        # ログ (必要に応じて縮めてもOK)
        log_meta = {
            "model": effective_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format.get("type", "text"),
        }
        print(f"[REQUEST_META] {json.dumps(log_meta, ensure_ascii=False)}")
        print(
            f"[LATENCY_MS] backend={vllm_ms} total={total_ms}, "
            f"tokens={usage}"
        )

        # クライアントには OpenAI 風のレスポンスを返す
        return jsonify(
            {
                "id": vllm_json.get("id", "chatcmpl-001"),
                "object": "chat.completion",
                "created": vllm_json.get(
                    "created", int(datetime.now().timestamp())
                ),
                "model": effective_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content_out},
                        "finish_reason": choice.get("finish_reason", "stop"),
                    }
                ],
                "usage": usage,
                "response_format": response_format,
            }
        )

    except Exception as e:
        print("[ERROR]", repr(e))
        return jsonify(
            {
                "error": {
                    "message": str(e),
                    "type": "internal_server_error",
                }
            }
        ), 500


if __name__ == "__main__":
    # vLLM サーバーとは別プロセス（or 別コンテナ）で動かす想定
    app.run(host="0.0.0.0", port=8002, threaded=True)

"""Small Flask server that exposes an OpenAI compatible vision-chat endpoint."""

from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import yaml
from pathlib import Path
from datetime import datetime
import re
import json
import importlib.util
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema
except Exception:
    jsonschema = None
import threading
import mlx.core as mx

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

if importlib.util.find_spec("torch") is None:
    raise RuntimeError(
        "PyTorch is required by the Hugging Face processor backend. "
        "Install torch/torchvision (see README) and restart the server."
    )

app = Flask(__name__)
infer_lock = threading.Lock()

MODEL_PATH = "mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16"
RESIZE_CONFIG_PATH = Path("config/resize.yaml")

# モデル読み込み（起動時に1回）
# MODEL_PATH を切り替えれば他モデルにも対応可。
model, processor = load(MODEL_PATH)
config = load_config(MODEL_PATH)
tokenizer = getattr(processor, "tokenizer", None)
if tokenizer is None:
    print("[WARN] tokenizer missing from processor; token counts will be approximate.")


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
    except Exception as exc:  # pragma: no cover - jsonschema optional
        return False, str(exc)


def parse_messages(messages: List[Dict[str, Any]]) -> Tuple[str, str, Optional[Image.Image]]:
    """Split incoming messages into system text, user text, and optional image."""

    system_prompt = ""
    user_content = ""
    image = None

    for message in messages:
        role = message.get("role")
        if role == "system":
            system_prompt = message.get("content", "")
        elif role == "user":
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        user_content += part.get("text", "") + "\n"
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        base64_data = url.split(",")[-1]
                        image = Image.open(BytesIO(base64.b64decode(base64_data)))
            else:
                user_content += str(content or "")
    return system_prompt.strip(), user_content.strip(), image


def count_tokens(text: Optional[str], *, add_special_tokens: bool = False) -> int:
    """Return tokenizer-accurate token counts (model-specific when available)."""

    if not text:
        return 0

    if tokenizer is None:
        return len(re.findall(r"\S+", text))

    try:
        if hasattr(tokenizer, "encode"):
            token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        else:
            encoded = tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            token_ids = encoded.get("input_ids", [])

        if isinstance(token_ids, list):
            if token_ids and isinstance(token_ids[0], list):
                return len(token_ids[0])
            return len(token_ids)
        if hasattr(token_ids, "shape"):
            return int(token_ids.shape[-1])
        return len(token_ids)
    except Exception:
        return len(re.findall(r"\S+", text))

def run_generation(prompt: str, image: Optional[Image.Image], max_tokens: int, temperature: float):
    """Run the heavy model call under a lock to keep MLX thread-safe."""

    with infer_lock:
        answer = generate(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        mx.synchronize()
    return answer.text


def format_response(raw_text: str, response_format: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Validate/format raw model text according to response_format rules."""

    rf_type = response_format.get("type", "text")
    if rf_type not in ("json_object", "json_schema"):
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


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completion():
    try:
        t_total0 = perf_counter()

        req = request.get_json(silent=True) or {}

        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 16384)
        model_name = req.get("model", "")
        temperature = req.get("temperature", 0.5)
        response_format = req.get("response_format", {"type": "text"})

        system_prompt, user_content, image = parse_messages(messages)
        format_instructions = build_format_instructions(response_format)
        effective_system = (system_prompt + "\n" + format_instructions).strip()
        final_prompt = f"{effective_system}\n{user_content}".strip()

        t_pre0 = perf_counter()
        # image = preprocess_image(image)

        formatted_prompt = apply_chat_template(
            processor, config, final_prompt, num_images=1 if image else 0
        )

        t_pre1 = perf_counter()
        t_gen0 = perf_counter()

        raw_text = run_generation(
            prompt=formatted_prompt,
            image=image,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        t_gen1 = perf_counter()

        content_out, error_payload = format_response(raw_text, response_format)
        if error_payload:
            return jsonify(error_payload), 400
        assert content_out is not None  # narrow type for static analyzers

        prompt_tokens = count_tokens(formatted_prompt)
        completion_tokens = count_tokens(content_out)
        total_tokens = prompt_tokens + completion_tokens

        t_total1 = perf_counter()

        # 各区間の時間（ms）
        pre_ms = int((t_pre1 - t_pre0) * 1000)           # 前処理（画像リサイズ＋テンプレ生成）
        gen_ms = int((t_gen1 - t_gen0) * 1000)           # モデル推論
        total_ms = int((t_total1 - t_total0) * 1000)     # リクエスト全体
        post_ms = total_ms - pre_ms - gen_ms             # 後処理（JSON抽出/検証等）

        # === ログ出力 ===
        request_meta = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format.get("type", "text"),
        }
        text_payload = {
            "system_prompt": system_prompt,
            "user_text": user_content,
            "assistant_text": content_out,
        }
        print(f"[REQUEST_META] {json.dumps(request_meta, ensure_ascii=False)}")
        print(f"[IO_TEXT] {json.dumps(text_payload, ensure_ascii=False)}")
        print(
            f"[TOKENS] prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}"
        )
        print(f"[LATENCY_MS] pre={pre_ms} gen={gen_ms} post={post_ms} total={total_ms}")

        return jsonify(
            {
                "id": "chatcmpl-001",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content_out},
                        "finish_reason": "stop",
                    }
                ],
                # 応答に反映（デバッグ用）
                "response_format": response_format,
            }
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, threaded=False)

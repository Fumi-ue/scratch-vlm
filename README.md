# scratch-llm

Flask ベースの簡易サーバーで、MLX の Vision-Language Model (VLM) を OpenAI 互換の `/v1/chat/completions` エンドポイントとして公開します。画像を含むチャットリクエストを受け取り、JSON レスポンスもサポートします。

## セットアップ
1. Python 3.11 以上を用意し、仮想環境を作成します。
2. 依存ライブラリをインストールします。
   ```bash
   pip install -r requirements.txt
   ```
   - Hugging Face の Qwen 系 Processor は内部で PyTorch を要求するため、`torch` と `torchvision` のビルドが必要です。`pip install` で失敗する場合は [PyTorch の公式手順](https://pytorch.org/get-started/locally/) に従ってインストールしてください。
3. Apple Silicon + macOS 上の MLX を想定しています。他プラットフォームでは `mlx` / `mlx-vlm` の代替を用意するか、コード側でモデル呼び出しを差し替えてください。

## 起動方法
```bash
export FLASK_APP=app.py
python app.py  # もしくは flask run --host 0.0.0.0 --port 8002
```

## 開発メモ
- `response_format` に `json_schema` を指定すると jsonschema バリデーションを行います。`jsonschema` が未インストールの場合はスキップされるので、厳格な JSON 応答が必要な場合は必ずインストールしてください。

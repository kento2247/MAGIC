# app

UI 関連のコードをここにまとめています。

## ファイル構成

- `app/gradio_app.py`
  - Gradio ベースのWeb UIです。
  - いまは demo pipeline を呼び出して、生成画像、クロップ、レイヤー、SVG、ZIP を一通り見せます。
- `app/pipeline.py`
  - ジョブディレクトリの作成、demo 画像生成、demo クロップ、demo レイヤー作成、SVG化、ZIP化を担当します。

## 起動

依存が入っていれば、以下で起動できます。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m app.gradio_app
```

## 出力先

ジョブごとの生成物は以下にまとまります。

```text
app/artifacts/jobs/<job_id>/
```

中身:

- `generated/`
- `crops/`
- `layers/`
- `vectorized/`
- `bundle/`
- `job_summary.json`

## 今後の差し替えポイント

demo pipeline は `app/pipeline.py` の `run_demo_job()` から始まっています。

本番化するときは主に以下を差し替えます。

1. `render_demo_generated_image()`
   - `paperbanana` の画像生成呼び出しへ置き換え
2. `save_crop_manifest()`
   - `crop.py` の汎用化版へ接続
3. `create_demo_layers()`
   - `run_layered_crops.py` の本番実行へ接続
4. `vectorize_module()`
   - 必要に応じて `phase2_svg.py` ベースのモジュール再構築へ拡張

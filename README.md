# MAGIC

`MAGIC: Model Architecture Generation with Interactive Components`

論文手法の説明文から、図示に必要な情報の整理、キャプション生成、図の生成、SVG書き出しまでを対話的に行うFlaskベースのプロトタイプです。現時点の主導線はCLIスクリプトではなくWebアプリです。

## 主な機能

- 手法説明のレビュー
  - `utils/InstructionCompletion.py` が説明文を評価し、図示に足りない情報があれば追加入力を促します。
- 説明文の整形
  - 同じく `InstructionCompletion` が、図生成向けのMarkdown要約と図キャプションを生成します。
- 図の生成
  - `utils/PaperBanana.py` を経由して画像生成パイプラインを呼び出します。
- SVGエクスポート
  - `utils/PingToSVG.py` がOCR、矢印抽出、SAMベースの領域推定を使ってSVG化します。
- デモサンプルの読み込み
  - `demo_sample/` 配下の既存サンプルをUIから選択できます。

## 主要ファイル

- `app.py`
  - Flaskアプリ本体です。トップ画面、デモサンプル取得、図生成、SVG出力APIを提供します。
- `templates/index.html`
  - Web UI本体です。
- `demo_sample/`
  - 既存の説明文、キャプション、サンプル画像をまとめたデモデータです。
- `assets/`
  - 実験用の入力画像です。
- `crop.py`
  - Qwen3-VLでモジュール領域を検出してクロップを保存する試験用スクリプトです。
- `phase2_svg.py`
  - クロップ済み画像からハイブリッドSVGを再構築する試験用スクリプトです。
- `main.py`
  - `Qwen-Image-Layered` 単体実行の確認用スクリプトです。
- `run_layered_crops.py`
  - クロップ済み画像をまとめてレイヤー分離する補助スクリプトです。

## セットアップ

前提:

- Python `3.10.0`
- `uv`

環境変数を用意します。

```bash
cp .env.template .env
```

`.env` に `OPENAI_API_KEY` を設定してください。

依存をインストールします。

```bash
uv sync
```

## 起動方法

```bash
uv run python app.py
```

起動後、`http://127.0.0.1:8000` を開いて利用します。

## Webアプリの流れ

1. 新規入力か、`demo_sample/` の既存サンプルを選択します。
2. 手法説明を入力します。
3. 説明が不足していれば、図示に必要な追加質問が返ります。
4. 十分な説明がそろうと、Markdown要約と図キャプションが生成されます。
5. 要約とキャプションを編集したうえで図を生成します。
6. 生成画像をSVGとして書き出せます。

## 補足

- `OPENAI_API_KEY` が未設定でも一部のローカル判定は動作しますが、レビュー品質と整形品質は低下します。
- 生成物やキャッシュはコミット対象から外しています。`generated_images/`、`output/`、`phase2_layers*/`、`phase2_output/`、`results/`、`__pycache__/`、`*.log` はローカル生成物として扱ってください。
- 旧来のCLI検証スクリプトは残していますが、現状の主導線は `app.py` です。

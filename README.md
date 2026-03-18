# mini_hackathon_2026

論文図解ベクタライズのミニハッカソン用プロトタイプです。

## 現在の構成

- `crop.py`
  - フェーズ1。Qwen3-VL を使って意味的モジュールを検出し、クロップ PNG を保存します。
  - 追加で `output/crop_manifest.json` を出力し、元画像上の座標を保持します。
- `main.py`
  - `Qwen-Image-Layered` の単体実行サンプルです。
- `phase2_svg.py`
  - フェーズ2。クロップ画像を `Qwen-Image-Layered` でレイヤー分離し、OCR/簡易分類/フォールバックを使ってハイブリッドSVGを再構築します。

## ハッカソン向けの実運用方針

現実的には、最初から全レイヤーを完全ベクタ化するよりも、

- テキストは OCR で `<text>`
- ボックスや線は Potrace や簡易図形化
- 難しいレイヤーは SVG 内に透過 PNG として埋め込み

という「ハイブリッドSVG」にするのが一番早いです。

これでも `draw.io` や `Illustrator` 上で、

- モジュール単位の `<g>` グループ
- OCRできたラベルの直接編集
- 失敗レイヤーの差し替え

がしやすくなります。

## 使い方

### 1. フェーズ1: クロップ + manifest

```bash
python3 crop.py
```

出力:

- `output/*.png`
- `output/crop_manifest.json`

### 2. フェーズ2: レイヤー分離 + SVG再構築

```bash
python3 phase2_svg.py \
  --manifest output/crop_manifest.json \
  --layers-dir phase2_layers \
  --svg-out phase2_output/reconstructed.svg
```

既に `phase2_layers/` にレイヤー PNG がある場合は、推論をスキップできます。

```bash
python3 phase2_svg.py \
  --manifest output/crop_manifest.json \
  --layers-dir phase2_layers \
  --svg-out phase2_output/reconstructed.svg \
  --skip-inference
```

## 追加で入れると強くなるもの

- `pytesseract`
  - テキストレイヤーを `<text>` 化しやすくなります。
- `potrace`
  - 線や単純図形を `<path>` 化しやすくなります。

依存がなくても `phase2_svg.py` 自体は動きますが、その場合はラスタ埋め込みフォールバックが増えます。

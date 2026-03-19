from pathlib import Path

from flask import Flask, jsonify
from openai import OpenAI
import base64
from PIL import Image
import re
import os
import json

app = Flask(__name__)

logger = app.logger


class ImageCropper:
    def __init__(
        self,
        api_key: str,
        output_dir: Path = Path("output"),
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    ):
        self.output_dir = output_dir
        self.client = OpenAI(
            base_url="http://http://127.0.0.1:22002/v1", api_key=api_key
        )
        self.output_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.prompt_text = (
            "You are a helpful assistant. Find and output the bounding boxes for the following modules in this model diagram:\n"
            "1. Landmark Distribution Parchfication Module\n"
            "2. Visual-Linguistic Spatial Integration Module\n"
            "3. Existence Aware Polygon Segmentation Module\n\n"
            "4. Input\n"
            "Please output in the format: Module Name: (xmin, ymin),(xmax, ymax)"
        )

    def crop(self, image_path: Path):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        self.logger.info("vLLMサーバーにリクエストを送信中...")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.1,
        )

        output_text = response.choices[0].message.content
        self.logger.info("--- Model Output ---")
        self.logger.info(output_text)
        self.logger.info("--------------------")

        # 3. 座標の抽出とクロップ（実際の出力形式に合わせた正規表現）
        img = Image.open(image_path)
        width, height = img.size
        boxes_found = False
        manifest = {
            "source_image": image_path,
            "image_width": width,
            "image_height": height,
            "modules": [],
        }

        # 行ごとに処理して、モジュール名と座標を取り出す
        for i, line in enumerate(output_text.strip().split("\n")):
            # (数字, 数字),(数字, 数字) のパターンを探す
            coord_match = re.search(r"\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)", line)

            if coord_match:
                boxes_found = True
                # Qwen3の出力に合わせて x, y の順で取得
                xmin, ymin, xmax, ymax = map(int, coord_match.groups())

                # モジュール名の抽出（行頭の「1. 」などを無視して「:」の前までを取得）
                name_match = re.search(r"^[0-9\.\s]*([^:]+):", line)
                if name_match:
                    # ファイル名に使えるように空白をアンダースコアに変換
                    module_name = name_match.group(1).strip().replace(" ", "_")
                else:
                    module_name = f"module_{i+1}"

                # 1000スケールから実際のピクセル座標へ変換
                left = int((xmin / 1000.0) * width)
                top = int((ymin / 1000.0) * height)
                right = int((xmax / 1000.0) * width)
                bottom = int((ymax / 1000.0) * height)

                # クリップ処理（画像サイズをはみ出ないように）
                left, top = max(0, left), max(0, top)
                right, bottom = min(width, right), min(height, bottom)

                cropped_img = img.crop((left, top, right, bottom))

                save_path = self.output_dir / f"{module_name}.png"
                cropped_img.save(save_path)
                self.logger.info(f"Saved: {save_path} (Coords: {left}, {top}, {right}, {bottom})")
                manifest["modules"].append(
                    {
                        "module_name": module_name,
                        "crop_path": save_path,
                        "bbox": {
                            "left": left,
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                        },
                        "normalized_bbox_1000": {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                        },
                        "crop_size": {
                            "width": right - left,
                            "height": bottom - top,
                        },
                    }
                )

        if not boxes_found:
            self.logger.error("エラー: 出力から座標が抽出できませんでした。")
            return

        manifest_path = self.output_dir / "crop_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved manifest: {manifest_path}")

@app.get("/")
@app.post("/crop_image")
def crop_image():
    pass



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

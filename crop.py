import base64
import json
import re
import os
from openai import OpenAI
from PIL import Image

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def detect_and_crop_vllm(image_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    
    client = OpenAI(
        base_url="http://127.0.0.1:22002/v1",
        api_key="dummy"
    )
    
    base64_image = encode_image_to_base64(image_path)
    model_name = "Qwen/Qwen3-VL-4B-Instruct" 

    prompt_text = (
        "You are a helpful assistant. Find and output the bounding boxes for the following modules in this model diagram:\n"
        "1. Landmark Distribution Parchfication Module\n"
        "2. Visual-Linguistic Spatial Integration Module\n"
        "3. Existence Aware Polygon Segmentation Module\n\n"
        "4. Input\n"
        "Please output in the format: Module Name: (xmin, ymin),(xmax, ymax)"
    )

    print("vLLMサーバーにリクエストを送信中...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=512,
        temperature=0.1
    )

    output_text = response.choices[0].message.content
    print("--- Model Output ---")
    print(output_text)
    print("--------------------")

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
    for i, line in enumerate(output_text.strip().split('\n')):
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
            
            save_path = f"{output_dir}/{module_name}.png"
            cropped_img.save(save_path)
            print(f"Saved: {save_path} (Coords: {left}, {top}, {right}, {bottom})")
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
        print("エラー: 出力から座標が抽出できませんでした。")
        return

    manifest_path = os.path.join(output_dir, "crop_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {manifest_path}")

if __name__ == "__main__":
    detect_and_crop_vllm("asserts/candidate1.png")

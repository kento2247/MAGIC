import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


@dataclass
class OCRWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float


@dataclass
class LayerSummary:
    layer_path: str
    kind: str
    bbox: dict[str, int]
    alpha_coverage: float
    opaque_ratio: float
    notes: list[str]
    ocr_words: list[OCRWord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen-Image-Layered on cropped modules and rebuild a structured SVG."
    )
    parser.add_argument(
        "--manifest",
        default="output/crop_manifest.json",
        help="Manifest JSON emitted by crop.py",
    )
    parser.add_argument(
        "--layers-dir",
        default="phase2_layers",
        help="Directory where per-module separated layers are written",
    )
    parser.add_argument(
        "--svg-out",
        default="phase2_output/reconstructed.svg",
        help="Final SVG path",
    )
    parser.add_argument(
        "--metadata-out",
        default="phase2_output/layer_metadata.json",
        help="Optional JSON summary of layer classifications",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=6,
        help="Number of layered outputs to request from Qwen-Image-Layered",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        help="Resolution bucket passed to Qwen-Image-Layered",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Seed for reproducible layer separation",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip Qwen inference and use existing PNGs under --layers-dir",
    )
    return parser.parse_args()


def load_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def image_to_base64_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def slugify(text: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in text)


def alpha_channel(image: Image.Image) -> Image.Image:
    if "A" in image.getbands():
        return image.getchannel("A")
    return Image.new("L", image.size, 255)


def alpha_stats(image: Image.Image) -> tuple[dict[str, int], float, float]:
    alpha = alpha_channel(image)
    bbox = alpha.getbbox()
    if bbox is None:
        return {"left": 0, "top": 0, "right": 0, "bottom": 0}, 0.0, 0.0

    hist = alpha.histogram()
    total = sum(hist) or 1
    non_zero = total - hist[0]
    opaque = sum(hist[200:])
    return (
        {"left": bbox[0], "top": bbox[1], "right": bbox[2], "bottom": bbox[3]},
        non_zero / total,
        opaque / total,
    )


def dominant_visible_color(image: Image.Image) -> tuple[int, int, int, int] | None:
    rgba = image.convert("RGBA")
    colors = rgba.getcolors(maxcolors=256 * 256)
    if not colors:
        return None

    visible = [entry for entry in colors if entry[1][3] > 10]
    if not visible:
        return None
    visible.sort(key=lambda item: item[0], reverse=True)
    return visible[0][1]


def color_to_svg_attrs(color: tuple[int, int, int, int], prefix: str = "fill") -> dict[str, str]:
    return {
        prefix: f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
        f"{prefix}-opacity": str(round(color[3] / 255, 3)),
    }


def looks_like_filled_box(image: Image.Image, bbox: dict[str, int]) -> bool:
    width = bbox["right"] - bbox["left"]
    height = bbox["bottom"] - bbox["top"]
    if width <= 4 or height <= 4:
        return False

    alpha = alpha_channel(image).crop((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))
    hist = alpha.histogram()
    total = sum(hist) or 1
    filled = total - hist[0]
    return (filled / total) > 0.72


def find_ocr_words(image: Image.Image) -> list[OCRWord]:
    try:
        import pytesseract
        from pytesseract import Output
    except Exception:
        return []

    data = pytesseract.image_to_data(image.convert("RGB"), output_type=Output.DICT)
    words: list[OCRWord] = []
    count = len(data.get("text", []))
    for idx in range(count):
        text = data["text"][idx].strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][idx])
        except Exception:
            confidence = -1.0
        if confidence < 30:
            continue
        words.append(
            OCRWord(
                text=text,
                left=int(data["left"][idx]),
                top=int(data["top"][idx]),
                width=int(data["width"][idx]),
                height=int(data["height"][idx]),
                confidence=confidence,
            )
        )
    return words


def classify_layer(image: Image.Image, layer_path: str) -> LayerSummary:
    bbox, alpha_coverage, opaque_ratio = alpha_stats(image)
    notes: list[str] = []
    ocr_words = find_ocr_words(image)

    width = max(1, bbox["right"] - bbox["left"])
    height = max(1, bbox["bottom"] - bbox["top"])
    aspect_ratio = max(width / height, height / width)

    if ocr_words:
        kind = "text"
        notes.append(f"OCR words detected: {len(ocr_words)}")
    elif alpha_coverage < 0.03 or aspect_ratio > 8.0:
        kind = "connector"
        notes.append("Low coverage or very thin aspect ratio")
    elif looks_like_filled_box(image, bbox):
        kind = "shape"
        notes.append("Dense alpha region; approximated as a box candidate")
    else:
        kind = "raster_fallback"
        notes.append("Could not confidently turn this layer into editable vectors")

    return LayerSummary(
        layer_path=layer_path,
        kind=kind,
        bbox=bbox,
        alpha_coverage=round(alpha_coverage, 4),
        opaque_ratio=round(opaque_ratio, 4),
        notes=notes,
        ocr_words=ocr_words,
    )


def load_qwen_pipeline():
    import torch
    from diffusers import QwenImageLayeredPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen-Image-Layered inference.")

    pipeline = QwenImageLayeredPipeline.from_pretrained(
        "Qwen/Qwen-Image-Layered",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_attention_slicing("auto")
    pipeline.enable_sequential_cpu_offload()
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    return pipeline, torch


def run_layer_separation(
    pipeline: Any,
    torch_module: Any,
    crop_path: str,
    output_dir: str,
    layers: int,
    resolution: int,
    seed: int,
) -> list[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image = Image.open(crop_path).convert("RGBA")
    prompt = (
        "Decompose this cropped academic diagram module into editable layers. "
        "Separate text labels, arrows, connector lines, border boxes, icons, and "
        "filled regions. Preserve exact geometry, transparent backgrounds, and do "
        "not merge unrelated elements."
    )
    negative_prompt = (
        "merged components, blurry text, duplicated arrows, missing borders, "
        "background canvas, extra decoration"
    )
    inputs = {
        "image": image,
        "generator": torch_module.Generator(device="cuda").manual_seed(seed),
        "true_cfg_scale": 4.0,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": 50,
        "num_images_per_prompt": 1,
        "layers": layers,
        "resolution": resolution,
        "cfg_normalize": True,
        "use_en_prompt": True,
    }
    with torch_module.inference_mode():
        output = pipeline(**inputs)
        output_layers = output.images[0]

    saved_paths: list[str] = []
    for index, output_layer in enumerate(output_layers):
        layer_path = os.path.join(output_dir, f"layer_{index:02d}.png")
        output_layer.save(layer_path)
        saved_paths.append(layer_path)
    return saved_paths


def existing_layers(output_dir: str) -> list[str]:
    return sorted(str(path) for path in Path(output_dir).glob("layer_*.png"))


def try_potrace(layer_path: str) -> ET.Element | None:
    if shutil.which("potrace") is None:
        return None

    image = Image.open(layer_path).convert("RGBA")
    alpha = alpha_channel(image)
    binary = alpha.point(lambda value: 255 if value > 80 else 0).convert("1")

    with tempfile.TemporaryDirectory() as tmpdir:
        pbm_path = os.path.join(tmpdir, "layer.pbm")
        svg_path = os.path.join(tmpdir, "layer.svg")
        binary.save(pbm_path)
        subprocess.run(
            ["potrace", pbm_path, "-s", "-o", svg_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        traced_svg = ET.parse(svg_path).getroot()

    group = ET.Element(f"{{{SVG_NS}}}g")
    for child in list(traced_svg):
        if child.tag.endswith("metadata"):
            continue
        group.append(child)
    return group


def append_embedded_image(
    parent: ET.Element,
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    element_id: str,
) -> None:
    href = f"data:image/png;base64,{image_to_base64_png(image)}"
    image_node = ET.SubElement(
        parent,
        f"{{{SVG_NS}}}image",
        {
            "id": element_id,
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            f"{{{XLINK_NS}}}href": href,
        },
    )
    image_node.set("preserveAspectRatio", "none")


def add_text_nodes(parent: ET.Element, summary: LayerSummary) -> bool:
    if not summary.ocr_words:
        return False

    text_group = ET.SubElement(parent, f"{{{SVG_NS}}}g", {"id": slugify(Path(summary.layer_path).stem) + "_text"})
    for idx, word in enumerate(summary.ocr_words):
        text_node = ET.SubElement(
            text_group,
            f"{{{SVG_NS}}}text",
            {
                "x": str(word.left),
                "y": str(word.top + word.height),
                "font-size": str(max(10, int(word.height * 0.9))),
                "font-family": "Arial, sans-serif",
                "fill": "#111111",
                "id": f"{slugify(Path(summary.layer_path).stem)}_word_{idx}",
            },
        )
        text_node.text = word.text
    return True


def add_shape_node(parent: ET.Element, image: Image.Image, summary: LayerSummary) -> bool:
    if summary.kind != "shape":
        return False

    bbox = summary.bbox
    fill = dominant_visible_color(image)
    if fill is None:
        return False

    ET.SubElement(
        parent,
        f"{{{SVG_NS}}}rect",
        {
            "id": slugify(Path(summary.layer_path).stem) + "_rect",
            "x": str(bbox["left"]),
            "y": str(bbox["top"]),
            "width": str(max(1, bbox["right"] - bbox["left"])),
            "height": str(max(1, bbox["bottom"] - bbox["top"])),
            "stroke": "#333333",
            "stroke-width": "1",
            **color_to_svg_attrs(fill),
        },
    )
    return True


def add_connector_node(parent: ET.Element, image: Image.Image, summary: LayerSummary) -> bool:
    if summary.kind != "connector":
        return False

    traced = None
    try:
        traced = try_potrace(summary.layer_path)
    except Exception:
        traced = None

    if traced is not None:
        traced.set("id", slugify(Path(summary.layer_path).stem) + "_trace")
        parent.append(traced)
        return True

    bbox = summary.bbox
    fill = dominant_visible_color(image) or (60, 60, 60, 255)
    ET.SubElement(
        parent,
        f"{{{SVG_NS}}}rect",
        {
            "id": slugify(Path(summary.layer_path).stem) + "_line",
            "x": str(bbox["left"]),
            "y": str(bbox["top"]),
            "width": str(max(1, bbox["right"] - bbox["left"])),
            "height": str(max(1, bbox["bottom"] - bbox["top"])),
            **color_to_svg_attrs(fill),
        },
    )
    return True


def build_module_group(module: dict[str, Any], layer_summaries: list[LayerSummary]) -> ET.Element:
    bbox = module["bbox"]
    module_group = ET.Element(
        f"{{{SVG_NS}}}g",
        {
            "id": slugify(module["module_name"]),
            "transform": f"translate({bbox['left']} {bbox['top']})",
            "data-source-crop": module["crop_path"],
        },
    )

    ET.SubElement(
        module_group,
        f"{{{SVG_NS}}}title",
    ).text = module["module_name"]

    for summary in layer_summaries:
        layer_image = Image.open(summary.layer_path).convert("RGBA")
        handled = False
        if summary.kind == "text":
            handled = add_text_nodes(module_group, summary)
        if not handled and summary.kind == "shape":
            handled = add_shape_node(module_group, layer_image, summary)
        if not handled and summary.kind == "connector":
            handled = add_connector_node(module_group, layer_image, summary)
        if not handled:
            append_embedded_image(
                module_group,
                layer_image,
                0,
                0,
                layer_image.width,
                layer_image.height,
                slugify(Path(summary.layer_path).stem) + "_raster",
            )
    return module_group


def build_svg_document(manifest: dict[str, Any], layer_map: dict[str, list[LayerSummary]]) -> ET.ElementTree:
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": str(manifest["image_width"]),
            "height": str(manifest["image_height"]),
            "viewBox": f"0 0 {manifest['image_width']} {manifest['image_height']}",
        },
    )

    ET.SubElement(
        root,
        f"{{{SVG_NS}}}desc",
    ).text = (
        "Hybrid SVG reconstructed from semantic crops and Qwen-Image-Layered outputs. "
        "Text nodes are editable; unresolved layers are preserved as embedded rasters."
    )

    for module in manifest["modules"]:
        module_id = module["module_name"]
        root.append(build_module_group(module, layer_map.get(module_id, [])))
    return ET.ElementTree(root)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    pipeline = None
    torch_module = None
    if not args.skip_inference:
        pipeline, torch_module = load_qwen_pipeline()

    layer_map: dict[str, list[LayerSummary]] = {}
    metadata: dict[str, Any] = {
        "manifest": args.manifest,
        "modules": {},
    }

    for module in manifest["modules"]:
        module_name = module["module_name"]
        module_dir = os.path.join(args.layers_dir, slugify(module_name))

        if args.skip_inference:
            layer_paths = existing_layers(module_dir)
            if not layer_paths:
                raise FileNotFoundError(
                    f"No precomputed layers found for module '{module_name}' in {module_dir}"
                )
        else:
            layer_paths = run_layer_separation(
                pipeline=pipeline,
                torch_module=torch_module,
                crop_path=module["crop_path"],
                output_dir=module_dir,
                layers=args.layers,
                resolution=args.resolution,
                seed=args.seed,
            )

        summaries = [
            classify_layer(Image.open(layer_path).convert("RGBA"), layer_path)
            for layer_path in layer_paths
        ]
        layer_map[module_name] = summaries
        metadata["modules"][module_name] = [asdict(summary) for summary in summaries]

    tree = build_svg_document(manifest, layer_map)
    ensure_parent(args.svg_out)
    tree.write(args.svg_out, encoding="utf-8", xml_declaration=True)

    ensure_parent(args.metadata_out)
    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"SVG written to: {args.svg_out}")
    print(f"Layer metadata written to: {args.metadata_out}")


if __name__ == "__main__":
    main()

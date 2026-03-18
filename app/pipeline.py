from __future__ import annotations

import json
import uuid
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from vectorize_single_module import build_svg, decide_layer, layer_paths


@dataclass
class JobArtifacts:
    job_id: str
    job_dir: str
    generated_image: str
    crop_manifest: str
    crops_dir: str
    layers_dir: str
    vectorized_dir: str
    bundle_zip: str
    crop_images: list[str]
    svg_files: list[str]
    log: list[str]


def jobs_root() -> Path:
    root = Path("app") / "artifacts" / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_dirs(job_id: str) -> dict[str, Path]:
    root = jobs_root() / job_id
    paths = {
        "root": root,
        "generated": root / "generated",
        "crops": root / "crops",
        "layers": root / "layers",
        "vectorized": root / "vectorized",
        "bundle": root / "bundle",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def new_job_id() -> str:
    return f"{now_stamp()}-{uuid.uuid4().hex[:8]}"


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_multiline_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    lines = wrap_text(draw, text, font, max(40, right - left - 24))
    y = top
    line_height = font.size + 8 if hasattr(font, "size") else 24
    for line in lines:
        if y + line_height > bottom:
            break
        draw.text((left, y), line, fill=fill, font=font)
        y += line_height


def render_demo_generated_image(
    paper_excerpt: str,
    caption: str,
    output_path: Path,
) -> Path:
    image = Image.new("RGBA", (1400, 900), "#f4f1ea")
    draw = ImageDraw.Draw(image)

    title_font = load_font(34)
    body_font = load_font(22)
    small_font = load_font(18)

    draw.rounded_rectangle((40, 30, 1360, 870), radius=28, fill="#fffdf8", outline="#c7b79c", width=4)
    draw.text((70, 60), "Paper Diagram Vectorization Demo", fill="#24323f", font=title_font)

    excerpt_box = (70, 120, 1330, 250)
    draw.rounded_rectangle(excerpt_box, radius=18, fill="#eef4f7", outline="#8ca6b5", width=3)
    draw.text((90, 140), "Paper excerpt", fill="#355264", font=small_font)
    draw_multiline_block(draw, paper_excerpt.strip() or "No excerpt provided.", (90, 170, 1310, 235), body_font, "#24323f")

    caption_box = (70, 275, 1330, 360)
    draw.rounded_rectangle(caption_box, radius=18, fill="#fff0de", outline="#c69054", width=3)
    draw.text((90, 295), "Caption", fill="#8b4c16", font=small_font)
    draw_multiline_block(draw, caption.strip() or "No caption provided.", (90, 322, 1310, 350), body_font, "#5a3718")

    module_boxes = {
        "Input": (110, 560, 360, 720),
        "Core_Module": (510, 510, 910, 770),
        "Output": (1080, 560, 1290, 720),
    }
    module_colors = {
        "Input": ("#edf6ff", "#6d9dc5"),
        "Core_Module": ("#fef3d8", "#c3983c"),
        "Output": ("#e8f7ea", "#6da16f"),
    }

    for name, box in module_boxes.items():
        fill, outline = module_colors[name]
        draw.rounded_rectangle(box, radius=20, fill=fill, outline=outline, width=5)
        draw.text((box[0] + 22, box[1] + 22), name.replace("_", " "), fill="#24323f", font=title_font)

    draw_multiline_block(draw, "figure crop or diagram context", (140, 620, 330, 700), body_font, "#304354")
    draw_multiline_block(draw, caption.strip() or "caption drives generation", (540, 600, 880, 745), body_font, "#5e4b21")
    draw_multiline_block(draw, "editable vector export", (1105, 620, 1270, 700), body_font, "#31553a")

    arrow_fill = "#424b57"
    draw.line((360, 640, 510, 640), fill=arrow_fill, width=8)
    draw.polygon([(510, 640), (484, 624), (484, 656)], fill=arrow_fill)
    draw.line((910, 640, 1080, 640), fill=arrow_fill, width=8)
    draw.polygon([(1080, 640), (1054, 624), (1054, 656)], fill=arrow_fill)

    image.save(output_path)
    return output_path


def save_crop_manifest(image_path: Path, output_dir: Path, modules: dict[str, tuple[int, int, int, int]]) -> Path:
    image = Image.open(image_path)
    manifest: dict[str, Any] = {
        "source_image": str(image_path),
        "image_width": image.width,
        "image_height": image.height,
        "modules": [],
    }

    for module_name, bbox in modules.items():
        left, top, right, bottom = bbox
        crop_path = output_dir / f"{module_name}.png"
        image.crop(bbox).save(crop_path)
        manifest["modules"].append(
            {
                "module_name": module_name,
                "crop_path": str(crop_path),
                "bbox": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                },
                "crop_size": {
                    "width": right - left,
                    "height": bottom - top,
                },
            }
        )

    manifest_path = output_dir / "crop_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def create_demo_layers(crop_path: Path, module_name: str, layers_dir: Path) -> list[Path]:
    crop = Image.open(crop_path).convert("RGBA")
    width, height = crop.size
    bg = Image.new("RGBA", crop.size, (255, 255, 255, 255))

    text_layer = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)
    text_font = load_font(24)
    text_draw.text((18, 18), module_name.replace("_", " "), fill="#182028", font=text_font)

    border_layer = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    border_draw = ImageDraw.Draw(border_layer)
    border_draw.rounded_rectangle((8, 8, width - 8, height - 8), radius=18, outline="#5c6b77", width=4)

    accent_layer = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    accent_draw = ImageDraw.Draw(accent_layer)
    accent_draw.rounded_rectangle((22, 58, width - 22, height - 24), radius=14, fill="#f6f0de", outline="#c4a95b", width=3)

    layers = [bg, text_layer, border_layer, accent_layer]
    saved: list[Path] = []
    for index, layer in enumerate(layers):
        path = layers_dir / f"{module_name}__layer_{index:02d}.png"
        layer.save(path)
        saved.append(path)
    return saved


def vectorize_module(crop_path: Path, layers_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    crop_image = Image.open(crop_path).convert("RGBA")
    module_name = crop_path.stem
    decisions = [decide_layer(path, crop_image, alpha_threshold=32) for path in layer_paths(str(layers_dir), module_name)]
    tree = build_svg(crop_image, decisions, alpha_threshold=32, keep_base_layer=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{module_name}.svg"
    json_path = output_dir / f"{module_name}.json"
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "crop_path": str(crop_path),
                "layers_dir": str(layers_dir),
                "layers": [asdict(decision) for decision in decisions],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return svg_path, json_path


def build_bundle(job_dir: Path, bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    zip_path = bundle_dir / "result_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(job_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, arcname=path.relative_to(job_dir))
    return zip_path


def run_demo_job(
    paper_excerpt: str,
    caption: str,
    keep_intermediate: bool = True,
) -> JobArtifacts:
    job_id = new_job_id()
    paths = job_dirs(job_id)
    log: list[str] = []

    generated_path = paths["generated"] / "generated_diagram.png"
    render_demo_generated_image(
        paper_excerpt=paper_excerpt,
        caption=caption,
        output_path=generated_path,
    )
    log.append("Generated a demo paper diagram image from the input text.")

    modules = {
        "Input": (110, 560, 360, 720),
        "Core_Module": (510, 510, 910, 770),
        "Output": (1080, 560, 1290, 720),
    }
    manifest_path = save_crop_manifest(generated_path, paths["crops"], modules)
    log.append("Created semantic module crops and a crop manifest.")

    for module in modules:
        crop_path = paths["crops"] / f"{module}.png"
        create_demo_layers(crop_path, module, paths["layers"])
    log.append("Created demo layer-separated PNGs for each module.")

    for module in modules:
        crop_path = paths["crops"] / f"{module}.png"
        vectorize_module(crop_path, paths["layers"], paths["vectorized"])
    log.append("Converted module crops into trial hybrid SVG files.")

    bundle_path = build_bundle(paths["root"], paths["bundle"])
    log.append("Packaged the generated assets into a downloadable ZIP.")
    crop_images = [str(paths["crops"] / f"{module}.png") for module in modules]
    svg_files = [str(paths["vectorized"] / f"{module}.svg") for module in modules]

    summary = JobArtifacts(
        job_id=job_id,
        job_dir=str(paths["root"]),
        generated_image=str(generated_path),
        crop_manifest=str(manifest_path),
        crops_dir=str(paths["crops"]),
        layers_dir=str(paths["layers"]),
        vectorized_dir=str(paths["vectorized"]),
        bundle_zip=str(bundle_path),
        crop_images=crop_images,
        svg_files=svg_files,
        log=log,
    )
    with open(paths["root"] / "job_summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    return summary

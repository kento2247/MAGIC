import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

import phase2_svg

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


@dataclass
class LayerDecision:
    layer_path: str
    action: str
    bbox: dict[str, int] | None
    bbox_coverage: float
    mean_alpha: float
    mean_diff_to_crop: float
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one Qwen-Image-Layered layer set into a trial hybrid SVG."
    )
    parser.add_argument(
        "--layers-dir",
        required=True,
        help="Directory containing layer PNGs. Supports both nested layer_XX.png and flat <module>__layer_XX.png layouts.",
    )
    parser.add_argument("--crop-path", required=True, help="Original cropped module PNG")
    parser.add_argument("--svg-out", required=True, help="Output SVG file")
    parser.add_argument("--metadata-out", default=None, help="Optional JSON metadata path")
    parser.add_argument("--alpha-threshold", type=int, default=32, help="Alpha threshold for visible-region bbox")
    parser.add_argument(
        "--keep-base-layer",
        action="store_true",
        help="Keep the layer that looks closest to the original crop",
    )
    return parser.parse_args()


def visible_mask(image: Image.Image, alpha_threshold: int) -> Image.Image:
    alpha = phase2_svg.alpha_channel(image)
    return alpha.point(lambda value: 255 if value > alpha_threshold else 0)


def bbox_dict(mask: Image.Image) -> dict[str, int] | None:
    bbox = mask.getbbox()
    if bbox is None:
        return None
    return {"left": bbox[0], "top": bbox[1], "right": bbox[2], "bottom": bbox[3]}


def bbox_coverage(mask: Image.Image, bbox: dict[str, int]) -> float:
    crop = mask.crop((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))
    hist = crop.histogram()
    total = sum(hist) or 1
    return sum(hist[1:]) / total


def mean_alpha(image: Image.Image) -> float:
    return ImageStat.Stat(phase2_svg.alpha_channel(image)).mean[0]


def mean_diff_to_crop(layer_image: Image.Image, crop_image: Image.Image) -> float:
    resized_crop = crop_image.convert("RGBA").resize(layer_image.size)
    diff = ImageChops.difference(layer_image.convert("RGBA"), resized_crop)
    return sum(ImageStat.Stat(diff).mean[:3]) / 3.0


def layer_paths(layers_dir: str, module_name: str | None = None) -> list[Path]:
    base = Path(layers_dir)
    nested = sorted(base.glob("layer_*.png"))
    if nested:
        return nested

    if module_name:
        flat = sorted(base.glob(f"{phase2_svg.slugify(module_name)}__layer_*.png"))
        if flat:
            return flat

    return []


def image_crop(image: Image.Image, bbox: dict[str, int]) -> Image.Image:
    return image.crop((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))


def is_full_canvas_bbox(image: Image.Image, bbox: dict[str, int]) -> bool:
    return (
        bbox["left"] == 0
        and bbox["top"] == 0
        and bbox["right"] == image.width
        and bbox["bottom"] == image.height
    )


def scale_bbox(
    bbox: dict[str, int],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> dict[str, int]:
    scale_x = dst_size[0] / src_size[0]
    scale_y = dst_size[1] / src_size[1]
    return {
        "left": round(bbox["left"] * scale_x),
        "top": round(bbox["top"] * scale_y),
        "right": round(bbox["right"] * scale_x),
        "bottom": round(bbox["bottom"] * scale_y),
    }


def likely_base_layer(
    decisions: list[LayerDecision],
    threshold: float = 80.0,
) -> str | None:
    candidates = [
        decision
        for decision in decisions
        if decision.bbox
        and decision.bbox["left"] == 0
        and decision.bbox["top"] == 0
        and decision.mean_alpha > 240
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda item: item.mean_diff_to_crop)
    if best.mean_diff_to_crop <= threshold:
        return best.layer_path
    return None


def decide_layer(
    layer_path: Path,
    crop_image: Image.Image,
    alpha_threshold: int,
) -> LayerDecision:
    image = Image.open(layer_path).convert("RGBA")
    mask = visible_mask(image, alpha_threshold)
    bbox = bbox_dict(mask)
    if bbox is None:
        return LayerDecision(
            layer_path=str(layer_path),
            action="skip_blank",
            bbox=None,
            bbox_coverage=0.0,
            mean_alpha=mean_alpha(image),
            mean_diff_to_crop=mean_diff_to_crop(image, crop_image),
            note="No visible region above threshold",
        )

    coverage = bbox_coverage(mask, bbox)
    diff_score = mean_diff_to_crop(image, crop_image)
    alpha_score = mean_alpha(image)
    width = bbox["right"] - bbox["left"]
    height = bbox["bottom"] - bbox["top"]

    if is_full_canvas_bbox(image, bbox) and alpha_score > 240 and diff_score > 100:
        action = "skip_canvas_fill"
        note = "Looks like a synthetic full-canvas fill layer"
    elif coverage > 0.92 and width > 40 and height > 40:
        action = "rect"
        note = "Dense visible region; approximated as rect"
    else:
        action = "image_crop"
        note = "Preserved as positioned raster fragment inside SVG"

    return LayerDecision(
        layer_path=str(layer_path),
        action=action,
        bbox=bbox,
        bbox_coverage=round(coverage, 4),
        mean_alpha=round(alpha_score, 2),
        mean_diff_to_crop=round(diff_score, 2),
        note=note,
    )


def add_rect(
    parent: ET.Element,
    image: Image.Image,
    bbox: dict[str, int],
    layer_id: str,
    canvas_size: tuple[int, int],
) -> None:
    scaled_bbox = scale_bbox(bbox, image.size, canvas_size)
    fill = phase2_svg.dominant_visible_color(image_crop(image, bbox)) or (220, 220, 220, 255)
    ET.SubElement(
        parent,
        f"{{{SVG_NS}}}rect",
        {
            "id": f"{layer_id}_rect",
            "x": str(scaled_bbox["left"]),
            "y": str(scaled_bbox["top"]),
            "width": str(scaled_bbox["right"] - scaled_bbox["left"]),
            "height": str(scaled_bbox["bottom"] - scaled_bbox["top"]),
            "stroke": "#333333",
            "stroke-width": "1",
            **phase2_svg.color_to_svg_attrs(fill),
        },
    )


def add_cropped_image(
    parent: ET.Element,
    image: Image.Image,
    bbox: dict[str, int],
    layer_id: str,
    canvas_size: tuple[int, int],
) -> None:
    scaled_bbox = scale_bbox(bbox, image.size, canvas_size)
    cropped = image_crop(image, bbox)
    scaled_width = max(1, scaled_bbox["right"] - scaled_bbox["left"])
    scaled_height = max(1, scaled_bbox["bottom"] - scaled_bbox["top"])
    scaled_crop = cropped.resize((scaled_width, scaled_height))
    phase2_svg.append_embedded_image(
        parent=parent,
        image=scaled_crop,
        x=scaled_bbox["left"],
        y=scaled_bbox["top"],
        width=scaled_crop.width,
        height=scaled_crop.height,
        element_id=f"{layer_id}_image",
    )


def build_svg(
    crop_image: Image.Image,
    decisions: list[LayerDecision],
    alpha_threshold: int,
    keep_base_layer: bool,
) -> ET.ElementTree:
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": str(crop_image.width),
            "height": str(crop_image.height),
            "viewBox": f"0 0 {crop_image.width} {crop_image.height}",
        },
    )
    ET.SubElement(root, f"{{{SVG_NS}}}desc").text = (
        "Trial hybrid SVG from Qwen-Image-Layered output. "
        "Dense regions become rects; other meaningful layers stay as positioned raster fragments."
    )

    group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "module"})

    base_layer = likely_base_layer(decisions)
    for decision in decisions:
        if decision.action == "skip_blank":
            continue
        if decision.action == "skip_canvas_fill":
            continue
        if not keep_base_layer and base_layer and decision.layer_path == base_layer:
            continue

        layer_path = Path(decision.layer_path)
        layer_id = phase2_svg.slugify(layer_path.stem)
        image = Image.open(layer_path).convert("RGBA")

        if decision.bbox is None:
            continue
        if decision.action == "rect":
            add_rect(group, image, decision.bbox, layer_id, crop_image.size)
        else:
            add_cropped_image(group, image, decision.bbox, layer_id, crop_image.size)

    return ET.ElementTree(root)


def main() -> None:
    args = parse_args()
    crop_image = Image.open(args.crop_path).convert("RGBA")
    module_name = Path(args.crop_path).stem

    decisions = [
        decide_layer(path, crop_image=crop_image, alpha_threshold=args.alpha_threshold)
        for path in layer_paths(args.layers_dir, module_name=module_name)
    ]
    if not decisions:
        raise FileNotFoundError(
            f"No layer PNGs found in {args.layers_dir!r} for module {module_name!r}"
        )

    tree = build_svg(
        crop_image=crop_image,
        decisions=decisions,
        alpha_threshold=args.alpha_threshold,
        keep_base_layer=args.keep_base_layer,
    )

    output_path = Path(args.svg_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    metadata_path = Path(args.metadata_out) if args.metadata_out else output_path.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "crop_path": args.crop_path,
                "layers_dir": args.layers_dir,
                "alpha_threshold": args.alpha_threshold,
                "keep_base_layer": args.keep_base_layer,
                "layers": [asdict(decision) for decision in decisions],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"SVG written to: {output_path}")
    print(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()

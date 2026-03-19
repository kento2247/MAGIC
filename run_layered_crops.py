import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import QwenImageLayeredPipeline
from PIL import Image


# Keep a single default prompt here so we can iterate on it quickly.
DEFAULT_PROMPT = (
    "Decompose this cropped academic architecture-diagram module into a small number of meaningful transparent layers "
    "for vector reconstruction. Extract all readable text onto separate text-only layers whenever possible. "
    "Separate bordered rectangles or container boxes from arrows, connector lines, and other directional marks. "
    "A box border and an arrow should not appear in the same layer unless they are physically inseparable. "
    "Do not produce empty layers, nearly blank layers, or full-canvas background-fill layers. "
    "Each output layer should contain only one semantic object family: text, box or border, arrow or connector, icon, "
    "or filled region. Preserve exact geometry, positions, sizes, colors, and transparent backgrounds. "
    "Do not merge text with shapes. Do not add, remove, simplify, or hallucinate content."
)

DEFAULT_NEGATIVE_PROMPT = (
    "empty layer, blank transparent layer, nearly blank layer, full canvas background layer, full canvas fill, "
    "text missing, unreadable text, blurry text, partial text, text fused into boxes, text fused into arrows, "
    "box and arrow in the same layer, merged components, merged connectors, duplicated arrows, broken lines, "
    "extra boxes, missing borders, decorative background, hallucinated shapes, omitted elements"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run Qwen-Image-Layered on cropped module PNGs in output/."
    )
    parser.add_argument("--input-dir", default="output", help="Directory with cropped PNG modules")
    parser.add_argument(
        "--output-dir",
        default="artifacts/phase2/layers",
        help="Directory for separated layer PNGs",
    )
    parser.add_argument("--glob", default="*.png", help="Glob pattern under --input-dir")
    parser.add_argument("--prompt", default=None, help="Override positive prompt")
    parser.add_argument("--negative-prompt", default=None, help="Override negative prompt")
    parser.add_argument("--layers", type=int, default=6, help="Number of output layers")
    parser.add_argument("--resolution", type=int, default=640, help="Qwen bucket resolution")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--seed", type=int, default=777, help="Random seed")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N images, 0 means all")
    return parser.parse_args()


def slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def resolve_prompts(args: argparse.Namespace) -> tuple[str, str]:
    prompt = args.prompt or DEFAULT_PROMPT
    negative_prompt = args.negative_prompt or DEFAULT_NEGATIVE_PROMPT
    return prompt, negative_prompt


def load_pipeline() -> QwenImageLayeredPipeline:
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script but is not available.")

    torch.cuda.empty_cache()
    print("Loading pipeline...", flush=True)
    pipeline = QwenImageLayeredPipeline.from_pretrained(
        "Qwen/Qwen-Image-Layered",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print("Enabling low-VRAM settings...", flush=True)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_attention_slicing("auto")
    pipeline.enable_sequential_cpu_offload()
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    return pipeline


def collect_inputs(input_dir: str, pattern: str, limit: int) -> list[Path]:
    paths = sorted(Path(input_dir).glob(pattern))
    paths = [path for path in paths if path.is_file()]
    if limit > 0:
        paths = paths[:limit]
    return paths


def run_one(
    pipeline: QwenImageLayeredPipeline,
    image_path: Path,
    output_dir: Path,
    prompt: str,
    negative_prompt: str,
    layers: int,
    resolution: int,
    steps: int,
    seed: int,
) -> None:
    print(f"Processing: {image_path}", flush=True)
    image = Image.open(image_path).convert("RGBA")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = slugify(image_path.stem)

    inputs = {
        "image": image,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "true_cfg_scale": 4.0,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "num_images_per_prompt": 1,
        "layers": layers,
        "resolution": resolution,
        "cfg_normalize": True,
        "use_en_prompt": True,
    }

    torch.cuda.empty_cache()
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_layers = output.images[0]

    saved_layers: list[str] = []
    print(f"Saving {len(output_layers)} layers into {output_dir}", flush=True)
    for idx, layer in enumerate(output_layers):
        layer_path = output_dir / f"{sample_name}__layer_{idx:02d}.png"
        layer.save(layer_path)
        saved_layers.append(str(layer_path))

    metadata_path = output_dir / f"{sample_name}__layers.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_image": str(image_path),
                "sample_name": sample_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "layers": saved_layers,
                "resolution": resolution,
                "steps": steps,
                "seed": seed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    prompt, negative_prompt = resolve_prompts(args)
    print(f"prompt={prompt}", flush=True)
    print(f"negative_prompt={negative_prompt}", flush=True)

    inputs = collect_inputs(args.input_dir, args.glob, args.limit)
    if not inputs:
        raise FileNotFoundError(f"No images found in {args.input_dir!r} matching {args.glob!r}")

    print(f"Found {len(inputs)} input image(s)", flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline()
    for image_path in inputs:
        run_one(
            pipeline=pipeline,
            image_path=image_path,
            output_dir=output_dir,
            prompt=prompt,
            negative_prompt=negative_prompt,
            layers=args.layers,
            resolution=args.resolution,
            steps=args.steps,
            seed=args.seed,
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

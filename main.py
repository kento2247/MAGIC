import os
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import QwenImageLayeredPipeline
from PIL import Image


def main() -> None:
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

    print("Loading input image...", flush=True)
    image = Image.open("asserts/test.png").convert("RGBA")
    inputs = {
        "image": image,
        "generator": torch.Generator(device="cuda").manual_seed(777),
        "true_cfg_scale": 4.0,
        "prompt": "Decompose this architecture diagram into region-aware layers. Put the left module and its internal contents on separate layers from the right module and its internal contents. Separate text labels, arrows, connector lines, and box shapes whenever possible. Preserve exact spatial layout and transparent backgrounds. Do not merge left and right regions.",
        "negative_prompt": "merged left and right regions, mixed components across layers, blurry text, duplicated arrows, broken lines, extra boxes",
        "num_inference_steps": 50,
        "num_images_per_prompt": 1,
        "layers": 6,
        "resolution": 640,      # Using different bucket (640, 1024) to determine the resolution. For this version, 640 is recommended
        "cfg_normalize": True,  # Whether enable cfg normalization.
        "use_en_prompt": True,  # Automatic caption language if user does not provide caption
    }

    print("Running inference...", flush=True)
    torch.cuda.empty_cache()
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]

    print(f"Saving {len(output_image)} images...", flush=True)
    for i, output_layer in enumerate(output_image):
        output_layer.save(f"outputs/test_{i}.png")

    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

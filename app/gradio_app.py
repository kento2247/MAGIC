import json
import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import run_demo_job


def crop_caption(crop_paths: list[str], index: int) -> str:
    if not crop_paths:
        return "No crops available"
    path = Path(crop_paths[index])
    return f"Crop {index + 1} / {len(crop_paths)}: `{path.stem}`"


def resolve_crop(crop_paths: list[str], svg_paths: list[str], index: int) -> tuple[str | None, str, int, str | None]:
    if not crop_paths:
        return None, "No crops available", 0, None
    safe_index = index % len(crop_paths)
    svg_path = svg_paths[safe_index] if safe_index < len(svg_paths) else None
    return crop_paths[safe_index], crop_caption(crop_paths, safe_index), safe_index, svg_path


def step_crop(crop_paths: list[str], svg_paths: list[str], index: int, delta: int):
    if not crop_paths:
        return None, "No crops available", 0, None
    next_index = (index + delta) % len(crop_paths)
    crop_path, label, safe_index, svg_path = resolve_crop(crop_paths, svg_paths, next_index)
    return crop_path, label, safe_index, svg_path


def run_ui_pipeline(paper_excerpt: str, caption: str):
    summary = run_demo_job(
        paper_excerpt=paper_excerpt,
        caption=caption,
    )
    crop_path, crop_label, crop_index, svg_path = resolve_crop(summary.crop_images, summary.svg_files, 0)

    return (
        summary.generated_image,
        crop_path,
        crop_label,
        summary.crop_images,
        summary.svg_files,
        crop_index,
        svg_path,
        json.dumps(summary.__dict__, ensure_ascii=False, indent=2),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Paper Diagram Vectorizer",
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="slate", neutral_hue="stone"),
        css="""
        .hero {padding: 22px 24px; border: 1px solid #d8c7a6; border-radius: 18px;
               background: linear-gradient(135deg, #1f3442 0%, #35586b 100%);}
        .hero h1, .hero p {color: #f7f2e8 !important; margin: 0;}
        .hero h1 {margin-bottom: 10px;}
        .hero p + p {margin-top: 10px;}
        .panel {border-radius: 16px;}
        .crop-nav button {min-width: 56px;}
        .compare-image {background: #0f1318; border-radius: 18px; padding: 12px;}
        textarea, input {font-size: 15px !important;}
        """,
    ) as demo:
        gr.Markdown(
            """
            <div class="hero">
              <h1>Paper Diagram Vectorizer</h1>
              <p>
                論文の該当箇所とキャプションを入力して生成すると、
                左に全体画像、右にクロップ画像を見比べながら結果を確認できます。
              </p>
              <p>
                いまは <b>demo pipeline</b> で最後まで通るUI骨組みです。後で paperbanana と実処理に差し替えられる構成にしています。
              </p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=6):
                excerpt = gr.Textbox(
                    label="Paper Excerpt",
                    lines=8,
                    placeholder="論文の該当箇所を貼り付けてください",
                    elem_classes=["panel"],
                )
                caption = gr.Textbox(
                    label="Image Caption",
                    lines=3,
                    placeholder="図のキャプションを入力してください",
                    elem_classes=["panel"],
                )
                generate = gr.Button("Generate", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=6):
                generated_image = gr.Image(
                    label="Generated Diagram",
                    type="filepath",
                    interactive=False,
                    height=520,
                    elem_classes=["compare-image"],
                )
            with gr.Column(scale=4):
                crop_image = gr.Image(
                    label="Cropped Module",
                    type="filepath",
                    interactive=False,
                    height=520,
                    elem_classes=["compare-image"],
                )
                crop_label = gr.Markdown("No crops available yet.")
                with gr.Row(elem_classes=["crop-nav"]):
                    prev_crop = gr.Button("◀", size="sm")
                    next_crop = gr.Button("▶", size="sm")

        with gr.Row():
            svg_file = gr.File(label="Download SVG")

        raw_summary = gr.Code(label="Job Summary JSON", language="json", visible=False)
        crop_paths_state = gr.State([])
        svg_paths_state = gr.State([])
        crop_index_state = gr.State(0)

        generate.click(
            fn=run_ui_pipeline,
            inputs=[excerpt, caption],
            outputs=[
                generated_image,
                crop_image,
                crop_label,
                crop_paths_state,
                svg_paths_state,
                crop_index_state,
                svg_file,
                raw_summary,
            ],
        )
        prev_crop.click(
            fn=lambda crop_paths, svg_paths, idx: step_crop(crop_paths, svg_paths, idx, -1),
            inputs=[crop_paths_state, svg_paths_state, crop_index_state],
            outputs=[crop_image, crop_label, crop_index_state, svg_file],
        )
        next_crop.click(
            fn=lambda crop_paths, svg_paths, idx: step_crop(crop_paths, svg_paths, idx, 1),
            inputs=[crop_paths_state, svg_paths_state, crop_index_state],
            outputs=[crop_image, crop_label, crop_index_state, svg_file],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()

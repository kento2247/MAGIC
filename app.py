import base64
import os
from io import BytesIO
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from utils.InstructionCompletion import InstructionCompletion

BASE_DIR = Path(__file__).resolve().parent
DEMO_SAMPLE_DIR = BASE_DIR / "demo_sample"

app = Flask(__name__)
instruction_completion = InstructionCompletion(os.getenv("OPENAI_API_KEY", ""))
paper_banana = None


def get_paper_banana():
    global paper_banana
    if paper_banana is None:
        from utils.PaperBanana import PaperBanana

        paper_banana = PaperBanana()
    return paper_banana


def list_demo_samples() -> list[str]:
    if not DEMO_SAMPLE_DIR.exists():
        return []
    return sorted(
        [entry.name for entry in DEMO_SAMPLE_DIR.iterdir() if entry.is_dir()],
        key=str.lower,
    )


def get_demo_sample_dir(sample_name: str) -> Path:
    if sample_name not in list_demo_samples():
        abort(404, description="Sample not found.")
    return DEMO_SAMPLE_DIR / sample_name


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_demo_sample_payload(sample_name: str) -> dict[str, str | None]:
    sample_dir = get_demo_sample_dir(sample_name)
    image_path = sample_dir / "sample.png"
    svg_path = sample_dir / "sample.svg"

    return {
        "name": sample_name,
        "description": read_text_if_exists(sample_dir / "description.md"),
        "caption": read_text_if_exists(sample_dir / "caption.txt"),
        "image_url": (
            url_for("demo_sample_image", sample_name=sample_name)
            if image_path.exists()
            else None
        ),
        "svg_url": (
            url_for("demo_sample_svg", sample_name=sample_name)
            if svg_path.exists()
            else None
        ),
    }


def image_to_data_url(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.get("/")
def index():
    demo_samples = list_demo_samples()
    return render_template(
        "index.html",
        demo_samples=demo_samples,
        initial_sample=demo_samples[0] if demo_samples else None,
    )


@app.get("/api/demo-sample")
def get_demo_sample():
    sample_name = (request.args.get("name") or "").strip()
    if not sample_name:
        return jsonify({"error": "name is required"}), 400
    return jsonify(build_demo_sample_payload(sample_name))


@app.post("/api/review")
def review_instruction():
    data = request.get_json(silent=True) or {}
    instruction = str(data.get("instruction") or "").strip()

    if not instruction:
        return jsonify({"error": "instruction is required"}), 400

    review = instruction_completion.check_sufficiently_detailed(instruction)
    response: dict[str, str | bool | None] = {
        "is_sufficiently_detailed": review.is_sufficiently_detailed,
        "additional_questions": review.additional_questions,
    }

    if review.is_sufficiently_detailed:
        converted = instruction_completion.convert_text_to_md(instruction)
        response.update(
            {
                "assistant_message": "説明は図示可能な粒度です。右側の要約とキャプションを必要に応じて編集してください。",
                "markdown_text": converted.markdown_text,
                "figure_caption": converted.figure_caption,
            }
        )
    else:
        response["assistant_message"] = (
            review.additional_questions or "不足情報を補足してください。"
        )

    return jsonify(response)


@app.post("/api/generate")
def generate_figure():
    data = request.get_json(silent=True) or {}
    description = str(data.get("description") or "").strip()
    caption = str(data.get("caption") or "").strip()
    mode = str(data.get("mode") or "new").strip() or "new"

    if not description:
        return jsonify({"error": "description is required"}), 400
    if not caption:
        return jsonify({"error": "caption is required"}), 400

    is_demo = mode != "new"

    try:
        image = get_paper_banana().generate(
            description=description,
            caption=caption,
            is_demo=is_demo,
        )
    except Exception as exc:
        return jsonify({"error": str(exc), "is_demo": is_demo}), 500

    return jsonify(
        {
            "image_data_url": image_to_data_url(image),
            "is_demo": is_demo,
        }
    )


@app.get("/demo-sample-image/<path:sample_name>")
def demo_sample_image(sample_name: str):
    sample_dir = get_demo_sample_dir(sample_name)
    image_path = sample_dir / "sample.png"
    if not image_path.exists():
        abort(404, description="Image not found.")
    return send_from_directory(sample_dir, image_path.name)


@app.get("/demo-sample-svg/<path:sample_name>")
def demo_sample_svg(sample_name: str):
    sample_dir = get_demo_sample_dir(sample_name)
    svg_path = sample_dir / "sample.svg"
    if not svg_path.exists():
        abort(404, description="SVG not found.")
    return send_from_directory(sample_dir, svg_path.name)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "8000")),
        debug=True,
    )

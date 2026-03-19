import gradio as gr

from utils.InstructionCompletion import (
    generate_figure,
    initial_chat_history,
    new_session_state,
    render_status,
    reset_session,
    submit_message,
)

APP_THEME = gr.themes.Soft(
    primary_hue="amber", secondary_hue="slate", neutral_hue="stone"
)
APP_CSS = """
body, .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(245, 194, 108, 0.18), transparent 32%),
        linear-gradient(180deg, #f5efe4 0%, #efe5d4 100%);
    font-family: "Hiragino Sans", "Yu Gothic", "Noto Sans JP", sans-serif;
}
.hero {
    padding: 24px 26px;
    border: 1px solid #d9b77c;
    border-radius: 22px;
    background: linear-gradient(135deg, #213b4c 0%, #31586e 55%, #9d6b2f 100%);
    box-shadow: 0 16px 40px rgba(33, 59, 76, 0.16);
}
.hero h1, .hero p {
    color: #fff8ef !important;
    margin: 0;
}
.hero h1 {
    margin-bottom: 10px;
    letter-spacing: 0.02em;
}
.hero p + p {
    margin-top: 10px;
}
.panel {
    border-radius: 18px;
}
.status-panel {
    padding: 14px 16px;
    border: 1px solid #d8c7a6;
    border-radius: 16px;
    background: rgba(255, 250, 241, 0.88);
}
.error-panel {
    padding: 14px 16px;
    border: 1px solid #d67a58;
    border-radius: 16px;
    background: #fff0ea;
}
textarea, input {
    font-size: 15px !important;
}
"""


def error_update(state: dict[str, object] | None):
    message = str((state or {}).get("error_message", "")).strip()
    if not message:
        return gr.update(value="", visible=False)
    return gr.update(value=f"**OpenAI Error**\n\n{message}", visible=True)


def handle_submit(
    user_message: str,
    chat_history: list[dict[str, str]] | None,
    state: dict[str, object] | None,
    current_markdown: str,
    current_caption: str,
):
    result = submit_message(
        user_message, chat_history, state, current_markdown, current_caption
    )
    return (
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        error_update(result[3]),
        result[5],
        None,
    )


def handle_reset():
    result = reset_session()
    return (
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        error_update(result[3]),
        result[5],
        None,
    )


def handle_generate(
    current_markdown: str,
    current_caption: str,
    state: dict[str, object] | None,
):
    image, next_state, status = generate_figure(
        current_markdown=current_markdown,
        current_caption=current_caption,
        state=state,
    )
    return image, next_state, status, error_update(next_state)


def build_app() -> gr.Blocks:
    initial_state = new_session_state()
    with gr.Blocks(title="Instruction Completion") as demo:
        gr.HTML(
            """
            <div class="hero">
                <h1>Instruction Completion</h1>
                <p>論文の手法説明を図示できる粒度まで整理し、不足情報は対話で補完します。</p>
                <p>整理済み Markdown と図キャプションは右側でそのまま編集できます。</p>
            </div>
            """
        )
        error_box = gr.Markdown("", visible=False, elem_classes=["error-panel"])

        with gr.Row():
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    value=initial_chat_history(),
                    label="Chat",
                    height=520,
                    elem_classes=["panel"],
                )
                chat_input = gr.Textbox(
                    label="Message",
                    lines=6,
                    placeholder="論文の手法説明や追加情報を入力してください",
                    elem_classes=["panel"],
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    reset = gr.Button("Reset")

            with gr.Column(scale=6):
                organized_markdown = gr.Textbox(
                    label="整理された説明 (Markdown)",
                    lines=18,
                    placeholder="OpenAI が整理した説明がここに表示されます",
                    elem_classes=["panel"],
                )
                caption = gr.Textbox(
                    label="図キャプション",
                    lines=6,
                    placeholder="ここにキャプション案が表示されます",
                    elem_classes=["panel"],
                )
                generate = gr.Button("Generate Figure", variant="secondary")
                generated_image = gr.Image(
                    label="PaperBanana Output",
                    type="pil",
                    height=240,
                    elem_classes=["panel"],
                )
                status = gr.Markdown(
                    render_status(initial_state), elem_classes=["status-panel"]
                )

        session_state = gr.State(initial_state)

        send.click(
            fn=handle_submit,
            inputs=[
                chat_input,
                chatbot,
                session_state,
                organized_markdown,
                caption,
            ],
            outputs=[
                chatbot,
                organized_markdown,
                caption,
                session_state,
                status,
                error_box,
                chat_input,
                generated_image,
            ],
        )
        chat_input.submit(
            fn=handle_submit,
            inputs=[
                chat_input,
                chatbot,
                session_state,
                organized_markdown,
                caption,
            ],
            outputs=[
                chatbot,
                organized_markdown,
                caption,
                session_state,
                status,
                error_box,
                chat_input,
                generated_image,
            ],
        )
        reset.click(
            fn=handle_reset,
            outputs=[
                chatbot,
                organized_markdown,
                caption,
                session_state,
                status,
                error_box,
                chat_input,
                generated_image,
            ],
        )
        generate.click(
            fn=handle_generate,
            inputs=[organized_markdown, caption, session_state],
            outputs=[generated_image, session_state, status, error_box],
        )

    return demo


app = build_app()


def main() -> None:
    app.launch(theme=APP_THEME, css=APP_CSS)


if __name__ == "__main__":
    main()

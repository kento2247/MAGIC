import gradio as gr

from utils.InstructionCompletion import (
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
        .hero {padding: 22px 24px; border: 1px solid #d8c7a6; border-radius: 18px;
               background: linear-gradient(135deg, #203746 0%, #36596d 100%);}
        .hero h1, .hero p {color: #f7f2e8 !important; margin: 0;}
        .hero h1 {margin-bottom: 10px;}
        .hero p + p {margin-top: 10px;}
        .panel {border-radius: 16px;}
        .status-panel {padding: 14px 16px; border: 1px solid #d8c7a6; border-radius: 16px; background: #fffaf1;}
        .error-panel {padding: 14px 16px; border: 1px solid #d67a58; border-radius: 16px; background: #fff0ea;}
        textarea, input {font-size: 15px !important;}
        """


def error_update(state: dict[str, object] | None):
    message = str((state or {}).get("error_message", "")).strip()
    if not message:
        return gr.update(value="", visible=False)
    return gr.update(value=f"**Gemini Error**\n\n{message}", visible=True)


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
        result[5],
        result[6],
        error_update(result[5]),
        result[7],
    )


def handle_reset():
    result = reset_session()
    return (
        result[0],
        result[1],
        result[2],
        result[5],
        result[6],
        error_update(result[5]),
        result[7],
    )


def build_app() -> gr.Blocks:
    initial_state = new_session_state()
    with gr.Blocks(title="Instruction Completion") as demo:
        error_box = gr.Markdown("", visible=False, elem_classes=["error-panel"])

        with gr.Row():
            with gr.Column(scale=7):
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

            with gr.Column(scale=5):
                caption = gr.Textbox(
                    label="図キャプション",
                    lines=14,
                    placeholder="ここにキャプション案が表示されます",
                    elem_classes=["panel"],
                )
                status = gr.Markdown(
                    render_status(initial_state), elem_classes=["status-panel"]
                )

        session_state = gr.State(initial_state)
        organized_markdown_state = gr.State("")

        send.click(
            fn=handle_submit,
            inputs=[
                chat_input,
                chatbot,
                session_state,
                organized_markdown_state,
                caption,
            ],
            outputs=[
                chatbot,
                organized_markdown_state,
                caption,
                session_state,
                status,
                error_box,
                chat_input,
            ],
        )
        chat_input.submit(
            fn=handle_submit,
            inputs=[
                chat_input,
                chatbot,
                session_state,
                organized_markdown_state,
                caption,
            ],
            outputs=[
                chatbot,
                organized_markdown_state,
                caption,
                session_state,
                status,
                error_box,
                chat_input,
            ],
        )
        reset.click(
            fn=handle_reset,
            outputs=[
                chatbot,
                organized_markdown_state,
                caption,
                session_state,
                status,
                error_box,
                chat_input,
            ],
        )

    return demo


app = build_app()


def main() -> None:
    app.launch(theme=APP_THEME, css=APP_CSS)


if __name__ == "__main__":
    main()

import json
import os
from functools import lru_cache
from typing import Any

from openai import OpenAI

from utils.PaperBanana import PaperBanana

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = """
あなたは、論文の手法説明を図示可能なレベルまで整理するアシスタントです。
目的は、論文の手法をあとから図に起こせるように、説明の充足度を判定し、
足りない情報だけをユーザに質問することです。

必ず JSON オブジェクトのみを返してください。キーは以下に固定します。
- organized_markdown: str
- is_sufficient: bool
- missing_information: list[str]
- follow_up_questions: list[str]
- caption: str

判定ルール:
- 図示に必要な構成要素、入出力、処理順序、関係性、分岐、繰り返し、重要ラベルが十分にわかるなら is_sufficient=true。
- 情報が足りない場合は is_sufficient=false にし、missing_information と follow_up_questions を具体的に返す。
- 不明な情報を勝手に補わない。不確実な点は organized_markdown に「不明」と明記する。
- organized_markdown は日本語の Markdown で、少なくとも以下の節を含める:
  1. 手法の目的
  2. 主要モジュール
  3. 入出力
  4. 処理フロー
  5. 図示に必要な視覚要素
  6. 不明点・要確認点
- caption は is_sufficient=true のときだけ自然な図キャプションを返し、不十分な場合は空文字でよい。
- follow_up_questions は、そのままユーザに見せられる日本語の質問文にする。
""".strip()


def initial_chat_history() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                "論文の手法説明を入力してください。図示に必要な情報が足りなければ、"
                "不足箇所だけ追加で質問します。"
            ),
        }
    ]


def new_session_state() -> dict[str, Any]:
    return {
        "phase": "awaiting_description",
        "analysis_round": 0,
        "is_sufficient": False,
        "organized_markdown": "",
        "caption": "",
        "missing_information": [],
        "follow_up_questions": [],
        "paper_banana_status": "未生成",
        "error_message": "",
    }


def render_status(state: dict[str, Any] | None) -> str:
    session = _coerce_state(state)
    phase_labels = {
        "awaiting_description": "初回説明待ち",
        "awaiting_details": "追加情報待ち",
        "ready": "編集・生成可能",
        "error": "エラー",
    }
    lines = [
        "### Session Status",
        f"- フェーズ: {phase_labels.get(str(session['phase']), '不明')}",
        f"- 判定: {'図示可能' if session['is_sufficient'] else '追加情報が必要'}",
        f"- 対話ラウンド: {int(session['analysis_round'])}",
        f"- PaperBanana: {session['paper_banana_status']}",
    ]
    missing_information = _string_list(session.get("missing_information"))
    if missing_information:
        lines.append("")
        lines.append("不足している情報:")
        lines.extend(f"- {item}" for item in missing_information[:5])
    return "\n".join(lines)


def reset_session() -> tuple[
    list[dict[str, str]],
    str,
    str,
    dict[str, Any],
    str,
    str,
]:
    session = new_session_state()
    return (
        initial_chat_history(),
        "",
        "",
        session,
        render_status(session),
        "",
    )


def submit_message(
    user_message: str,
    chat_history: list[dict[str, str]] | None,
    state: dict[str, Any] | None,
    current_markdown: str,
    current_caption: str,
) -> tuple[
    list[dict[str, str]],
    str,
    str,
    dict[str, Any],
    str,
    str,
]:
    session = _coerce_state(state)
    history = _coerce_history(chat_history)
    session["error_message"] = ""
    session["organized_markdown"] = (current_markdown or "").strip()
    session["caption"] = (current_caption or "").strip()

    message = (user_message or "").strip()
    if not message:
        session["error_message"] = (
            "メッセージが空です。手法の説明か追加情報を入力してください。"
        )
        return (
            history,
            session["organized_markdown"],
            session["caption"],
            session,
            render_status(session),
            "",
        )

    history.append({"role": "user", "content": message})

    try:
        assessment = _assess_instruction(
            history=history,
            current_markdown=session["organized_markdown"],
            current_caption=session["caption"],
        )
    except Exception as exc:
        session["phase"] = "error"
        session["error_message"] = str(exc)
        history.append(
            {
                "role": "assistant",
                "content": (
                    "OpenAI API の呼び出しに失敗しました。環境変数 `OPENAI_API_KEY` と"
                    " モデル設定を確認してください。"
                ),
            }
        )
        return (
            history,
            session["organized_markdown"],
            session["caption"],
            session,
            render_status(session),
            "",
        )

    session["analysis_round"] = int(session["analysis_round"]) + 1
    session["organized_markdown"] = assessment["organized_markdown"]
    session["is_sufficient"] = assessment["is_sufficient"]
    session["missing_information"] = assessment["missing_information"]
    session["follow_up_questions"] = assessment["follow_up_questions"]
    session["error_message"] = ""

    if session["is_sufficient"]:
        session["phase"] = "ready"
        session["caption"] = assessment["caption"] or session["caption"]
        session["paper_banana_status"] = "未生成"
        history.append(
            {
                "role": "assistant",
                "content": (
                    "図示に必要な情報が揃いました。右側の Markdown とキャプションを"
                    "必要に応じて編集し、`Generate Figure` を押してください。"
                ),
            }
        )
    else:
        session["phase"] = "awaiting_details"
        history.append(
            {
                "role": "assistant",
                "content": _format_follow_up_message(
                    assessment["follow_up_questions"],
                    assessment["missing_information"],
                ),
            }
        )

    return (
        history,
        session["organized_markdown"],
        session["caption"],
        session,
        render_status(session),
        "",
    )


def generate_figure(
    current_markdown: str,
    current_caption: str,
    state: dict[str, Any] | None,
) -> tuple[object | None, dict[str, Any], str]:
    session = _coerce_state(state)
    session["error_message"] = ""
    session["organized_markdown"] = (current_markdown or "").strip()
    session["caption"] = (current_caption or "").strip()

    if not session["organized_markdown"]:
        session["error_message"] = (
            "整理済みの説明が空です。まずチャットで説明を整理してください。"
        )
        return None, session, render_status(session)

    if not session["caption"]:
        session["error_message"] = (
            "キャプションが空です。キャプションを入力してください。"
        )
        return None, session, render_status(session)

    paper_banana = PaperBanana()
    image = paper_banana.generate(
        description=session["organized_markdown"],
        caption=session["caption"],
    )
    session["paper_banana_status"] = "生成済み"
    return image, session, render_status(session)


def _assess_instruction(
    history: list[dict[str, str]],
    current_markdown: str,
    current_caption: str,
) -> dict[str, Any]:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_assessment_prompt(
                    history=history,
                    current_markdown=current_markdown,
                    current_caption=current_caption,
                ),
            },
        ],
    )
    content = response.choices[0].message.content or ""
    return _normalize_assessment(_extract_json(content))


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "環境変数 `OPENAI_API_KEY` が設定されていません。"
            " `export OPENAI_API_KEY=...` を実行してから起動してください。"
        )

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    return OpenAI(api_key=api_key, base_url=base_url)


def _build_assessment_prompt(
    history: list[dict[str, str]],
    current_markdown: str,
    current_caption: str,
) -> str:
    transcript = []
    for message in history:
        role = str(message.get("role", "user")).upper()
        content = str(message.get("content", "")).strip()
        if content:
            transcript.append(f"{role}: {content}")

    transcript_block = "\n\n".join(transcript)
    markdown_block = current_markdown or "(未入力)"
    caption_block = current_caption or "(未入力)"
    return f"""
以下はユーザとの対話履歴です。

{transcript_block}

現在ユーザが保持している整理済み Markdown:
{markdown_block}

現在ユーザが保持している図キャプション:
{caption_block}

この内容をもとに、図示可能なレベルまで説明が十分か判定してください。
十分でない場合は、図に起こすために最低限必要な情報だけを質問してください。
""".strip()


def _normalize_assessment(payload: dict[str, Any]) -> dict[str, Any]:
    organized_markdown = str(payload.get("organized_markdown", "")).strip()
    is_sufficient = bool(payload.get("is_sufficient", False))
    missing_information = _string_list(payload.get("missing_information"))
    follow_up_questions = _string_list(payload.get("follow_up_questions"))
    caption = str(payload.get("caption", "")).strip() if is_sufficient else ""

    if not organized_markdown:
        raise RuntimeError("OpenAI の応答に `organized_markdown` が含まれていません。")

    if not is_sufficient and not follow_up_questions:
        follow_up_questions = [
            f"{index}. {item} を教えてください。"
            for index, item in enumerate(missing_information, start=1)
        ]

    return {
        "organized_markdown": organized_markdown,
        "is_sufficient": is_sufficient,
        "missing_information": missing_information,
        "follow_up_questions": follow_up_questions,
        "caption": caption,
    }


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI の JSON 応答を解釈できませんでした。") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("OpenAI の応答が JSON オブジェクトではありませんでした。")
    return payload


def _coerce_state(state: dict[str, Any] | None) -> dict[str, Any]:
    session = new_session_state()
    if isinstance(state, dict):
        session.update(state)
    session["missing_information"] = _string_list(session.get("missing_information"))
    session["follow_up_questions"] = _string_list(session.get("follow_up_questions"))
    session["organized_markdown"] = str(session.get("organized_markdown", "")).strip()
    session["caption"] = str(session.get("caption", "")).strip()
    session["paper_banana_status"] = (
        str(session.get("paper_banana_status", "未生成")).strip() or "未生成"
    )
    session["error_message"] = str(session.get("error_message", "")).strip()
    return session


def _coerce_history(
    chat_history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if not chat_history:
        return initial_chat_history()

    normalized_history: list[dict[str, str]] = []
    for message in chat_history:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip() or "assistant"
        content = str(message.get("content", "")).strip()
        if content:
            normalized_history.append({"role": role, "content": content})
    return normalized_history or initial_chat_history()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _format_follow_up_message(
    questions: list[str],
    missing_information: list[str],
) -> str:
    lines = [
        "図示にはもう少し情報が必要です。以下を教えてください。",
    ]
    if missing_information:
        lines.append("")
        lines.append("不足情報:")
        lines.extend(f"- {item}" for item in missing_information)
    if questions:
        lines.append("")
        lines.append("質問:")
        lines.extend(f"- {question}" for question in questions)
    return "\n".join(lines)

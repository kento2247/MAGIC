from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None


@dataclass(slots=True)
class ReviewResult:
    is_sufficiently_detailed: bool
    additional_questions: str | None = None


@dataclass(slots=True)
class ConvertResult:
    markdown_text: str
    figure_caption: str


class InstructionCompletion:
    def __init__(self, OPENAI_API_KEY: str | None = None, model_name: str = "gpt-5"):
        api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.temperature = float(os.getenv("INSTRUCTION_COMPLETION_TEMPERATURE", "0.2"))
        self.client = OpenAI(api_key=api_key) if api_key and OpenAI is not None else None

    def check_sufficiently_detailed(self, instruction: str) -> ReviewResult:
        """
        Check if the instruction is sufficiently detailed.
        Returns a ReviewResult object containing the evaluation.
        """
        cleaned_instruction = instruction.strip()
        if not cleaned_instruction:
            return ReviewResult(
                is_sufficiently_detailed=False,
                additional_questions=self._build_missing_questions(
                    has_input_output=False,
                    has_flow=False,
                    has_components=False,
                    has_connections=False,
                    is_empty=True,
                ),
            )

        system_prompt = """
You help turn paper method descriptions into figure-ready specifications.
Judge whether the input contains enough information to draw a method diagram.

Treat the description as sufficiently detailed only if it clearly includes most of:
- the main input and output
- the important modules or stages
- the order of processing / data flow
- key relationships such as branching, fusion, skip connections, iteration, or supervision

If the description is not sufficient, ask only the minimum follow-up questions needed.
Return JSON with exactly this schema:
{
  "is_sufficiently_detailed": true,
  "additional_questions": null
}

Rules:
- When information is sufficient, set additional_questions to null.
- When information is insufficient, set additional_questions to a concise Japanese bullet list.
- Do not include any text outside the JSON object.
""".strip()

        user_prompt = f"Method description:\n{cleaned_instruction}"

        try:
            payload = self._chat_json(system_prompt, user_prompt)
            is_detailed = self._coerce_bool(payload["is_sufficiently_detailed"])
            questions = payload.get("additional_questions")
            if questions is not None:
                questions = str(questions).strip() or None
            return ReviewResult(
                is_sufficiently_detailed=is_detailed,
                additional_questions=None if is_detailed else questions,
            )
        except Exception:
            return self._fallback_review(cleaned_instruction)

    def convert_text_to_md(self, instruction: str) -> ConvertResult:
        """
        Convert the instruction text to markdown format.
        Returns a ConvertResult object containing the converted text and figure caption.
        """
        cleaned_instruction = instruction.strip()
        if not cleaned_instruction:
            return ConvertResult(
                markdown_text="# 手法概要\n\n入力がまだありません。",
                figure_caption="手法概要図",
            )

        system_prompt = """
You convert paper method descriptions into markdown that is ready for figure generation.
Write in Japanese and organize the content so that a diagram generator or human designer can directly use it.

Return JSON with exactly this schema:
{
  "markdown_text": "string",
  "figure_caption": "string"
}

Requirements:
- markdown_text must be valid markdown.
- Prefer these sections when the information exists:
  1. 手法の目的
  2. 入力
  3. 主要モジュール
  4. 処理フロー
  5. 図示のポイント
- Preserve the meaning of the user's description and do not invent technical details.
- If some detail is unclear, explicitly write "明示されていない".
- figure_caption must be a short Japanese caption suitable for a paper figure.
- Do not include any text outside the JSON object.
""".strip()

        user_prompt = f"Method description:\n{cleaned_instruction}"

        try:
            payload = self._chat_json(system_prompt, user_prompt)
            markdown_text = str(payload["markdown_text"]).strip()
            figure_caption = str(payload["figure_caption"]).strip()
            return ConvertResult(
                markdown_text=markdown_text,
                figure_caption=figure_caption,
            )
        except Exception:
            return self._fallback_convert(cleaned_instruction)

    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("OpenAI client is not configured.")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""
        return self._parse_json_object(content)

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        raw_text = text.strip()
        if not raw_text:
            raise ValueError("OpenAI returned empty content.")

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match is None:
                raise
            parsed = json.loads(match.group(0))

        if not isinstance(parsed, dict):
            raise ValueError("OpenAI response is not a JSON object.")
        return parsed

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
        return bool(value)

    def _fallback_review(self, instruction: str) -> ReviewResult:
        lowered = instruction.lower()
        has_input_output = any(
            token in lowered for token in ("input", "output", "入力", "出力")
        )
        has_flow = any(
            token in instruction
            for token in ("->", "→", "その後", "次に", "続いて", "最後に")
        )
        has_components = (
            sum(
                token in lowered
                for token in (
                    "module",
                    "block",
                    "encoder",
                    "decoder",
                    "attention",
                    "fusion",
                    "head",
                    "backbone",
                    "branch",
                    "network",
                    "layer",
                    "モジュール",
                    "層",
                    "段階",
                )
            )
            >= 2
        )
        has_connections = any(
            token in lowered
            for token in (
                "concat",
                "concatenate",
                "skip",
                "residual",
                "fusion",
                "merge",
                "branch",
                "接続",
                "結合",
                "分岐",
                "融合",
                "統合",
                "反復",
                "繰り返し",
            )
        )

        is_detailed = (
            len(instruction) >= 120
            and sum((has_input_output, has_flow, has_components, has_connections)) >= 3
        )
        if is_detailed:
            return ReviewResult(
                is_sufficiently_detailed=True, additional_questions=None
            )

        return ReviewResult(
            is_sufficiently_detailed=False,
            additional_questions=self._build_missing_questions(
                has_input_output=has_input_output,
                has_flow=has_flow,
                has_components=has_components,
                has_connections=has_connections,
                is_empty=False,
            ),
        )

    def _build_missing_questions(
        self,
        *,
        has_input_output: bool,
        has_flow: bool,
        has_components: bool,
        has_connections: bool,
        is_empty: bool,
    ) -> str:
        questions: list[str] = []

        if is_empty:
            questions.append(
                "- まず、手法全体が何を入力として受け取り、何を出力するのかを教えてください。"
            )
            questions.append(
                "- 主要なモジュールを順番に列挙し、それぞれの役割を簡潔に教えてください。"
            )
            questions.append("- モジュール同士がどう接続されるかを教えてください。")
            return "\n".join(questions)

        if not has_input_output:
            questions.append(
                "- 入力と出力を明確にしてください。画像、テキスト、特徴量など何を受け取り何を返しますか。"
            )
        if not has_components:
            questions.append("- 主要なモジュールや処理段階を順番に列挙してください。")
        if not has_flow:
            questions.append(
                "- 各モジュールがどの順番で動くのか、処理フローを説明してください。"
            )
        if not has_connections:
            questions.append(
                "- 分岐、結合、残差接続、反復など、モジュール間の関係を教えてください。"
            )

        if not questions:
            questions.append("- 図に入れるべき重要要素をもう少し具体化してください。")

        return "\n".join(questions)

    def _fallback_convert(self, instruction: str) -> ConvertResult:
        sentences = [
            sentence.strip(" -\t")
            for sentence in re.split(r"(?:\n+|(?<=[。．.!?])\s+)", instruction)
            if sentence.strip()
        ]
        overview = sentences[0] if sentences else instruction
        flow_lines = (
            "\n".join(f"- {sentence}" for sentence in sentences) or "- 明示されていない"
        )
        components = self._extract_component_lines(instruction)

        markdown_text = "\n".join(
            [
                "# 手法概要",
                "",
                "## 手法の目的",
                overview,
                "",
                "## 入力",
                (
                    "- 明示されていない"
                    if "入力" not in instruction and "input" not in instruction.lower()
                    else f"- {instruction}"
                ),
                "",
                "## 主要モジュール",
                components,
                "",
                "## 処理フロー",
                flow_lines,
                "",
                "## 図示のポイント",
                "- 主要な処理ブロックを左から右、または上から下の順に配置する",
                "- 分岐や結合がある場合は矢印で明示する",
                "- 学習専用の要素がある場合は推論フローと区別して示す",
            ]
        )

        caption_body = re.sub(r"\s+", " ", overview).strip("。． ")
        if not caption_body:
            caption_body = "手法概要"
        figure_caption = f"{caption_body}を示す概略図。"

        return ConvertResult(
            markdown_text=markdown_text,
            figure_caption=figure_caption,
        )

    def _extract_component_lines(self, instruction: str) -> str:
        component_patterns = (
            r"[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*",
            r"[ぁ-んァ-ン一-龥A-Za-z0-9_-]*モジュール",
            r"[ぁ-んァ-ン一-龥A-Za-z0-9_-]*ブロック",
            r"[ぁ-んァ-ン一-龥A-Za-z0-9_-]*層",
        )
        found: list[str] = []

        for pattern in component_patterns:
            for match in re.findall(pattern, instruction):
                value = match.strip()
                if len(value) < 2:
                    continue
                if value not in found:
                    found.append(value)

        if not found:
            return "- 明示されていない"
        return "\n".join(f"- {name}" for name in found[:8])

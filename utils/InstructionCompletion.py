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


@dataclass(slots=True)
class ReviewSignals:
    has_input: bool
    has_output: bool
    component_names: list[str]
    has_flow: bool
    flow_marker_count: int
    has_connections: bool
    non_space_length: int

    @property
    def component_count(self) -> int:
        return len(self.component_names)


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
                    has_input=False,
                    has_output=False,
                    component_count=0,
                    has_flow=False,
                    has_connections=False,
                    has_minimum_substance=False,
                    is_empty=True,
                ),
            )

        signals = self._analyze_instruction(cleaned_instruction)
        deterministic_review = self._review_from_signals(signals)
        if not deterministic_review.is_sufficiently_detailed:
            return deterministic_review

        system_prompt = """
You help turn paper method descriptions into figure-ready specifications.
Judge whether the input contains enough information to draw a method diagram.

Treat the description as sufficiently detailed only if all of the following are clear:
- the main input
- the main output
- at least two important modules or stages
- the processing order / data flow

Also check whether important relationships are described when they exist, such as:
- branching or merging
- fusion or concatenation
- skip/residual connections
- iteration / refinement loops
- supervision / losses used in the figure

Return false when the text is only a high-level summary, only names a model family, or could map to many different diagrams.
If any required item is missing or ambiguous, return false.

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

        detected_components = ", ".join(signals.component_names[:8]) or "none"
        user_prompt = "\n".join(
            [
                f"Method description:\n{cleaned_instruction}",
                "",
                "Deterministic checklist from the application:",
                f"- explicit input: {signals.has_input}",
                f"- explicit output: {signals.has_output}",
                f"- detected component count: {signals.component_count}",
                f"- detected components: {detected_components}",
                f"- has flow markers: {signals.has_flow}",
                f"- has relationship detail: {signals.has_connections}",
                "",
                "Use the description itself as the source of truth. The checklist is only supporting context.",
            ]
        )

        try:
            payload = self._chat_json(system_prompt, user_prompt)
            is_detailed = self._coerce_bool(payload["is_sufficiently_detailed"])
            questions = payload.get("additional_questions")
            if questions is not None:
                questions = str(questions).strip() or None
            if not is_detailed and not questions:
                questions = deterministic_review.additional_questions
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
        return self._review_from_signals(self._analyze_instruction(instruction))

    def _review_from_signals(self, signals: ReviewSignals) -> ReviewResult:
        has_components = signals.component_count >= 2
        has_minimum_substance = (
            signals.non_space_length >= 80
            or signals.component_count >= 3
            or signals.flow_marker_count >= 2
        )
        has_structure_detail = (
            signals.has_connections
            or signals.component_count >= 3
            or signals.flow_marker_count >= 2
        )

        is_detailed = (
            signals.has_input
            and signals.has_output
            and has_components
            and signals.has_flow
            and has_minimum_substance
            and has_structure_detail
        )
        if is_detailed:
            return ReviewResult(
                is_sufficiently_detailed=True, additional_questions=None
            )

        return ReviewResult(
            is_sufficiently_detailed=False,
            additional_questions=self._build_missing_questions(
                has_input=signals.has_input,
                has_output=signals.has_output,
                component_count=signals.component_count,
                has_flow=signals.has_flow,
                has_connections=signals.has_connections,
                has_minimum_substance=has_minimum_substance,
                is_empty=False,
            ),
        )

    def _analyze_instruction(self, instruction: str) -> ReviewSignals:
        normalized = re.sub(r"\s+", " ", instruction).strip()
        lowered = normalized.lower()
        component_names = self._extract_component_names(instruction)
        flow_marker_count = self._count_flow_markers(instruction)

        has_input = self._contains_pattern(
            lowered,
            (
                r"\binput(?:s)?\b",
                r"\breceiv(?:e|es|ed|ing)\b",
                r"\bfeed(?:s|ing)?\b",
                r"入力",
                r"受け取",
                r"与えられ",
            ),
        )
        has_output = self._contains_pattern(
            lowered,
            (
                r"\boutput(?:s)?\b",
                r"\bpredict(?:s|ed|ion|ions)?\b",
                r"\bgenerat(?:e|es|ed|ing)\b",
                r"\bproduc(?:e|es|ed|ing)\b",
                r"\breturn(?:s|ed|ing)?\b",
                r"出力",
                r"予測",
                r"生成",
                r"推定",
                r"返",
            ),
        )
        has_connections = self._contains_pattern(
            lowered,
            (
                r"\bconcat(?:enat(?:e|es|ed|ing))?\b",
                r"\bskip\b",
                r"\bresidual\b",
                r"\bfusion\b",
                r"\bmerge\b",
                r"\bbranch(?:es|ing)?\b",
                r"\bloop\b",
                r"\biterat(?:e|es|ed|ion|ive)\b",
                r"\bsupervis(?:e|ed|ion)\b",
                r"\bloss(?:es)?\b",
                r"接続",
                r"結合",
                r"分岐",
                r"融合",
                r"統合",
                r"反復",
                r"繰り返し",
                r"損失",
                r"監督",
            ),
        )

        return ReviewSignals(
            has_input=has_input,
            has_output=has_output,
            component_names=component_names,
            has_flow=flow_marker_count >= 1,
            flow_marker_count=flow_marker_count,
            has_connections=has_connections,
            non_space_length=len(re.sub(r"\s+", "", normalized)),
        )

    def _count_flow_markers(self, instruction: str) -> int:
        sequence_patterns = (
            r"->",
            r"→",
            r"\bthen\b",
            r"\bnext\b",
            r"\bfollowed by\b",
            r"\bafter that\b",
            r"\bfinally\b",
            r"まず",
            r"次に",
            r"続いて",
            r"その後",
            r"そのあと",
            r"最後に",
            r"最終的に",
        )
        count = 0
        for pattern in sequence_patterns:
            count += len(re.findall(pattern, instruction, flags=re.IGNORECASE))

        ordered_lines = len(
            re.findall(r"(?m)^\s*(?:\d+[.)]|[①-⑩])\s+", instruction)
        )
        return max(count, ordered_lines)

    def _contains_pattern(self, text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _build_missing_questions(
        self,
        *,
        has_input: bool,
        has_output: bool,
        component_count: int,
        has_flow: bool,
        has_connections: bool,
        has_minimum_substance: bool,
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

        if not has_input:
            questions.append(
                "- 入力を明確にしてください。画像、テキスト、特徴量など何を受け取りますか。"
            )
        if not has_output:
            questions.append(
                "- 出力を明確にしてください。最終的に何を予測・生成・返すのですか。"
            )
        if component_count < 2:
            questions.append(
                "- 主要なモジュールや処理段階を少なくとも2段階以上、順番に列挙してください。"
            )
        if not has_flow:
            questions.append(
                "- 各モジュールがどの順番で動くのか、処理フローを説明してください。"
            )
        if not has_connections and component_count < 3:
            questions.append(
                "- モジュール間のつながりを具体化してください。単純な直列ならその旨、分岐・結合・反復があるならそれも書いてください。"
            )
        if not has_minimum_substance:
            questions.append(
                "- 1文の要約ではなく、入力から出力までを2〜4段階程度に分けて具体的に説明してください。"
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
        found = self._extract_component_names(instruction)
        if not found:
            return "- 明示されていない"
        return "\n".join(f"- {name}" for name in found[:8])

    def _extract_component_names(self, instruction: str) -> list[str]:
        component_patterns = (
            r"\b[A-Z]{2,}[A-Za-z0-9_-]*\b",
            r"[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*",
            r"[ぁ-んァ-ン一-龥A-Za-z0-9_-]*モジュール",
            r"[ぁ-んァ-ン一-龥A-Za-z0-9_-]*ブロック",
            r"[ぁ-んァ-ン一-龥A-Za-z0-9_-]*層",
            r"[ぁ-んァ-ン一-龥A-Za-z0-9_-]*エンコーダ",
            r"[ぁ-んァ-ン一-龥A-Za-z0-9_-]*デコーダ",
            r"[ぁ-んァ-ン一-龥A-Za-z0-9_-]*ヘッド",
            r"[ぁ-んァ-ン一-龥A-Za-z0-9_-]*バックボーン",
        )
        found: list[str] = []
        seen_normalized: set[str] = set()
        ignored = {
            "a",
            "an",
            "and",
            "by",
            "for",
            "if",
            "in",
            "it",
            "method",
            "our",
            "that",
            "the",
            "their",
            "then",
            "there",
            "these",
            "this",
            "those",
            "we",
        }

        for pattern in component_patterns:
            for match in re.findall(pattern, instruction):
                value = match.strip(" -\t:,.()[]{}")
                if len(value) < 2:
                    continue
                normalized = value.lower()
                if normalized in ignored:
                    continue
                if normalized not in seen_normalized:
                    found.append(value)
                    seen_normalized.add(normalized)

        keyword_components = (
            ("encoder", "encoder"),
            ("decoder", "decoder"),
            ("attention", "attention"),
            ("fusion", "fusion"),
            ("backbone", "backbone"),
            ("head", "head"),
            ("neck", "neck"),
            ("branch", "branch"),
            ("module", "module"),
            ("block", "block"),
            ("layer", "layer"),
            ("network", "network"),
            ("エンコーダ", "エンコーダ"),
            ("デコーダ", "デコーダ"),
            ("注意", "注意機構"),
            ("融合", "融合"),
            ("バックボーン", "バックボーン"),
            ("ヘッド", "ヘッド"),
            ("ブロック", "ブロック"),
            ("モジュール", "モジュール"),
            ("層", "層"),
            ("段階", "段階"),
        )
        lowered = instruction.lower()
        for token, label in keyword_components:
            if token in lowered and label.lower() not in seen_normalized:
                found.append(label)
                seen_normalized.add(label.lower())

        return found

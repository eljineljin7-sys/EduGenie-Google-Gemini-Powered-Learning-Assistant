import json
import logging
import re
from typing import Any, Dict, List

from errors import AIServiceError
from gemini_client import generate
from schemas import QuizQuestion

logger = logging.getLogger("edugenie.quiz")

NUM_QUESTIONS = 3
NUM_OPTIONS = 4

QUIZ_SYSTEM_PROMPT = f"""You write multiple-choice quizzes for students.
Rules:
- Generate exactly {NUM_QUESTIONS} questions.
- Each question has exactly {NUM_OPTIONS} distinct options and exactly one correct answer.
- Wrong options (distractors) must be plausible but clearly incorrect.
- If the input is a passage, base every question ONLY on facts stated in the passage.
  If the input is just a topic name, ask about core, well-established facts of that topic.
- "answer" must be copied exactly, character for character, from one of the options.
- Return valid JSON only: no Markdown, no code fences, no explanation.
Format:
[{{"question": "...", "options": ["...", "...", "...", "..."], "answer": "..."}}]"""

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)


def clean_json_block(raw: str) -> str:
    text = raw.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    elif not text.startswith(("[", "{")):
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def _resolve_answer(answer: str, options: List[str]) -> str:
    if answer in options:
        return answer
    folded = [o.casefold() for o in options]
    if answer.casefold() in folded:
        return options[folded.index(answer.casefold())]
    letter = answer.strip().rstrip(").:").upper()
    if len(letter) == 1 and "A" <= letter <= "D":
        return options[ord(letter) - ord("A")]
    raise ValueError(f"answer {answer!r} does not match any option")


def validate_quiz(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) != 1:
            raise ValueError("expected a JSON array of questions")
        data = lists[0]
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of questions")
    if len(data) != NUM_QUESTIONS:
        raise ValueError(f"expected {NUM_QUESTIONS} questions, got {len(data)}")

    quiz = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"question {i} is not an object")
        question, options, answer = item.get("question"), item.get("options"), item.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {i} has no question text")
        if not isinstance(options, list) or len(options) != NUM_OPTIONS:
            raise ValueError(f"question {i} must have exactly {NUM_OPTIONS} options")
        if not all(isinstance(o, str) and o.strip() for o in options):
            raise ValueError(f"question {i} has an empty or non-text option")
        options = [o.strip() for o in options]
        if len({o.casefold() for o in options}) != NUM_OPTIONS:
            raise ValueError(f"question {i} has duplicate options")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"question {i} has no answer")
        quiz.append(
            {
                "question": question.strip(),
                "options": options,
                "answer": _resolve_answer(answer.strip(), options),
            }
        )
    return quiz


def parse_quiz(raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(clean_json_block(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    return validate_quiz(data)


def generate_quiz(text: str, attempts: int = 2) -> List[Dict[str, Any]]:
    prompt = f"Create the quiz from this input:\n\"\"\"\n{text}\n\"\"\""
    for attempt in range(1, attempts + 1):
        raw = generate(
            prompt,
            system_instruction=QUIZ_SYSTEM_PROMPT,
            response_schema=list[QuizQuestion],
        )
        try:
            return parse_quiz(raw)
        except ValueError as exc:
            logger.warning("Invalid quiz from Gemini (attempt %d/%d): %s | raw=%r", attempt, attempts, exc, raw[:500])
    raise AIServiceError(
        "The AI returned a quiz in an unexpected format. Please try again, "
        "or provide a slightly longer passage.",
        502,
    )

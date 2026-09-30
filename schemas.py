from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator

LIMITS = {
    "qa": 2000,
    "explain": 300,
    "quiz": 8000,
    "summarize": 20000,
    "recommend": 200,
}
MIN_SUMMARY_WORDS = 20


def clean_text(value: str, *, label: str, max_chars: int) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"Please enter {label}.")
    if len(text) > max_chars:
        raise ValueError(
            f"Your input is too long ({len(text):,} characters). "
            f"The maximum is {max_chars:,} characters - please shorten it."
        )
    return text


class QARequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def _validate(cls, v: str) -> str:
        return clean_text(v, label="a question", max_chars=LIMITS["qa"])


class ExplainRequest(BaseModel):
    topic: str

    @field_validator("topic")
    @classmethod
    def _validate(cls, v: str) -> str:
        return clean_text(v, label="a concept to explain", max_chars=LIMITS["explain"])


class QuizRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _validate(cls, v: str) -> str:
        return clean_text(v, label="a topic or passage for the quiz", max_chars=LIMITS["quiz"])


class SummaryRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _validate(cls, v: str) -> str:
        text = clean_text(v, label="some text to summarize", max_chars=LIMITS["summarize"])
        if len(text.split()) < MIN_SUMMARY_WORDS:
            raise ValueError(
                f"Please paste a longer passage (at least {MIN_SUMMARY_WORDS} words) to summarize."
            )
        return text


class LearningPathRequest(BaseModel):
    topic: str
    level: Optional[Literal["beginner", "intermediate", "advanced"]] = None

    @field_validator("topic")
    @classmethod
    def _validate(cls, v: str) -> str:
        return clean_text(v, label="a subject to learn", max_chars=LIMITS["recommend"])


class TextResponse(BaseModel):
    success: bool = True
    result: str


class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str


class QuizResponse(BaseModel):
    success: bool = True
    quiz: List[QuizQuestion]


class Resource(BaseModel):
    type: str
    title: str


class LearningStage(BaseModel):
    level: str
    duration: str
    goal: str
    topics: List[str]
    steps: List[str]
    resources: List[Resource]


class LearningPath(BaseModel):
    topic: str
    overview: str
    stages: List[LearningStage]
    tips: List[str] = []


class LearningPathResponse(BaseModel):
    success: bool = True
    learning_path: LearningPath

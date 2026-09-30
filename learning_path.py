import json
import logging
from typing import Any, Dict, List, Optional

from errors import AIServiceError
from gemini_client import generate
from quiz_module import clean_json_block
from schemas import LearningPath

logger = logging.getLogger("edugenie.learning_path")

LEVELS = ["Beginner", "Intermediate", "Advanced"]

PATH_SYSTEM_PROMPT = """You are an expert curriculum designer creating personalized learning roadmaps.
- Start from fundamentals and progress step by step toward advanced concepts.
- Organize the roadmap into stages by difficulty: "Beginner", "Intermediate", "Advanced"
  (use exactly these level names, in that order). If the learner already has a level,
  start at that level and skip earlier stages.
- For each stage give: a realistic suggested duration (e.g. "2-3 weeks"), a one-sentence goal,
  the topics in the recommended study order, concrete step-by-step actions (including practice),
  and 2-4 well-known learning resources.
- Resources: give the name of a real, widely known video course, article/tutorial, book,
  or official documentation, with type one of "video", "article", "book", "course", "docs".
  Do NOT include URLs.
- "overview" is 1-2 sentences; "tips" is 2-4 short study tips.
- Return valid JSON only, no Markdown."""


def _string_list(value: Any, field: str, stage: int) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"stage {stage} {field} is not a list")
    items = [v.strip() for v in value if isinstance(v, str) and v.strip()]
    if not items:
        raise ValueError(f"stage {stage} has no {field}")
    return items


def validate_learning_path(data: Any, learner_level: Optional[str]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    stages = data.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("no stages")

    clean_stages = []
    last_index = -1
    for i, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            raise ValueError(f"stage {i} is not an object")
        level = str(stage.get("level", "")).strip().capitalize()
        if level not in LEVELS:
            raise ValueError(f"stage {i} has unknown level {level!r}")
        if LEVELS.index(level) < last_index:
            raise ValueError("stages are not ordered from easier to harder")
        last_index = LEVELS.index(level)
        resources = [
            {"type": str(r.get("type") or "").strip().lower() or "resource", "title": r["title"].strip()}
            for r in stage.get("resources") or []
            if isinstance(r, dict) and isinstance(r.get("title"), str) and r["title"].strip()
        ]
        clean_stages.append(
            {
                "level": level,
                "duration": str(stage.get("duration") or "Self-paced").strip(),
                "goal": str(stage.get("goal") or "").strip(),
                "topics": _string_list(stage.get("topics"), "topics", i),
                "steps": _string_list(stage.get("steps"), "steps", i),
                "resources": resources,
            }
        )

    start = LEVELS.index(learner_level.capitalize()) if learner_level else 0
    missing = set(LEVELS[start:]) - {s["level"] for s in clean_stages}
    if missing:
        raise ValueError(f"roadmap is incomplete, missing: {sorted(missing)}")

    tips = data.get("tips")
    return {
        "topic": str(data.get("topic") or "").strip(),
        "overview": str(data.get("overview") or "").strip(),
        "stages": clean_stages,
        "tips": [t.strip() for t in tips if isinstance(t, str) and t.strip()] if isinstance(tips, list) else [],
    }


def recommend_learning_path(topic: str, level: Optional[str] = None, attempts: int = 2) -> Dict[str, Any]:
    level_text = f"The learner's current level is: {level}." if level else "The learner's level is unknown; start from the basics."
    prompt = f"Create a learning roadmap for: {topic}\n{level_text}"
    for attempt in range(1, attempts + 1):
        raw = generate(prompt, system_instruction=PATH_SYSTEM_PROMPT, response_schema=LearningPath)
        try:
            path = validate_learning_path(json.loads(clean_json_block(raw)), level)
            path["topic"] = path["topic"] or topic
            return path
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Invalid learning path (attempt %d/%d): %s | raw=%r", attempt, attempts, exc, raw[:500])
    raise AIServiceError("The AI returned an incomplete learning path. Please try again.", 502)

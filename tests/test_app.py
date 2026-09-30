import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import httpx
import pytest

os.environ["EDUGENIE_PRELOAD_MODEL"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

import explanation_module
import gemini_client
from errors import AIServiceError
from main import app
from quiz_module import clean_json_block, parse_quiz

client = TestClient(app, raise_server_exceptions=False)

VALID_QUIZ = [
    {"question": "What does the Earth revolve around?", "options": ["The Moon", "The Sun", "Mars", "Jupiter"], "answer": "The Sun"},
    {"question": "How long is one revolution?", "options": ["24 hours", "30 days", "About 365 days", "10 years"], "answer": "About 365 days"},
    {"question": "What is one complete trip around the Sun called?", "options": ["Rotation", "Revolution", "Eclipse", "Orbit decay"], "answer": "Revolution"},
]


def stage(level: str) -> dict:
    return {
        "level": level, "duration": "2 weeks", "goal": f"{level} goal",
        "topics": [f"{level} topic"], "steps": [f"{level} step"],
        "resources": [{"type": "book", "title": f"{level} book"}],
    }


VALID_PATH = {
    "topic": "SQL", "overview": "Learn SQL.",
    "stages": [stage("Beginner"), stage("Intermediate"), stage("Advanced")],
    "tips": ["Practice daily"],
}


class FakeModels:
    def __init__(self, outputs: List[object]):
        self.outputs = list(outputs)
        self.calls: List[dict] = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, BaseException):
            raise out
        return SimpleNamespace(text=out)


@pytest.fixture
def fake_gemini(monkeypatch):
    monkeypatch.setattr(gemini_client, "SERVER_ERROR_RETRY_DELAY_SECONDS", 0)

    def install(*outputs: object) -> FakeModels:
        models = FakeModels(outputs)
        monkeypatch.setattr(gemini_client, "get_client", lambda: SimpleNamespace(models=models))
        return models
    return install


def assert_error(resp, status: int, contains: Optional[str] = None) -> str:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["success"] is False and isinstance(body["error"], str)
    assert "Traceback" not in body["error"]
    if contains:
        assert contains.lower() in body["error"].lower(), body["error"]
    return body["error"]


def test_homepage_loads():
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "Welcome to EduGenie" in html
    for task in ["qa", "explain", "summarize", "quiz", "recommend"]:
        assert f'data-task="{task}"' in html
    for button in ["Get Answer", "Explain", "Summarize", "Generate Quiz", "Get Recommendations"]:
        assert f">{button}</button>" in html
    assert "/static/app.js" in html and "/static/style.css" in html
    assert 'name="viewport"' in html


def test_static_assets_served():
    js = client.get("/static/app.js")
    assert js.status_code == 200
    for endpoint in ['"/explain"', '"/qa"', '"/quiz"', '"/summarize"', '"/learn/recommendations"']:
        assert endpoint in js.text
    assert client.get("/static/style.css").status_code == 200


@pytest.mark.parametrize("path,field", [
    ("/qa", "question"), ("/explain", "topic"), ("/quiz", "text"),
    ("/summarize", "text"), ("/learn/recommendations", "topic"),
])
@pytest.mark.parametrize("value", ["", "   \n\t "])
def test_empty_input_rejected(path, field, value, fake_gemini):
    models = fake_gemini("should not be called")
    assert_error(client.post(path, json={field: value}), 422, "please enter")
    assert models.calls == []


@pytest.mark.parametrize("path,field,limit", [
    ("/qa", "question", 2000), ("/explain", "topic", 300), ("/quiz", "text", 8000),
    ("/summarize", "text", 20000), ("/learn/recommendations", "topic", 200),
])
def test_too_long_input_rejected(path, field, limit):
    assert_error(client.post(path, json={field: "word " * limit}), 422, "too long")


def test_missing_field_wrong_type_and_bad_json():
    assert_error(client.post("/qa", json={}), 422, "question")
    assert_error(client.post("/qa", json={"question": 123}), 422, "question")
    assert_error(
        client.post("/qa", content=b"{not json", headers={"Content-Type": "application/json"}), 422, "valid JSON"
    )
    assert_error(client.post("/qa"), 422, "missing")
    assert_error(client.post("/qa", json=["a list"]), 422, "not a JSON object")


def test_summary_needs_enough_words():
    assert_error(client.post("/summarize", json={"text": "Too short to summarize."}), 422, "at least 20 words")


def test_invalid_level_rejected():
    assert_error(client.post("/learn/recommendations", json={"topic": "SQL", "level": "expert"}), 422, "level")


def test_unknown_route_returns_json_error():
    assert_error(client.get("/nope"), 404)


def test_missing_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    gemini_client.get_client.cache_clear()
    assert_error(client.post("/qa", json={"question": "What is the largest ocean?"}), 503, "GEMINI_API_KEY")
    gemini_client.get_client.cache_clear()


def test_client_is_created_once(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    gemini_client.get_client.cache_clear()
    assert gemini_client.get_client() is gemini_client.get_client()
    gemini_client.get_client.cache_clear()


@pytest.mark.parametrize("exc,status,text", [
    (genai_errors.ClientError(429, {"error": {"message": "quota"}}), 429, "limit"),
    (genai_errors.ClientError(429, {"error": {"message": "quota", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}), 429, "daily"),
    (genai_errors.ClientError(404, {"error": {"message": "model not found"}}), 503, "GEMINI_MODEL"),
    (genai_errors.ClientError(400, {"error": {"message": "API key not valid"}}), 503, "API key"),
    (genai_errors.ClientError(403, {"error": {"message": "denied"}}), 503, "API key"),
    (genai_errors.ServerError(503, {"error": {"message": "overloaded"}}), 503, "temporarily unavailable"),
    (httpx.ReadTimeout("timed out"), 504, "too long"),
    (httpx.ConnectError("no network"), 503, "internet"),
])
def test_gemini_errors_mapped(exc, status, text, fake_gemini):
    fake_gemini(exc)
    assert_error(client.post("/qa", json={"question": "What is gravity?"}), status, text)


def test_gemini_server_error_retried_once(fake_gemini):
    overloaded = genai_errors.ServerError(503, {"error": {"message": "high demand"}})
    models = fake_gemini(overloaded, "Recovered answer.")
    resp = client.post("/qa", json={"question": "What is gravity?"})
    assert resp.json() == {"success": True, "result": "Recovered answer."}
    assert len(models.calls) == 2


def test_gemini_server_error_gives_up_after_retry(fake_gemini):
    models = fake_gemini(genai_errors.ServerError(503, {"error": {"message": "high demand"}}))
    assert_error(client.post("/qa", json={"question": "What is gravity?"}), 503, "temporarily unavailable")
    assert len(models.calls) == 2


def test_client_errors_not_retried(fake_gemini):
    models = fake_gemini(genai_errors.ClientError(429, {"error": {"message": "quota"}}))
    assert_error(client.post("/qa", json={"question": "What is gravity?"}), 429)
    assert len(models.calls) == 1


def test_empty_gemini_response(fake_gemini):
    fake_gemini("")
    assert_error(client.post("/qa", json={"question": "What is gravity?"}), 502, "empty")


def test_api_key_never_leaks(monkeypatch, fake_gemini):
    secret = "SECRET-KEY-12345"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    fake_gemini(genai_errors.ClientError(400, {"error": {"message": f"API key not valid: {secret}"}}))
    resp = client.post("/qa", json={"question": "Hi?"})
    assert secret not in resp.text
    assert secret not in client.get("/").text


def test_unexpected_error_is_generic(monkeypatch):
    def boom(_):
        raise RuntimeError("internal detail /secret/path")
    monkeypatch.setattr("main.answer_question", boom)
    err = assert_error(client.post("/qa", json={"question": "Why?"}), 500, "something went wrong")
    assert "secret" not in err


def test_qa_success(fake_gemini):
    models = fake_gemini("The **Pacific Ocean** is the largest ocean on Earth.")
    resp = client.post("/qa", json={"question": "  What is the largest ocean?  "})
    assert resp.status_code == 200
    assert resp.json() == {"success": True, "result": "The **Pacific Ocean** is the largest ocean on Earth."}
    call = models.calls[0]
    assert call["contents"] == "What is the largest ocean?"
    assert "educational assistant" in call["config"].system_instruction
    assert call["model"] == gemini_client.GEMINI_MODEL


def test_summarize_success(fake_gemini):
    passage = ("The water cycle describes how water moves on Earth. Water evaporates from oceans, "
               "condenses into clouds, and falls as precipitation. The water cycle describes how water moves.")
    models = fake_gemini("- Water evaporates, condenses and precipitates.")
    resp = client.post("/summarize", json={"text": passage})
    assert resp.json() == {"success": True, "result": "- Water evaporates, condenses and precipitates."}
    assert passage in models.calls[0]["contents"]
    assert "do not add information" in models.calls[0]["config"].system_instruction.lower()
    assert models.calls[0]["config"].temperature is None


def test_clean_json_block_variants():
    payload = json.dumps(VALID_QUIZ)
    assert clean_json_block(payload) == payload
    assert clean_json_block(f"```json\n{payload}\n```") == payload
    assert clean_json_block(f"```\n{payload}\n```") == payload
    assert clean_json_block(f"  ```JSON\n{payload}```  ") == payload
    assert clean_json_block(f"Here is your quiz:\n{payload}\nGood luck!") == payload


def test_quiz_success_with_code_fence(fake_gemini):
    models = fake_gemini(f"```json\n{json.dumps(VALID_QUIZ)}\n```")
    text = "The Earth revolves around the Sun. One complete revolution takes approximately 365 days."
    resp = client.post("/quiz", json={"text": text})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True and body["quiz"] == VALID_QUIZ
    for q in body["quiz"]:
        assert len(q["options"]) == 4 and q["answer"] in q["options"]
    assert models.calls[0]["config"].response_mime_type == "application/json"
    assert text in models.calls[0]["contents"]


def test_quiz_answer_normalization():
    data = json.loads(json.dumps(VALID_QUIZ))
    data[0]["answer"] = "the sun"
    data[1]["answer"] = "C"
    quiz = parse_quiz(json.dumps({"quiz": data}))
    assert quiz[0]["answer"] == "The Sun"
    assert quiz[1]["answer"] == "About 365 days"


def _mutated(fn) -> str:
    data = json.loads(json.dumps(VALID_QUIZ))
    fn(data)
    return json.dumps(data)


@pytest.mark.parametrize("raw,reason", [
    ("not json at all", "invalid JSON"),
    ("{'question': 'single quotes'}", "invalid JSON"),
    (json.dumps(VALID_QUIZ[:2]), "expected 3 questions"),
    (json.dumps(VALID_QUIZ + VALID_QUIZ[:1]), "expected 3 questions"),
    (_mutated(lambda d: d[0]["options"].pop()), "exactly 4 options"),
    (_mutated(lambda d: d[1].__setitem__("answer", "Pluto")), "does not match"),
    (_mutated(lambda d: d[2].__setitem__("question", "  ")), "no question text"),
    (_mutated(lambda d: d[0]["options"].__setitem__(1, "")), "empty or non-text option"),
    (_mutated(lambda d: d[0]["options"].__setitem__(1, "the moon")), "duplicate options"),
    (_mutated(lambda d: d[0].pop("answer")), "no answer"),
    (json.dumps({"a": [], "b": []}), "JSON array"),
])
def test_parse_quiz_rejects_malformed(raw, reason):
    with pytest.raises(ValueError, match=reason):
        parse_quiz(raw)


def test_quiz_malformed_output_returns_502(fake_gemini):
    models = fake_gemini(json.dumps(VALID_QUIZ[:2]))
    assert_error(client.post("/quiz", json={"text": "Photosynthesis"}), 502, "unexpected format")
    assert len(models.calls) == 2


def test_quiz_retry_recovers(fake_gemini):
    fake_gemini("```json\n[broken", json.dumps(VALID_QUIZ))
    resp = client.post("/quiz", json={"text": "Photosynthesis"})
    assert resp.status_code == 200 and resp.json()["quiz"] == VALID_QUIZ


def test_learning_path_success(fake_gemini):
    models = fake_gemini(json.dumps(VALID_PATH))
    resp = client.post("/learn/recommendations", json={"topic": "SQL"})
    assert resp.status_code == 200, resp.text
    path = resp.json()["learning_path"]
    assert [s["level"] for s in path["stages"]] == ["Beginner", "Intermediate", "Advanced"]
    assert "start from the basics" in models.calls[0]["contents"]


def test_learning_path_respects_level(fake_gemini):
    data = dict(VALID_PATH, stages=[stage("intermediate"), stage("ADVANCED")])
    models = fake_gemini(json.dumps(data))
    resp = client.post("/learn/recommendations", json={"topic": "SQL", "level": "intermediate"})
    assert resp.status_code == 200, resp.text
    assert [s["level"] for s in resp.json()["learning_path"]["stages"]] == ["Intermediate", "Advanced"]
    assert "intermediate" in models.calls[0]["contents"]


@pytest.mark.parametrize("data", [
    dict(VALID_PATH, stages=[stage("Beginner"), stage("Intermediate")]),
    dict(VALID_PATH, stages=[stage("Advanced"), stage("Beginner"), stage("Intermediate")]),
    dict(VALID_PATH, stages=[]),
    dict(VALID_PATH, stages=[stage("Beginner"), stage("Intermediate"), dict(stage("Advanced"), topics=[])]),
    ["not", "an", "object"],
])
def test_learning_path_invalid_returns_502(data, fake_gemini):
    fake_gemini(json.dumps(data))
    assert_error(client.post("/learn/recommendations", json={"topic": "SQL"}), 502, "incomplete")


def test_explain_success(monkeypatch):
    monkeypatch.setattr(explanation_module, "explain_topic", lambda t: f"{t} is how plants make food.")
    resp = client.post("/explain", json={"topic": "Photosynthesis"})
    assert resp.json() == {"success": True, "result": "Photosynthesis is how plants make food."}


def test_explain_model_still_loading(monkeypatch):
    monkeypatch.setattr(explanation_module, "explain_model", None)
    explanation_module._load_lock.acquire()
    try:
        assert_error(client.post("/explain", json={"topic": "Gravity"}), 503, "still loading")
    finally:
        explanation_module._load_lock.release()


def test_explain_model_load_failure(monkeypatch):
    monkeypatch.setattr(explanation_module, "explain_model", None)

    def fail():
        raise OSError("disk full")
    monkeypatch.setattr(explanation_module, "load_model", fail)
    err = assert_error(client.post("/explain", json={"topic": "Gravity"}), 503, "could not be loaded")
    assert "disk full" not in err


def test_ai_service_error_default_status():
    assert AIServiceError("x").status_code == 502

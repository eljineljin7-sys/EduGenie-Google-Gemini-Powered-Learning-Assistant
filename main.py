import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

import explanation_module
from errors import AIServiceError
from learning_path import recommend_learning_path
from qna import answer_question
from quiz_module import generate_quiz
from schemas import (
    LIMITS,
    ExplainRequest,
    LearningPathRequest,
    LearningPathResponse,
    QARequest,
    QuizRequest,
    QuizResponse,
    SummaryRequest,
    TextResponse,
)
from summary_module import summarize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("edugenie")

BASE_DIR = Path(__file__).resolve().parent


def _preload_explanation_model() -> None:
    try:
        explanation_module.load_model()
    except Exception:
        logger.exception("Background load of the explanation model failed; will retry on first /explain request")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("EDUGENIE_PRELOAD_MODEL", "true").lower() in ("1", "true", "yes"):
        threading.Thread(target=_preload_explanation_model, daemon=True).start()
    yield


app = FastAPI(title="EduGenie", description="Google Gemini powered learning assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"success": False, "error": message})


@app.exception_handler(AIServiceError)
async def ai_error_handler(request: Request, exc: AIServiceError) -> JSONResponse:
    return _error(exc.status_code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    if first.get("type") == "json_invalid":
        return _error(422, "The request body must be valid JSON.")
    if tuple(first.get("loc", ())) == ("body",):
        return _error(422, "The request body is missing or is not a JSON object.")
    message = str(first.get("msg", "Invalid input."))
    if message.startswith("Value error, "):
        message = message[len("Value error, "):]
    elif first.get("loc"):
        message = f"Invalid '{first['loc'][-1]}': {message}"
    return _error(422, message)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error(exc.status_code, str(exc.detail))


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "Something went wrong on the server. Please try again.")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"limits": LIMITS})


@app.post("/qa", response_model=TextResponse)
def qa(payload: QARequest):
    return TextResponse(result=answer_question(payload.question))


@app.post("/explain", response_model=TextResponse)
def explain(payload: ExplainRequest):
    return TextResponse(result=explanation_module.explain_topic(payload.topic))


@app.post("/quiz", response_model=QuizResponse)
def quiz(payload: QuizRequest):
    return {"success": True, "quiz": generate_quiz(payload.text)}


@app.post("/summarize", response_model=TextResponse)
def summarize(payload: SummaryRequest):
    return TextResponse(result=summarize_text(payload.text))


@app.post("/learn/recommendations", response_model=LearningPathResponse)
def learn_recommendations(payload: LearningPathRequest):
    return {"success": True, "learning_path": recommend_learning_path(payload.topic, payload.level)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")))

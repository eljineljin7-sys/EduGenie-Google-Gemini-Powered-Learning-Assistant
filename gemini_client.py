import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from errors import AIServiceError

load_dotenv(Path(__file__).resolve().parent / ".env")

logger = logging.getLogger("edugenie.gemini")
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

SERVER_ERROR_RETRY_DELAY_SECONDS = 2

DEFAULT_MODEL = "gemini-3.1-flash-lite"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise AIServiceError(
            "Gemini is not configured on the server. Add GEMINI_API_KEY to the .env file "
            "and restart the app.",
            status_code=503,
        )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_SECONDS * 1000),
    )


def _translate_api_error(exc: genai_errors.APIError) -> AIServiceError:
    code = getattr(exc, "code", None)
    text = str(exc).lower()
    if code in (401, 403) or (code == 400 and "api key" in text):
        return AIServiceError("The Gemini API key was rejected. Please check GEMINI_API_KEY.", 503)
    if code == 404:
        return AIServiceError(
            f"The Gemini model '{GEMINI_MODEL}' is not available. Set GEMINI_MODEL to a supported model.",
            503,
        )
    if code == 429:
        if "perday" in text:
            return AIServiceError(
                f"The daily Gemini quota for model '{GEMINI_MODEL}' has been used up. Try again tomorrow, "
                "or set GEMINI_MODEL in .env to another model (e.g. gemini-3.1-flash-lite).",
                429,
            )
        return AIServiceError("Gemini's usage limit was reached. Please wait a moment and try again.", 429)
    if code is not None and code >= 500:
        return AIServiceError("Gemini is temporarily unavailable. Please try again shortly.", 503)
    return AIServiceError("Gemini could not process this request. Please rephrase and try again.", 502)


def generate(
    prompt: str,
    *,
    system_instruction: str,
    response_schema: Optional[Any] = None,
) -> str:
    config = types.GenerateContentConfig(system_instruction=system_instruction)
    if response_schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema

    client = get_client()
    try:
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
        except genai_errors.ServerError as exc:
            logger.warning("Gemini server error (code=%s), retrying once: %s", exc.code, exc)
            time.sleep(SERVER_ERROR_RETRY_DELAY_SECONDS)
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
    except genai_errors.APIError as exc:
        logger.warning("Gemini API error (code=%s): %s", getattr(exc, "code", None), exc)
        raise _translate_api_error(exc) from exc
    except httpx.TimeoutException as exc:
        logger.warning("Gemini request timed out after %ss", GEMINI_TIMEOUT_SECONDS)
        raise AIServiceError("Gemini took too long to respond. Please try again.", 504) from exc
    except httpx.HTTPError as exc:
        logger.warning("Network error talking to Gemini: %s", exc)
        raise AIServiceError("Could not reach Gemini. Check the server's internet connection.", 503) from exc

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        logger.warning("Gemini returned an empty response: %r", response)
        raise AIServiceError(
            "Gemini returned an empty response (it may have been blocked by safety filters). "
            "Please try rephrasing.",
            502,
        )
    return text

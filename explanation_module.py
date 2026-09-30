import logging
import os
import threading
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase

from errors import AIServiceError

logger = logging.getLogger("edugenie.explain")

MODEL_NAME = "MBZUAI/LaMini-Flan-T5-783M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_dtype_name = os.getenv("EXPLAIN_MODEL_DTYPE", "bfloat16").strip().lower()
if _dtype_name not in _DTYPES:
    logger.warning("Unknown EXPLAIN_MODEL_DTYPE %r; using bfloat16", _dtype_name)
    _dtype_name = "bfloat16"
DTYPE = _DTYPES[_dtype_name]

explain_tokenizer: Optional[PreTrainedTokenizerBase] = None
explain_model: Optional[AutoModelForSeq2SeqLM] = None

_load_lock = threading.Lock()
_generate_lock = threading.Lock()


def load_model() -> None:
    global explain_tokenizer, explain_model
    with _load_lock:
        if explain_model is not None:
            return
        logger.info("Loading %s on %s as %s (first run downloads ~3 GB)...", MODEL_NAME, DEVICE, _dtype_name)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, dtype=DTYPE, low_cpu_mem_usage=True
        ).to(DEVICE)
        model.eval()
        explain_tokenizer, explain_model = tokenizer, model
        logger.info("%s loaded.", MODEL_NAME)


def is_loaded() -> bool:
    return explain_model is not None


def _ensure_model() -> None:
    if explain_model is not None:
        return
    if _load_lock.locked():
        raise AIServiceError(
            "The explanation model is still loading (the first run downloads about 3 GB). "
            "Please try again in a minute.",
            503,
        )
    try:
        load_model()
    except Exception as exc:
        logger.exception("Failed to load %s", MODEL_NAME)
        raise AIServiceError(
            "The local explanation model could not be loaded. Check the server logs.", 503
        ) from exc


def explain_topic(topic: str) -> str:
    _ensure_model()
    input_text = (
        f"Explain the concept of '{topic}' "
        "in a simple and clear way for a school student."
    )
    inputs = explain_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    try:
        with _generate_lock, torch.inference_mode():
            outputs = explain_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
            )
    except Exception as exc:
        logger.exception("LaMini generation failed")
        raise AIServiceError("The explanation could not be generated. Please try again.", 500) from exc

    explanation = explain_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if not explanation:
        raise AIServiceError("The model returned an empty explanation. Please try again.", 502)
    return explanation

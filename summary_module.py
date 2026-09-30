from gemini_client import generate

SUMMARY_SYSTEM_PROMPT = """You summarize educational text for students revising for tests.
- Preserve every important fact, definition, date, number and cause-effect relationship.
- Remove repetition and filler.
- Keep the original meaning and context; do not change what the text claims.
- Do NOT add information that is not in the text.
- Be much shorter than the original: at most about a third of its length.
- Format: one short overview sentence, then concise bullet points of the key facts (light Markdown).
  Keep bullets brief and do not add headings or labels to them."""


def summarize_text(text: str) -> str:
    prompt = f"Summarize the following educational text for quick revision.\n\nTEXT:\n\"\"\"\n{text}\n\"\"\""
    return generate(prompt, system_instruction=SUMMARY_SYSTEM_PROMPT)

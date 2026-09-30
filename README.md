# EduGenie — Google Gemini Powered Learning Assistant

EduGenie is a lightweight AI learning assistant for students and self-learners. It is a
single FastAPI app with a simple web page. The page has one section per feature, each with
its own input box, button and result card:

| Task | What it does | AI engine |
|------|--------------|-----------|
| **Explain** | Explains a concept simply, for a school student | Local model `MBZUAI/LaMini-Flan-T5-783M` |
| **QnA** | Answers general / academic questions | Google Gemini |
| **Quiz** | Makes 3 multiple-choice questions (4 options each) from a topic or passage; click an option to check it | Google Gemini |
| **Summary** | Condenses a long passage into revision notes | Google Gemini |
| **Recommend Path** | Builds a Beginner → Intermediate → Advanced learning roadmap with timelines and resources | Google Gemini |

## Architecture

```
Browser (templates/index.html + static/app.js)
   │  fetch() JSON POST
   ▼
FastAPI (main.py) ── validation (schemas.py) ── error handlers → {"success": false, "error": "..."}
   ├── /explain               → explanation_module.py → LaMini-Flan-T5 (local, loaded once)
   ├── /qa                    → qna.py            ┐
   ├── /quiz                  → quiz_module.py    │→ gemini_client.py → Gemini API
   ├── /summarize             → summary_module.py │   (one shared client)
   └── /learn/recommendations → learning_path.py  ┘
```

- AI calls are blocking, so the routes are plain `def` functions that FastAPI runs in its
  threadpool. A slow model call doesn't freeze the server.
- The LaMini model loads **once**, in a background thread at startup, so Gemini features work
  immediately. It uses CUDA when available and falls back to CPU.
- If Gemini is briefly overloaded (a 5xx error such as "high demand"), the request is retried
  once after 2 seconds before an error is shown.
- Quiz and learning-path output from Gemini is **never trusted blindly**. Code fences are
  stripped, the text is parsed with `json.loads` (never `eval`), the structure is validated,
  and a bad response is retried once before a clean error is returned.

## Folder structure

```
EduGenie/
├── main.py                - FastAPI app, routes, error handlers, frontend serving
├── explanation_module.py  - LaMini-Flan-T5-783M local explanations
├── qna.py                 - Gemini Q&A
├── quiz_module.py         - Gemini quiz + clean_json_block() + validation
├── summary_module.py      - Gemini summarization
├── learning_path.py       - Gemini learning roadmap + validation
├── gemini_client.py       - Shared Gemini client, model config, error translation
├── schemas.py             - Pydantic request/response models, input limits
├── errors.py              - AIServiceError (user-safe error messages)
├── templates/index.html   - Web page
├── static/style.css       - Styles (responsive)
├── static/app.js          - Frontend logic (plain JavaScript)
├── tests/test_app.py      - API/parsing tests (Gemini mocked)
├── requirements.txt
├── requirements-dev.txt   - + pytest
├── .env.example
└── .gitignore
```

## Prerequisites

- **Python 3.11** (on macOS: `brew install python@3.11`)
- About **4 GB free disk** for PyTorch and the LaMini model, and **8 GB RAM** recommended
- A **Gemini API key** from <https://aistudio.google.com/apikey> (needed for everything except Explain)
- Internet access (for Gemini, and to download the LaMini model the first time)

## Setup

1. Create and activate a virtual environment with Python 3.11
   (on Windows, activate with `.venv\Scripts\activate`):

   ```bash
   cd EduGenie
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Configure your Gemini API key. Copy the template (on Windows: `copy .env.example .env`),
   then edit `.env` and set `GEMINI_API_KEY=your_key_here`:

   ```bash
   cp .env.example .env
   ```

Never put a real key in `.env.example`. That file is a template meant to be shared.

`.env` is listed in `.gitignore`, so it won't be committed. The key stays on the server and is
never sent to the browser.

### Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GEMINI_API_KEY` | Yes (for Gemini features) | — | Your Gemini API key |
| `GEMINI_MODEL` | No | `gemini-3.6-flash` | Gemini model id |
| `GEMINI_TIMEOUT_SECONDS` | No | `60` | How long to wait for Gemini |
| `EDUGENIE_PRELOAD_MODEL` | No | `true` | Load LaMini at startup (`false` = load on first `/explain`) |
| `EXPLAIN_MODEL_DTYPE` | No | `bfloat16` | LaMini precision: `bfloat16` (~1.5 GB RAM) or `float32` (~3 GB RAM, ~6 GB peak while loading) |

## Running

With the virtual environment activated, start the app with:

```bash
python main.py
```

For development, `uvicorn main:app --reload` restarts the server whenever you edit a `.py` file.

With `--reload`, every saved `.py` file restarts the server, which also reloads the local
model. On an 8 GB machine, prefer `python main.py` unless you are editing code.

Open <http://127.0.0.1:8000>. Interactive API docs are at <http://127.0.0.1:8000/docs>.

**First run:** the LaMini model (~3 GB) downloads from Hugging Face into
`~/.cache/huggingface`. Until it has loaded, **Explain** answers "the explanation model is
still loading". The other features work immediately.

## API endpoints

All endpoints accept and return JSON. Every error has the form
`{"success": false, "error": "Human-readable message"}`.

| Method | Path | Request body | Success response |
|--------|------|--------------|------------------|
| GET | `/` | — | Web page |
| POST | `/qa` | `{"question": str}` (≤ 2,000 chars) | `{"success": true, "result": str}` (Markdown) |
| POST | `/explain` | `{"topic": str}` (≤ 300 chars) | `{"success": true, "result": str}` |
| POST | `/quiz` | `{"text": str}` (≤ 8,000 chars) | `{"success": true, "quiz": [{question, options[4], answer}] × 3}` |
| POST | `/summarize` | `{"text": str}` (20 words – 20,000 chars) | `{"success": true, "result": str}` (Markdown) |
| POST | `/learn/recommendations` | `{"topic": str, "level"?: "beginner"\|"intermediate"\|"advanced"}` | `{"success": true, "learning_path": {...}}` |

Input that is empty, only whitespace, or over the limit is rejected with HTTP 422. It is never
silently truncated.

### Examples

**Q&A**

```bash
curl -X POST http://127.0.0.1:8000/qa -H "Content-Type: application/json" \
     -d '{"question": "What is the largest ocean?"}'
```

```json
{"success": true, "result": "The **Pacific Ocean** is the largest ocean on Earth..."}
```

**Explain**

```bash
curl -X POST http://127.0.0.1:8000/explain -H "Content-Type: application/json" \
     -d '{"topic": "Photosynthesis"}'
```

```json
{"success": true, "result": "Photosynthesis is the process by which plants ..."}
```

**Quiz**

```bash
curl -X POST http://127.0.0.1:8000/quiz -H "Content-Type: application/json" \
     -d '{"text": "The Earth revolves around the Sun. One complete revolution takes approximately 365 days."}'
```

```json
{"success": true, "quiz": [
  {"question": "What does the Earth revolve around?",
   "options": ["The Moon", "The Sun", "Mars", "Jupiter"], "answer": "The Sun"}]}
```

The real response always contains 3 questions.

**Learning path**

```bash
curl -X POST http://127.0.0.1:8000/learn/recommendations -H "Content-Type: application/json" \
     -d '{"topic": "SQL", "level": "beginner"}'
```

```json
{"success": true, "learning_path": {
  "topic": "SQL", "overview": "...",
  "stages": [{"level": "Beginner", "duration": "2-3 weeks", "goal": "...",
              "topics": ["..."], "steps": ["..."],
              "resources": [{"type": "book", "title": "..."}]}],
  "tips": ["..."]}}
```

The real response has one stage per level, from the learner's level up to Advanced.

**Invalid input** (returns HTTP 422)

```bash
curl -X POST http://127.0.0.1:8000/qa -H "Content-Type: application/json" -d '{"question": "  "}'
```

```json
{"success": false, "error": "Please enter a question."}
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests replace Gemini with a fake client, so they need no API key or network. They cover
validation, error mapping, quiz/learning-path parsing (including malformed AI output) and page
serving.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Gemini is not configured on the server" | Create `.env` with `GEMINI_API_KEY=...` and restart uvicorn. |
| "The Gemini API key was rejected" | The key is wrong or revoked. Create a new one in Google AI Studio. |
| "The Gemini model '…' is not available" | Set `GEMINI_MODEL` in `.env` to a model listed at <https://ai.google.dev/gemini-api/docs/models>. |
| "Gemini's usage limit was reached" | Free-tier rate limit. Wait a minute and try again. |
| "The daily Gemini quota … has been used up" | Free-tier keys allow a small number of requests per model per day (e.g. 20). Wait until tomorrow, or set `GEMINI_MODEL` to another model such as `gemini-3.1-flash-lite`. Quiz and learning path may use 2 requests when the first answer is malformed. |
| "Gemini is temporarily unavailable" | Google's servers are overloaded (already retried once). Try again shortly. |
| "The explanation model is still loading" | First-run download/load is still going on. Watch the server log for `LaMini-Flan-T5-783M loaded`. |
| "The local explanation model could not be loaded" | Check disk space and internet access, then look at the server log. |
| Explain is slow | It runs a 783M-parameter model on your CPU. About 5–10 seconds per answer is normal. |
| Model load takes minutes / Mac becomes sluggish | The system is out of RAM and swapping. Keep `EXPLAIN_MODEL_DTYPE=bfloat16`, close memory-heavy apps, and avoid `--reload`. |
| `zsh: command not found: python` | Activate the virtual environment first (`source .venv/bin/activate`), or use `python3`. |

## Limitations

- **Gemini model change:** the original design used `gemini-1.5-pro`, which Google has retired,
  and the `google-generativeai` SDK it relied on is deprecated. EduGenie now uses the supported
  `google-genai` SDK with a configurable model (`GEMINI_MODEL`). The app behaves the same way.
- LaMini-Flan-T5 is a small model. Its explanations are short and simple, and may sometimes be
  imprecise. Output is sampled, so the same topic gives slightly different wording each time.
- Explain runs on the CPU unless you have CUDA. Requests are handled one at a time to limit
  memory use (~1.5 GB RAM while loaded in bfloat16).
- AI answers, quizzes and suggested resources can contain mistakes. Resources are given by name,
  not as links, so the app never shows invented URLs. Search for them by name.
- No accounts, history or progress tracking. Each request is independent.

## Future scope (not implemented)

Voice interaction, multilingual support, PDF/image input, progress tracking and dashboards,
gamification (badges, streaks), adaptive paths, group study, teacher/parent dashboards, and
LMS integration (Moodle, Google Classroom). Each AI feature is its own module behind its own
route, so new features can be added the same way.

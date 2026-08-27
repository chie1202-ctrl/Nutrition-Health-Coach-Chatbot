# NutriCoachAI

**Privacy-Preserving Health Coach Chatbot** — MSc dissertation research prototype.

NutriCoachAI is a **local-first** health coaching application: user profiles, chat, cross-session memory, RAG-grounded answers, 7-day meal plans, and weight tracking run on your machine. Health data is stored in local SQLite; the language model runs through **Ollama** on `localhost` — not a cloud coaching API.

> **Scope:** General wellness coaching only — not medical advice, diagnosis, or treatment. This is a research and demonstration system, not a clinical product.

**Development status:** Application code is frozen and demo-ready for MSc submission. This public repository contains the source code, setup files, evaluation scripts, and technical documentation needed to reproduce the prototype locally. Local research notes, dissertation drafts, generated evaluation outputs, databases, vector indexes, dependency folders, and source PDFs are intentionally excluded from version control.

---

## Overview

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + Vite 5 |
| Backend | FastAPI + Uvicorn |
| Database | SQLite (`backend/database/health_coach.db`) |
| LLM | Ollama — default `deepseek-r1:8b` via `langchain-ollama` |
| RAG | Chroma + HuggingFace embeddings over local PDFs in `my_knowledge/` |
| Memory | Cross-session summaries (production default **M2**) |

**Key capabilities:**

- **SSE chat streaming** — incremental assistant replies (`POST /users/{id}/chat/stream`)
- **RAG citations** — source filename chips under assistant messages when `rag_ready: true`
- **Cross-session memory** — “New Conversation” closes a session, summarizes it, and injects compact memory into later prompts
- **Memory Viewer** — read profile context, session summaries, and session history; regenerate eligible summaries
- **Goal Progress** — dashboard card for current / target / remaining weight and trend
- **7-day meal plan** — live LLM JSON generation with validation and template fallback when the model fails
- **Settings → About** — system status (Ollama, RAG, memory mode) and **medical disclaimer**

When Ollama is not running, chat send and meal-plan generation are **disabled** in the UI; the API returns **HTTP 503** for AI endpoints (no silent cloud fallback).

---

## Requirements

- **Python 3.10+** and **Node.js 18+** with npm
- **[Ollama](https://ollama.com)** installed and on your `PATH`
- **~6 GB** disk space for `deepseek-r1:8b`
- **Network on first run** — HuggingFace may download the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) when RAG initializes
- **RAG source PDFs** — place the documents listed in `my_knowledge/SOURCES.md` into `my_knowledge/` if you want full RAG functionality
- **macOS / Linux** (or WSL) recommended; paths below use bash

---

## Quick start (recommended)

From the project root:

```bash
chmod +x start.sh
./start.sh
```

The launcher will:

1. Ensure **Ollama** is running (starts `ollama serve` in the background if needed)
2. **Pull** `deepseek-r1:8b` if not already present
3. Create a Python **virtual environment** at `.venv/` and install `backend/requirements.txt`
4. Run **`npm install`** in `frontend/` if needed
5. Load `backend/.env` (or `backend/.env.example` if `.env` is absent)
6. Start **backend** on `127.0.0.1:8000` and **frontend** on `127.0.0.1:5173`

Then open:

- **App:** [http://127.0.0.1:5173](http://127.0.0.1:5173)
- **Health check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Confirm the health JSON includes `"ollama_reachable": true` and `"rag_ready": true` before demoing chat or meal plans.

Press **Ctrl+C** in the terminal running `./start.sh` to stop backend and frontend.

**Logs:** `/tmp/nutricoach-backend.log`, `/tmp/nutricoach-frontend.log` (and `/tmp/nutricoach-ollama.log` if the launcher started Ollama).

---

## Ollama setup and verification

### Install and pull the model

```bash
# Install Ollama from https://ollama.com, then:
ollama pull deepseek-r1:8b
```

### Verify Ollama is running

```bash
curl -s http://127.0.0.1:11434/api/tags | head -c 200
```

You should see `deepseek-r1:8b` in the model list.

### Verify through the app

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

| Field | Expected for full demo |
|-------|-------------------------|
| `ollama_reachable` | `true` |
| `ollama_model` | `deepseek-r1:8b` |
| `rag_ready` | `true` |
| `memory_mode` | `M2` |

You can also open **Settings (⚙) → System** in the UI for the same status.

### Configuration

Defaults live in `backend/.env.example`. Copy to `backend/.env` only if you need overrides:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_MODEL` | `deepseek-r1:8b` | Chat and meal-plan model |
| `OLLAMA_REASONING` | `false` | Disable reasoning traces in production |
| `OLLAMA_NUM_PREDICT` | `768` | Chat token budget |
| `MEAL_PLAN_NUM_PREDICT` | `2048` | Meal-plan JSON token budget |
| `MEAL_PLAN_TEMPERATURE` | `0.1` | Lower variance for structured meal-plan JSON |
| `MEMORY_MODE` | `M2` | Cross-session memory injection mode |

Optional semantic-memory judge settings are also available for evaluation runs:
`OPENAI_API_KEY`, `OPENAI_JUDGE_MODEL`, `OPENAI_JUDGE_MAX_OUTPUT_TOKENS`, and
`OPENAI_JUDGE_TIMEOUT`. These are only needed for the OpenAI API judge path; the
copy-paste ChatGPT judging workflow can be used without an API key.

---

## Project structure

```text
Coach_ChatBot/
├── start.sh                 # Canonical one-command launcher
├── frontend/                # React + Vite SPA
├── backend/
│   ├── main.py              # FastAPI routes (REST + SSE)
│   ├── logic.py             # Domain logic, RAG, memory, meal plans
│   ├── database/            # Database schema/helper code; local DB files are generated
│   ├── tests/               # pytest suite
│   └── eval/                # Evaluation runners, protocols, and rubrics
├── my_knowledge/            # RAG source catalogue; PDFs are local-only
└── scripts/                 # Playwright validation and utility harnesses
```

The following runtime or research artifacts are intentionally not committed: `.venv/`, `frontend/node_modules/`, `frontend/dist/`, SQLite databases, Chroma vector indexes, `backend/eval/results/`, `backend/eval/locomo/data/`, local RAG PDFs, meeting notes, dissertation drafts, and experiment logs.

---

## Configuration

**Backend** — `backend/.env.example` (loaded automatically by `./start.sh`):

- Ollama URL, model, temperature, memory limits, session timeouts

**Frontend** — `frontend/.env.example`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Copy to `frontend/.env` if the backend runs on a different host or port.

---

## Manual setup (advanced)

Use separate terminals if you are not using `./start.sh`.

```bash
# Terminal 1 — Ollama (if not already running)
ollama serve
ollama pull deepseek-r1:8b

# Terminal 2 — Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
set -a && source backend/.env.example && set +a
cd backend && uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 3 — Frontend
cd frontend && npm install && npm run dev
```

Note: the launcher uses a **repo-root** `.venv`, binds **127.0.0.1** (not `0.0.0.0`), and does not pass `--reload` to Uvicorn.

---

## Using the application

### First-time onboarding

When no users exist, a full-screen **onboarding wizard** runs: profile → health goal → diet preferences and allergies → initial weight → main dashboard.

### Chat and RAG

- Type a message in the chat panel; replies stream via **SSE**.
- Ask questions grounded in the knowledge base (e.g. dietary guidelines); when RAG is ready, **source citation chips** appear under the assistant reply.
- **New Conversation** closes the current session, triggers background summarization, and starts a fresh session. A summarization banner may appear briefly while the summary is written.

### Cross-session memory and Memory Viewer

- After closing a session, ask about prior goals or constraints in the new session — the coach should recall summarized context (validated in Entry 021 step 07c).
- Click **View Memory** in the chat header to open the **Memory Viewer**: profile context, long-term cumulative memory (after rollup threshold), recent session summaries, and session history. Eligible closed sessions can be **regenerated**.

### Meal plan and weight tracking

- **Generate 7-Day Plan** produces a live LLM meal plan when Ollama is healthy (`llm_degraded: false`). The UI shows **AI Generated** or **Template Fallback** badges.
- Log weight via the sidebar; view **7 / 14 / 30-day** charts in the sidebar thumbnail or modal.
- **Goal Progress** on the dashboard shows current vs target weight, remaining kg, progress %, and a trend message when a target is set.

### Settings and medical disclaimer

Open **⚙ Settings and help** in the top bar:

- **System** — Ollama, model, RAG, and memory status
- **About & Help** — privacy narrative, getting started, and the **medical disclaimer**

The main dashboard no longer shows a persistent disclaimer banner (Entry 026); reviewers and demos should open **Settings → About** for the disclaimer text.

---

## 5-minute demo path

Suitable for supervisor review or defense rehearsal. Requires `./start.sh` with `ollama_reachable` and `rag_ready` true.

| Step | Action | What to show |
|------|--------|--------------|
| 1 | Create a user with goal `lose_weight` and allergy `shellfish` | Profile fields drive meal-plan validation |
| 2 | Ask a normal coaching question | SSE streaming reply |
| 3 | Ask a RAG question (e.g. vegetable recommendations per Dietary Guidelines) | **Source chips** under the reply |
| 4 | Send a cross-session seed message (e.g. “lose 5 kg in three months, no shellfish”) | Day-1 coaching |
| 5 | Click **New Conversation**; wait for summarization if banner appears | Session close + summary |
| 6 | Ask “What was my goal and what food should I avoid?” | **Cross-session recall** |
| 7 | Click **Generate 7-Day Plan** | Live plan, 7 days, no shellfish; **AI Generated** badge |
| 8 | Open **Settings → About** | **Medical disclaimer** and system status |
| 9 | Send an unsafe prompt (e.g. starvation for fast weight loss) | **Safety notice** in chat |
| 10 | Optional: open **View Memory**, **Goal Progress**, or resize to mobile width | Level 2 polish features |

This path aligns with `scripts/entry_021_b1_desktop_validation.mjs`. Generated validation outputs are local artifacts and are not included in the public repository.

---

## Validation and tests

### Automated unit and API tests

```bash
source .venv/bin/activate
pip install -r backend/requirements.txt
pytest backend/tests -q
```

### Playwright harnesses (stack must be running)

Requires Ollama, `deepseek-r1:8b`, and Playwright (via `frontend/node_modules`):

| Script | Entry | Scope |
|--------|-------|--------|
| `node scripts/entry_021_b1_desktop_validation.mjs` | 021 | Full desktop demo — **14/14** canonical |
| `node scripts/entry_020_mobile_validation.mjs` | 020 | Mobile 375×812 + 390×844 |
| `node scripts/entry_022_memory_viewer_validation.mjs` | 022 | Memory Viewer |
| `node scripts/entry_023_onboarding_validation.mjs` | 023 | Onboarding wizard |
| `node scripts/entry_024_goal_progress_validation.mjs` | 024 | Goal Progress card |

Generated validation JSON, logs, and screenshots are written under `backend/eval/results/` during local runs. That directory is ignored by Git because it contains generated artifacts rather than source code.

### Semantic memory evaluation

The expanded semantic memory evaluation used for the dissertation is implemented under `backend/eval/`:

| File | Purpose |
|------|---------|
| `memory_semantic_scripts.json` | Eight scripted cross-session memory scenarios |
| `run_memory_semantic_eval.py` | Runs M0/M1/M2/M3 memory conditions with repeated runs |
| `make_memory_semantic_judge_pack.py` | Creates copy-paste ChatGPT judging packs |
| `merge_memory_semantic_judgements.py` | Merges ChatGPT judgements into summary CSV/JSON files |

To run the response-generation stage without API-based judging:

```bash
source .venv/bin/activate
OLLAMA_REASONING=false OLLAMA_TEMPERATURE=0 SUMMARY_TEMPERATURE=0 \
python backend/eval/run_memory_semantic_eval.py --fresh-eval-db --repeats 3 --skip-ai-grading
```

Generated outputs are written to `backend/eval/results/`, which is intentionally ignored by Git.
For non-API judging, run `make_memory_semantic_judge_pack.py` on the generated JSON, paste the
chunks into ChatGPT, then merge the returned judgements with `merge_memory_semantic_judgements.py`.

**Meal-plan reliability probe** (backend only, no browser):

```bash
source .venv/bin/activate
set -a && source backend/.env.example && set +a
python3 scripts/meal_plan_reliability_probe.py 5
```

Typical result after Entry 027 hardening: **4/5 live success (~80%)**. One attempt may return **Template Fallback** — the UI still shows 7 days and does not crash. Re-run **Generate 7-Day Plan** if the badge shows Template Fallback during a demo.

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| Chat send button disabled | Ollama not running | Run `./start.sh` or `ollama serve`; confirm `/health` → `ollama_reachable: true` |
| `Failed to fetch` in UI | Backend not reachable | Ensure `./start.sh` is running; check `VITE_API_BASE_URL` matches backend port |
| Warning banner: “Ollama is not running” | Same as above | Start Ollama; refresh the page |
| HTTP **503** on chat or meal-plan | Ollama unreachable when AI was requested | Start Ollama and pull `deepseek-r1:8b` |
| `rag_ready: false` on `/health` | RAG not initialized | Ensure `my_knowledge/` contains PDFs; wait for first-time embedding download; check backend log |
| UI looks outdated after `git pull` | Stale Vite dev server | Stop all `npm run dev` processes; restart `./start.sh`; hard refresh (`Cmd+Shift+R` / `Ctrl+Shift+R`) |
| App on port **5174** instead of 5173 | Port 5173 already in use | Stop the other dev server or use the URL printed by Vite; avoid running two frontends |
| Meal plan shows **Template Fallback** | LLM returned invalid JSON after retries | Check Ollama load; generated failure captures, when present, are stored locally under `backend/eval/results/` |
| Summarization banner after New Conversation | Normal — session summary in progress | Wait ~10–30s; chat remains enabled (Entry 026) |
| Long first startup | Model pull or embedding download | Allow several minutes on first `./start.sh` |

---

## Repository contents

| Path | Purpose |
|------|---------|
| `backend/` | FastAPI API, local database logic, RAG, memory, meal planning, and tests |
| `frontend/` | React + Vite web interface |
| `scripts/` | Local launch, validation, and evaluation utilities |
| `backend/eval/` | Evaluation runners, rubrics, protocols, and scripted scenarios |
| `my_knowledge/SOURCES.md` | Catalogue of external PDF sources used for the local RAG corpus |
| `start.sh` | One-command local launcher |

Private/local working material such as meeting notes, dissertation drafts, experiment logs, generated screenshots, local databases, vector indexes, and virtual environments is kept outside the public GitHub repository.

---

## Privacy note

NutriCoachAI is designed for **local execution**: profiles, chat history, meal plans, and metrics stay in on-device SQLite. Inference uses local Ollama. RAG embeddings may require a one-time download from HuggingFace Hub. There is no account system, HTTPS deployment, or cloud sync in this prototype.

---

*MSc project — NutriCoachAI / Coach_ChatBot. For academic review and demonstration; not for clinical use.*

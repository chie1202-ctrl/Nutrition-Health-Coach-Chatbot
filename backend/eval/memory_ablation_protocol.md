# SeCom-aligned memory ablation protocol (five modes)

Formal protocol for cross-session memory ablation aligned with SeCom (Maharana et al., 2024) baselines. Separates **three independent metrics**; do not substitute one for another.

## Redesign note (2026-07-30) — primary long-context comparison

**Do not** treat M2-with-global-`MEMORY_BUDGET` vs uncapped M3 as proof that summarisation is more efficient.

**Canonical redesign:**

| Setting | Value |
|---------|-------|
| `MEMORY_BUDGET_ENABLED` | **`false`** (default) — no whole-blob truncate |
| M2 bounds | Per-component: cumulative ≤1200, ≤2×800 session summaries, ≤4 active turns × `ACTIVE_TURN_MAX_CHARS` |
| Modes for fidelity | **M2**, **M3** (full transcript), **M3_MATCH** (recency transcript truncated to M2 length) |
| QA path | `process_eval_qa_message` — **neutral** factual prompt (not nutrition coach) |
| Token/TTFT path | `run_long_context_token_eval.py` / `run_long_context_latency_eval.py` — fixed **8 turns/session**, vary closed-session count |

**Claims allowed:**

1. Under growing history, component-bounded M2 injects fewer tokens than full-transcript M3 (representation / compression, not a global axe).
2. At matched injection length, M2 vs M3_MATCH QA F1 tests whether hybrid summaries beat/lose to recency truncation.
3. M2 vs M3 QA F1 describes the trade-off vs full history.

Harness: `run_locomo_memory_eval.py --modes M2,M3,M3_MATCH --neutral-qa`

---

## Purpose

Compare five memory injection strategies on **LOCOMO** official QA to justify production **M2** (PROD-M2) versus literature baselines **ANC-Zero**, **LIT-RecurSum**, **LIT-SessionRet**, and **ANC-Full**.

**Not claiming:** beat SeCom SOTA; reproduce SeCom segment system.

## Three evaluation axes (must report separately)

| Axis | Question | Harness | Primary metric |
|------|----------|---------|----------------|
| **QA performance** | Does memory support factual QA? | `run_locomo_memory_eval.py` | **QA F1** (token F1 vs gold) |
| **Context cost** | How large is injected memory? | `run_locomo_memory_eval.py` + `run_locomo_token_curve.py` | **Injection tokens** + token vs session-count curve |
| **Latency** | How long until first token / full reply? | `run_locomo_latency_eval.py` (Method A) | **TTFT** + **total latency** |

**Invalid:** inferring latency from token counts; using short-script keyword pass as LOCOMO QA evidence.

## Memory mode definitions (five conditions)

| ID | Mode code | SeCom analogue | Mechanism | RAG |
|----|-----------|----------------|-----------|-----|
| ANC-Zero | `M0` | Zero History | No cross-session memory in prompt | Off |
| LIT-RecurSum | `RECURSUM` | RecurSum | Rolling summary only + current-session turns | Off |
| LIT-SessionRet | `SESSION_RET` | Session-Level | BM25 top-k sessions under 4k token budget | Off |
| PROD-M2 | `M2` | (ours) | Cumulative + recent summaries + active turns (budget-capped) | Off |
| ANC-Full | `M3` | Full History | Full prior transcript, uncapped | Off |

Implementation: [`backend/logic.py`](../logic.py) — `build_memory_context()`, `update_recursum_memory()`, `index_closed_session_for_retrieval()`.

### Locked SessionRet settings

| Setting | Value | Env override |
|---------|-------|--------------|
| Retrieval budget | 4096 tokens | `SESSION_RET_MAX_TOKENS` |
| Top-k sessions | 5 | `SESSION_RET_TOP_K` |
| Retriever | BM25 (local, no embedding index) | — |

## Dataset

| Item | Detail |
|------|--------|
| Source | Official [LoCoMo](https://arxiv.org/abs/2402.17152) `locomo10.json` (snap-research/LoCoMo) |
| Loader | `backend/eval/locomo/load_locomo.py` (download/cache + normalize) |
| QA | Official annotated `qa` pairs (not GPT-4-generated) |
| Subset | Up to 20 conversations stratified by length (short/medium/long); `locomo10.json` contains 10 samples — all used when fewer than 20 exist |
| Seeding | `backend/eval/locomo/seed_locomo.py` → SQLite sessions + per-mode memory update on `close_session` |

**QA scoring:** token-level F1 between model answer and gold (`load_locomo.token_f1`); optional exact match column in JSON.

## Model and runtime config

| Role | Model | Settings |
|------|-------|----------|
| Primary ablation | `deepseek-r1:7b` | `EVAL_REPLY_ENGLISH=true`; RAG off |
| Deployment confirm | `deepseek-r1:8b` | `OLLAMA_REASONING=false`; rerun **M2** + **M3** only (`--confirm-8b`) |

## Procedure (per LOCOMO sample × mode)

1. Reset eval user `LOCOMO_{sample_id}_{mode}` (isolated SQLite state).
2. Seed all dialogue sessions; on each close call `finalize_closed_session_memory()` for the active mode.
3. For each official QA pair: `process_chat_message(..., rag_store=None, memory_mode=m)`; record `qa_f1`, `injection_tokens`, `latency_ms`.
4. **Token curve** (separate harness): truncate to N ∈ {1,2,4,6,8} closed sessions; measure injection tokens only (no chat LLM).
5. **Latency** (separate harness): same session cutoffs; SSE Method A with `repeats=2`.

## Harness commands

### QA F1 + injection tokens (primary)

```bash
# Full subset, 7B (Ollama required for RECURSUM/M2 summaries)
EVAL_REPLY_ENGLISH=true OLLAMA_MODEL=deepseek-r1:7b \
  backend/.venv/bin/python backend/eval/run_locomo_memory_eval.py

# Smoke / offline summaries
LOCOMO_EVAL_FAST=true backend/.venv/bin/python backend/eval/run_locomo_memory_eval.py \
  --samples 1 --max-qa-per-sample 2 --fast
```

### Token vs history length

```bash
LOCOMO_EVAL_FAST=false backend/.venv/bin/python backend/eval/run_locomo_token_curve.py
LOCOMO_EVAL_FAST=true backend/.venv/bin/python backend/eval/run_locomo_token_curve.py --fast
```

### Latency (Method A)

```bash
backend/.venv/bin/python backend/eval/run_locomo_latency_eval.py --repeats 2
```

### 8B confirmation (M2 + M3 only)

```bash
OLLAMA_MODEL=deepseek-r1:8b OLLAMA_REASONING=false \
  backend/.venv/bin/python backend/eval/run_locomo_memory_eval.py --confirm-8b
```

## Output artefacts

Canonical JSON under `backend/eval/results/`:

| File pattern | Contents |
|--------------|----------|
| `locomo_memory_eval_{timestamp}_7b.json` | Per-sample QA F1, tokens, latency |
| `locomo_memory_eval_{timestamp}_8b.json` | 8B confirm (M2, M3) |
| `locomo_token_curve_{timestamp}.json` | Injection tokens vs N sessions × 5 modes |
| `locomo_latency_{timestamp}.json` | TTFT + total latency vs N sessions × 5 modes |

## Thesis tables and figures (Ch5)

**Table — LOCOMO overall (7B subset)**

| Mode | QA F1 ↑ | Mean injection tokens ↓ | Median TTFT (ms) ↓ | Median total latency (ms) ↓ |

**Figure 1** — Injection tokens vs closed-session count (five lines, one per mode).

**Figure 2** — TTFT vs closed-session count (five lines).

Populate from canonical JSON after live runs; short-script M0–M3 tables (Entry 008) remain supplementary domain evidence.

## Preconditions

1. Ollama reachable (`check_ollama_reachable()`)
2. Target model pulled (`deepseek-r1:7b` / `deepseek-r1:8b`)
3. For live RECURSUM/M2 seeding: `LOCOMO_EVAL_FAST=false`
4. No concurrent RAG-heavy jobs during latency runs

## Legacy short-session eval (M0–M3, supplementary)

Nutrition-domain scripts (`scripts.json`, `run_memory_eval.py`) remain valid **supplementary** evidence. They use keyword pass rate, not LOCOMO QA F1. See prior sections in git history / Entry 008 for M1/M1.5 short-session canonical JSON.

Long-context synthetic eval (Entries 041–042) remains for M2 vs M3 cost at scale on synthetic dialogue.

## Related files

- [`locomo/load_locomo.py`](locomo/load_locomo.py) — dataset loader + F1
- [`locomo/seed_locomo.py`](locomo/seed_locomo.py) — dialogue → SQLite
- [`run_locomo_memory_eval.py`](run_locomo_memory_eval.py) — QA F1 runner
- [`run_locomo_token_curve.py`](run_locomo_token_curve.py) — token curve
- [`run_locomo_latency_eval.py`](run_locomo_latency_eval.py) — latency
- [`run_memory_eval.py`](run_memory_eval.py) — legacy short-session ablation
- [`long_context_scenarios.py`](long_context_scenarios.py) — synthetic long-context seeding

## Scope and limitations (Ch5)

- LOCOMO is general long-term dialogue, not nutrition-specific.
- F1 on short gold answers may miss paraphrase; no human eval.
- SessionRet uses BM25 only (locked for local reproducibility).
- Subset size n ≤ 10 from `locomo10.json` unless extended dataset added.

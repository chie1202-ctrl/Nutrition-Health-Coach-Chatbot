# C1 — RAG k-sweep evaluation protocol

## Purpose

Measure **retrieval accuracy** (recall@k, MRR) and **steady-state latency** for `retrieve_rag_context()` across k values, after corpus expansion (C11).

## Scope

| Track | Default | Notes |
|-------|---------|-------|
| **Recall** | English `chat` gold cases | Deterministic — one pass per (case, k) |
| **Latency** | Same subset | Statistical — interleaved k, multiple rounds |
| **Food-choice / meal-plan** | Optional | Separate query types; report in isolation |

Filter with `--locale en` to exclude non-English prompts from headline metrics.

## Preconditions

1. `my_knowledge/*.pdf` present; Chroma index built (`backend/database/vector_db/`).
2. Run from `backend/`: `.venv/bin/python eval/run_rag_k_sweep_eval.py`
3. Do not run other RAG-heavy jobs on the same machine during latency rounds (reduces noise).

## Procedure (harness: `run_rag_k_sweep_eval.py`)

### Phase A — Setup

1. `initialize_rag()` once per run.
2. Log corpus PDF count, chunk count, gold case count, k values.

### Phase B — Warmup (excluded from metrics)

1. **10** retrieval calls with a fixed generic query, `k=4`.
2. Purpose: load HuggingFace embedding weights and warm Chroma query path.
3. **Cold-start** latency (first call after process start) is logged separately as `cold_start_ms` and **not** mixed into k comparison.

### Phase C — Recall@k (deterministic)

1. For each gold case and each k: one `retrieve_rag_context(message, k)`.
2. **Hit** if any `expected_sources` basename appears in retrieved source list.
3. Order of k blocks does **not** affect recall (same results each run).

### Phase D — Latency (steady-state)

1. **Interleaved** design: for each round, for each case, shuffle k ∈ {2,3,4,…} and measure one query per k.
2. Default **5 rounds** → `cases × rounds × |k|` samples per k.
3. Timer: `time.perf_counter()` around `retrieve_rag_context` only (not JSON export).
4. Report per k: mean, median, stdev, p95, min, max.

**Why interleaved?** Measuring all of k=2 before k=3 confounds k with cache/order effects (observed: k=2 looked ~13 ms vs k=3 ~8 ms in block design; ~9 ms each when interleaved).

### Phase E — Recommendation

1. Compare recall@k across k (must be identical or higher at larger k).
2. If recall ties, prefer **lowest median latency** and **smallest k** (shorter LLM context).
3. Production default remains **k=2** unless recall gain at k>2 is demonstrated on gold set.

## Known limitations

| Limitation | Impact |
|------------|--------|
| Gold `expected_sources` are PDF-level, not chunk/page | Lenient recall; does not prove chunk quality |
| Embedding model fixed (`all-MiniLM-L6-v2`) | Results not portable to other embedders |
| No LLM e2e in this harness | Latency excludes ~30 s generation (see C10) |
| Short food-choice queries | Low recall unless query expansion is implemented in `logic.py` |

## Artifacts

- Gold set: `backend/eval/rag_retrieval_gold.json`
- Results: `backend/eval/results/rag_k_sweep_eval_<timestamp>.json`

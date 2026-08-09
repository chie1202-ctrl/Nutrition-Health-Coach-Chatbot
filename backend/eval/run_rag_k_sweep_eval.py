#!/usr/bin/env python3
"""C1: RAG retrieval gold-set eval + k-value sweep (protocol: rag_k_sweep_protocol.md)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Sequence

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import logic  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
GOLD_PATH = os.path.join(EVAL_DIR, "rag_retrieval_gold.json")
DEFAULT_K_VALUES = [2, 3, 4]
WARMUP_ROUNDS = 10
LATENCY_ROUNDS = 5
WARMUP_QUERY = "warmup dietary guidelines sodium vegetables protein physical activity"
FOOD_CHOICE_QUERY_PREFIX = "sodium glycemic takeaway comparison "


def _load_gold() -> List[Dict[str, Any]]:
    with open(GOLD_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _is_english(text: str) -> bool:
    return not any(ord(ch) > 127 for ch in text)


def _filter_gold(
    gold: Sequence[Dict[str, Any]],
    *,
    locale: Optional[str] = None,
    query_types: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    rows = list(gold)
    if query_types:
        allowed = set(query_types)
        rows = [row for row in rows if row.get("query_type") in allowed]
    if locale == "en":
        rows = [row for row in rows if _is_english(str(row.get("message") or ""))]
    return rows


def _resolve_query(case: Dict[str, Any], *, food_choice_expanded: bool = False) -> str:
    message = str(case.get("message") or "").strip()
    if case.get("query_type") == "food_choice" and food_choice_expanded:
        return f"{FOOD_CHOICE_QUERY_PREFIX}{message}".strip()
    return message


def _recall_hit(retrieved: List[str], expected: List[str]) -> bool:
    if not expected:
        return True
    retrieved_set = set(retrieved)
    return any(src in retrieved_set for src in expected)


def _mrr(retrieved: List[str], expected: List[str]) -> float:
    if not expected:
        return 1.0
    expected_set = set(expected)
    for rank, src in enumerate(retrieved, start=1):
        if src in expected_set:
            return 1.0 / rank
    return 0.0


def _latency_stats(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {
            "n": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "stdev_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    ordered = sorted(samples)
    p95_index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "n": len(samples),
        "mean_ms": round(mean(ordered), 2),
        "median_ms": round(median(ordered), 2),
        "stdev_ms": round(pstdev(ordered), 2) if len(ordered) > 1 else 0.0,
        "p95_ms": round(ordered[p95_index], 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
    }


def _summarize_recall(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"cases": 0, "recall_at_k": 0.0, "mrr": 0.0}
    return {
        "cases": len(rows),
        "recall_at_k": round(mean(1.0 if row["recall_hit"] else 0.0 for row in rows), 4),
        "mrr": round(mean(row["mrr"] for row in rows), 4),
    }


def _warmup(rag_store, rounds: int = WARMUP_ROUNDS) -> None:
    for _ in range(rounds):
        logic.retrieve_rag_context(rag_store, WARMUP_QUERY, k=4)


def _measure_cold_start_ms(rag_store) -> float:
    t0 = time.perf_counter()
    logic.retrieve_rag_context(rag_store, WARMUP_QUERY, k=2)
    return round((time.perf_counter() - t0) * 1000, 2)


def _evaluate_recall(
    rag_store,
    gold: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    *,
    food_choice_expanded: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    sweep: Dict[str, List[Dict[str, Any]]] = {}
    for k in k_values:
        rows: List[Dict[str, Any]] = []
        for case in gold:
            message = _resolve_query(case, food_choice_expanded=food_choice_expanded)
            _, sources = logic.retrieve_rag_context(rag_store, message, k=k)
            expected = list(case.get("expected_sources") or [])
            rows.append({
                "case_id": case["id"],
                "query_type": case.get("query_type", "chat"),
                "message": message,
                "k": k,
                "expected_sources": expected,
                "retrieved_sources": sources,
                "recall_hit": _recall_hit(sources, expected),
                "mrr": round(_mrr(sources, expected), 4),
            })
        sweep[str(k)] = rows
    return sweep


def _evaluate_latency_interleaved(
    rag_store,
    gold: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    *,
    rounds: int = LATENCY_ROUNDS,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    samples_by_k: Dict[int, List[float]] = {k: [] for k in k_values}

    for _round in range(rounds):
        for case in gold:
            message = _resolve_query(case)
            ks = list(k_values)
            rng.shuffle(ks)
            for k in ks:
                t0 = time.perf_counter()
                logic.retrieve_rag_context(rag_store, message, k=k)
                samples_by_k[k].append((time.perf_counter() - t0) * 1000)

    return {str(k): _latency_stats(samples_by_k[k]) for k in k_values}


def _recommend_k(
    recall_summary: Dict[str, Dict[str, Any]],
    latency_summary: Dict[str, Dict[str, Any]],
    *,
    baseline_k: int = 2,
) -> Dict[str, Any]:
    best_k = baseline_k
    best_recall = recall_summary.get(str(baseline_k), {}).get("recall_at_k", 0.0)
    best_median = latency_summary.get(str(baseline_k), {}).get("median_ms", 0.0)

    for k_str, recall in recall_summary.items():
        k = int(k_str)
        lat = latency_summary.get(k_str, {})
        recall_at_k = recall.get("recall_at_k", 0.0)
        median_ms = lat.get("median_ms", 0.0)
        if recall_at_k > best_recall + 1e-9:
            best_k = k
            best_recall = recall_at_k
            best_median = median_ms
        elif abs(recall_at_k - best_recall) < 1e-9 and median_ms < best_median - 0.5:
            best_k = k
            best_median = median_ms
        elif (
            abs(recall_at_k - best_recall) < 1e-9
            and abs(median_ms - best_median) < 0.5
            and k < best_k
        ):
            best_k = k

    return {
        "baseline_k": baseline_k,
        "recommended_k": best_k,
        "rationale": "Highest recall@k; tie-break lower median latency; then smaller k",
    }


def run_eval(
    k_values: Optional[List[int]] = None,
    *,
    locale: Optional[str] = "en",
    query_types: Optional[List[str]] = None,
    latency_rounds: int = LATENCY_ROUNDS,
    seed: int = 42,
    include_food_choice_expansion: bool = True,
) -> Dict[str, Any]:
    k_values = k_values or list(DEFAULT_K_VALUES)
    rag_store = logic.initialize_rag()
    if rag_store is None:
        raise RuntimeError("RAG store unavailable — check my_knowledge/ and vector_db/")

    gold_all = _load_gold()
    gold = _filter_gold(gold_all, locale=locale, query_types=query_types)

    corpus_count = rag_store._collection.count() if hasattr(rag_store, "_collection") else 0
    cold_start_ms = _measure_cold_start_ms(rag_store)
    _warmup(rag_store, rounds=WARMUP_ROUNDS)

    recall_by_k = _evaluate_recall(rag_store, gold, k_values)
    recall_summary = {k: _summarize_recall(rows) for k, rows in recall_by_k.items()}
    latency_summary = _evaluate_latency_interleaved(
        rag_store,
        gold,
        k_values,
        rounds=latency_rounds,
        seed=seed,
    )

    food_choice_compare: Optional[Dict[str, Any]] = None
    if include_food_choice_expansion:
        fc_gold = _filter_gold(gold_all, query_types=["food_choice"])
        if fc_gold:
            plain = _evaluate_recall(rag_store, fc_gold, [2], food_choice_expanded=False)["2"]
            expanded = _evaluate_recall(rag_store, fc_gold, [2], food_choice_expanded=True)["2"]
            food_choice_compare = {
                "k": 2,
                "plain": _summarize_recall(plain),
                "expanded_query": _summarize_recall(expanded),
                "cases": [
                    {
                        "case_id": p["case_id"],
                        "plain_retrieved": p["retrieved_sources"],
                        "expanded_retrieved": e["retrieved_sources"],
                        "plain_hit": p["recall_hit"],
                        "expanded_hit": e["recall_hit"],
                    }
                    for p, e in zip(plain, expanded)
                ],
            }

    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "eval": "C1_rag_k_sweep",
        "protocol": "backend/eval/rag_k_sweep_protocol.md",
        "methodology": {
            "warmup_rounds": WARMUP_ROUNDS,
            "warmup_excluded_from_latency": True,
            "latency_design": "interleaved_per_case",
            "latency_rounds": latency_rounds,
            "latency_seed": seed,
            "timer": "time.perf_counter",
            "locale_filter": locale,
            "query_types": list(query_types) if query_types else None,
            "cold_start_ms_first_query": cold_start_ms,
        },
        "corpus_pdf_count": len(
            [name for name in os.listdir(logic.PDF_DIR) if name.lower().endswith(".pdf")]
        ) if os.path.isdir(logic.PDF_DIR) else 0,
        "corpus_chunk_count": corpus_count,
        "gold_cases_total": len(gold_all),
        "gold_cases_evaluated": len(gold),
        "k_values": k_values,
        "production_k_current": 2,
        "recall_summary": recall_summary,
        "latency_summary": latency_summary,
        "recommendation": _recommend_k(recall_summary, latency_summary, baseline_k=2),
        "recall_cases_by_k": recall_by_k,
        "food_choice_query_expansion": food_choice_compare,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="C1 RAG k-sweep eval")
    parser.add_argument("--k-values", default="2,3,4", help="Comma-separated k values")
    parser.add_argument("--locale", default="en", help="Locale filter: en or all")
    parser.add_argument(
        "--query-types",
        default="chat",
        help="Comma-separated query types (chat,food_choice,meal_plan) or 'all'",
    )
    parser.add_argument("--latency-rounds", type=int, default=LATENCY_ROUNDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-food-choice-compare", action="store_true")
    args = parser.parse_args()

    k_values = [int(part.strip()) for part in args.k_values.split(",") if part.strip()]
    locale = None if args.locale == "all" else args.locale
    query_types = None if args.query_types == "all" else [
        part.strip() for part in args.query_types.split(",") if part.strip()
    ]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = run_eval(
        k_values=k_values,
        locale=locale,
        query_types=query_types,
        latency_rounds=args.latency_rounds,
        seed=args.seed,
        include_food_choice_expansion=not args.no_food_choice_compare,
    )
    out_path = os.path.join(RESULTS_DIR, f"rag_k_sweep_eval_{result['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(json.dumps({
        "output": out_path,
        "methodology": result["methodology"],
        "gold_cases_evaluated": result["gold_cases_evaluated"],
        "recall_summary": result["recall_summary"],
        "latency_summary": result["latency_summary"],
        "recommendation": result["recommendation"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

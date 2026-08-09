#!/usr/bin/env python3
"""LoCoMo memory fidelity eval: M2 vs M3 vs M3_MATCH with evidence-aligned QA.

Fixes (2026-07-30):
  - Seed session date/time headers into chat history
  - Select only QA whose evidence sessions are within seeded closed sessions
  - Diversify fact types (early / recent / temporal / preference)
  - Neutral QA prompt with relative-date resolution instructions
  - MEMORY_BUDGET_ENABLED=false; modes M2 / M3 / M3_MATCH

Usage:
  MEMORY_BUDGET_ENABLED=false OLLAMA_MODEL=deepseek-r1:8b OLLAMA_REASONING=false \\
    backend/.venv/bin/python backend/eval/run_locomo_memory_eval.py \\
      --samples 3 --max-closed-sessions 8 --max-qa-per-sample 5 \\
      --modes M2,M3,M3_MATCH --evidence-aligned
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import logic  # noqa: E402

from locomo.load_locomo import (  # noqa: E402
    exact_match,
    get_subset_summary,
    load_locomo_samples,
    normalize_qa_pairs,
    select_evidence_aligned_qa,
    select_stratified_subset,
    token_f1,
)
from locomo.seed_locomo import seed_locomo_conversation  # noqa: E402
from eval_db import add_eval_db_arguments, eval_db_metadata, setup_eval_database  # noqa: E402

RESULTS_DIR = os.path.join(EVAL_DIR, "results")
DEFAULT_MODES = ["M2", "M3", "M3_MATCH"]
EVAL_REPLY_ENGLISH_SUFFIX = "\n\nReply in English only."
_CONTENT_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "and",
    "or",
    "by",
    "for",
    "with",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "his",
    "her",
    "their",
    "from",
    "that",
    "this",
    "it",
    "as",
    "at",
}


def _eval_prompt(question: str) -> str:
    if os.getenv("EVAL_REPLY_ENGLISH", "true").strip().lower() in ("false", "0", "no"):
        return question
    return f"{question}{EVAL_REPLY_ENGLISH_SUFFIX}"


def _apply_fast_mode(fast: bool) -> None:
    if fast:
        logic._invoke_summary_llm = lambda _prompt: ""  # type: ignore[attr-defined]


def _model_tag() -> str:
    model = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
    if "8b" in model.lower():
        return "8b"
    return "7b"


def _content_words(text: str) -> List[str]:
    raw = (text or "").lower().replace(",", " ").replace(".", " ").replace(";", " ")
    return [tok for tok in raw.split() if tok and tok not in _CONTENT_STOPWORDS]


def _is_idk(prediction: str) -> bool:
    pred = (prediction or "").strip().lower()
    if not pred:
        return True
    needles = (
        "i do not know",
        "i don't know",
        "i dont know",
        "do not know",
        "don't know",
        "cannot determine",
        "can't determine",
        "no information",
        "not mentioned",
        "not in the memory",
        "memory does not",
        "memory doesn't",
    )
    return any(n in pred for n in needles)


def _memory_contains_gold(memory_text: str, gold: str) -> bool:
    """Heuristic: enough gold content words appear in injected memory."""
    gold_words = _content_words(gold)
    if not gold_words:
        return False
    mem = (memory_text or "").lower()
    hits = sum(1 for w in gold_words if w in mem)
    if len(gold_words) <= 2:
        return hits == len(gold_words)
    return (hits / len(gold_words)) >= 0.4


def _evidence_keyword_hit(prediction: str, gold: str) -> float:
    gold_words = _content_words(gold)
    if not gold_words:
        return 0.0
    pred = (prediction or "").lower()
    hits = sum(1 for w in gold_words if w in pred)
    return round(hits / len(gold_words), 4)


def _diagnosis_bucket(*, memory_contains_gold: bool, idk: bool) -> str:
    if not memory_contains_gold:
        return "not_in_memory"
    if idk:
        return "in_memory_but_idk"
    return "in_memory_answered"


def _relaxed_label(prediction: str, gold: str, qa_f1: float) -> str:
    """Heuristic Correct / Partial / Incorrect for reporting (not human gold)."""
    if _is_idk(prediction):
        return "Incorrect"
    if exact_match(prediction, gold) or qa_f1 >= 0.75:
        return "Correct"
    if qa_f1 >= 0.25:
        return "Partial"
    gold_toks = set(_content_words(gold))
    pred_toks = set(_content_words(prediction))
    if gold_toks and len(gold_toks & pred_toks) / len(gold_toks) >= 0.4:
        return "Partial"
    return "Incorrect"


def run_sample_modes(
    sample: Dict[str, Any],
    modes: List[str],
    *,
    max_qa_per_sample: int = 5,
    max_closed_sessions: int = 8,
    fast_summaries: bool = False,
    neutral_qa: bool = True,
    evidence_aligned: bool = True,
) -> List[Dict[str, Any]]:
    """Evaluate each mode on an isolated seeded user to avoid QA-turn contamination."""
    if evidence_aligned:
        qa_pairs = select_evidence_aligned_qa(
            sample,
            max_closed_sessions=max_closed_sessions,
            max_qa=max_qa_per_sample,
        )
    else:
        qa_pairs = normalize_qa_pairs(sample)[: max(0, max_qa_per_sample)]

    if not qa_pairs:
        sample_id = str(sample.get("sample_id") or "unknown")
        return [
            {
                "sample_id": sample_id,
                "memory_mode": logic.normalize_memory_mode(mode),
                "user_id": None,
                "closed_session_count": 0,
                "qa_count": 0,
                "mean_qa_f1": 0.0,
                "mean_injection_tokens": 0.0,
                "qa_results": [],
                "warning": "no_evidence_aligned_qa",
            }
            for mode in modes
        ]

    results: List[Dict[str, Any]] = []
    for mode in modes:
        mode = logic.normalize_memory_mode(mode)
        seeded = seed_locomo_conversation(
            sample,
            # M3/M3_MATCH evaluate transcript views, but M3_MATCH must be matched
            # against the same M2 summary context length. Seed isolated users with
            # M2 finalization so summary state exists without sharing QA turns.
            "M2",
            user_label=f"LOCOMO_{sample.get('sample_id')}_{mode}_isolated",
            max_closed_sessions=max_closed_sessions,
            fast_summaries=fast_summaries,
        )
        user_id = seeded["user_id"]
        qa_rows: List[Dict[str, Any]] = []
        for qa in qa_pairs:
            question = _eval_prompt(qa["question"])
            started = time.perf_counter()
            if neutral_qa:
                response = logic.process_eval_qa_message(
                    user_id,
                    question,
                    memory_mode=mode,
                    force_new_session=True,
                    persist=False,
                )
            else:
                response = logic.process_chat_message(
                    user_id,
                    question,
                    rag_store=None,
                    force_new_session=True,
                    memory_mode=mode,
                )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            reply = logic.strip_think_tags(response.get("reply") or "")
            memory_used = response.get("memory_used") or {}
            memory_text = response.get("memory_text") or ""
            memory_chars = response.get("memory_chars")
            if memory_chars is None:
                memory_chars = len(memory_text)
            f1 = token_f1(reply, qa["answer"])
            gold = qa["answer"]
            contains_gold = _memory_contains_gold(memory_text, gold)
            idk = _is_idk(reply)
            keyword_hit = _evidence_keyword_hit(reply, gold)
            bucket = _diagnosis_bucket(memory_contains_gold=contains_gold, idk=idk)
            qa_rows.append(
                {
                    "qa_index": qa["qa_index"],
                    "question": qa["question"],
                    "gold_answer": gold,
                    "prediction": reply,
                    "qa_f1": f1,
                    "exact_match": exact_match(reply, gold),
                    "relaxed_label": _relaxed_label(reply, gold, f1),
                    "is_idk": idk,
                    "memory_contains_gold": contains_gold,
                    "evidence_keyword_hit": keyword_hit,
                    "diagnosis_bucket": bucket,
                    "category": qa.get("category"),
                    "fact_type": qa.get("fact_type"),
                    "evidence": qa.get("evidence"),
                    "evidence_session_ids": qa.get("evidence_session_ids"),
                    "max_evidence_session": qa.get("max_evidence_session"),
                    "injection_tokens": memory_used.get("estimated_memory_tokens"),
                    "memory_chars": memory_chars,
                    "matched_budget_chars": memory_used.get("matched_budget_chars"),
                    "latency_ms": latency_ms,
                    "eval_neutral_prompt": bool(response.get("eval_neutral_prompt") or neutral_qa),
                }
            )

        f1_scores = [row["qa_f1"] for row in qa_rows]
        tokens = [row["injection_tokens"] for row in qa_rows if row.get("injection_tokens") is not None]
        n = max(1, len(qa_rows))
        results.append(
            {
                "sample_id": seeded["sample_id"],
                "memory_mode": mode,
                "user_id": user_id,
                "closed_session_count": seeded["closed_session_count"],
                "qa_count": len(qa_rows),
                "mean_qa_f1": round(statistics.mean(f1_scores), 4) if f1_scores else 0.0,
                "mean_injection_tokens": round(statistics.mean(tokens), 1) if tokens else 0.0,
                "relaxed_counts": {
                    label: sum(1 for row in qa_rows if row["relaxed_label"] == label)
                    for label in ("Correct", "Partial", "Incorrect")
                },
                "idk_rate": round(sum(1 for row in qa_rows if row["is_idk"]) / n, 4),
                "memory_contains_gold_rate": round(
                    sum(1 for row in qa_rows if row["memory_contains_gold"]) / n, 4
                ),
                "diagnosis_counts": {
                    "not_in_memory": sum(1 for row in qa_rows if row["diagnosis_bucket"] == "not_in_memory"),
                    "in_memory_but_idk": sum(
                        1 for row in qa_rows if row["diagnosis_bucket"] == "in_memory_but_idk"
                    ),
                    "in_memory_answered": sum(
                        1 for row in qa_rows if row["diagnosis_bucket"] == "in_memory_answered"
                    ),
                },
                "mean_evidence_keyword_hit": round(
                    statistics.mean([row["evidence_keyword_hit"] for row in qa_rows]), 4
                )
                if qa_rows
                else 0.0,
                "qa_results": qa_rows,
            }
        )
    return results


def aggregate_by_mode(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["memory_mode"]].append(row)
    summary: List[Dict[str, Any]] = []
    for mode, items in sorted(grouped.items()):
        f1_values = [item["mean_qa_f1"] for item in items]
        token_values = [item["mean_injection_tokens"] for item in items]
        relaxed = {"Correct": 0, "Partial": 0, "Incorrect": 0}
        diagnosis = {"not_in_memory": 0, "in_memory_but_idk": 0, "in_memory_answered": 0}
        idk_rates: List[float] = []
        contains_rates: List[float] = []
        keyword_hits: List[float] = []
        for item in items:
            for label, count in (item.get("relaxed_counts") or {}).items():
                relaxed[label] = relaxed.get(label, 0) + int(count)
            for label, count in (item.get("diagnosis_counts") or {}).items():
                diagnosis[label] = diagnosis.get(label, 0) + int(count)
            if item.get("idk_rate") is not None:
                idk_rates.append(float(item["idk_rate"]))
            if item.get("memory_contains_gold_rate") is not None:
                contains_rates.append(float(item["memory_contains_gold_rate"]))
            if item.get("mean_evidence_keyword_hit") is not None:
                keyword_hits.append(float(item["mean_evidence_keyword_hit"]))
        qa_total = sum(diagnosis.values()) or 1
        summary.append(
            {
                "memory_mode": mode,
                "sample_count": len(items),
                "mean_qa_f1": round(statistics.mean(f1_values), 4) if f1_values else 0.0,
                "mean_injection_tokens": round(statistics.mean(token_values), 1) if token_values else 0.0,
                "relaxed_counts": relaxed,
                "mean_idk_rate": round(statistics.mean(idk_rates), 4) if idk_rates else 0.0,
                "mean_memory_contains_gold_rate": round(statistics.mean(contains_rates), 4)
                if contains_rates
                else 0.0,
                "mean_evidence_keyword_hit": round(statistics.mean(keyword_hits), 4) if keyword_hits else 0.0,
                "diagnosis_counts": diagnosis,
                "diagnosis_rates": {
                    key: round(val / qa_total, 4) for key, val in diagnosis.items()
                },
            }
        )
    return summary


def aggregate_by_fact_type(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean F1 per mode × fact_type across all QA items."""
    buckets: Dict[tuple, List[float]] = defaultdict(list)
    for run in rows:
        mode = run["memory_mode"]
        for qa in run.get("qa_results") or []:
            fact_type = qa.get("fact_type") or "unknown"
            buckets[(mode, fact_type)].append(float(qa.get("qa_f1") or 0.0))
    out: List[Dict[str, Any]] = []
    for (mode, fact_type), values in sorted(buckets.items()):
        out.append(
            {
                "memory_mode": mode,
                "fact_type": fact_type,
                "qa_count": len(values),
                "mean_qa_f1": round(statistics.mean(values), 4) if values else 0.0,
            }
        )
    return out


def write_results(payload: Dict[str, Any], model_tag: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = payload.get("generated_at") or datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"locomo_memory_eval_{stamp}_{model_tag}.json")
    csv_path = os.path.join(RESULTS_DIR, f"locomo_memory_eval_{stamp}_{model_tag}.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "memory_mode",
                "mean_qa_f1",
                "mean_injection_tokens",
                "qa_count",
                "closed_session_count",
                "correct",
                "partial",
                "incorrect",
            ],
        )
        writer.writeheader()
        for row in payload.get("runs", []):
            rc = row.get("relaxed_counts") or {}
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "memory_mode": row["memory_mode"],
                    "mean_qa_f1": row["mean_qa_f1"],
                    "mean_injection_tokens": row["mean_injection_tokens"],
                    "qa_count": row["qa_count"],
                    "closed_session_count": row["closed_session_count"],
                    "correct": rc.get("Correct", 0),
                    "partial": rc.get("Partial", 0),
                    "incorrect": rc.get("Incorrect", 0),
                }
            )
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo evidence-aligned memory fidelity eval")
    parser.add_argument("--samples", type=int, default=3, help="Stratified subset size")
    parser.add_argument("--max-qa-per-sample", type=int, default=5, help="Evidence-aligned QA per conversation")
    parser.add_argument(
        "--max-closed-sessions",
        type=int,
        default=8,
        help="Seed this many LoCoMo sessions before QA",
    )
    parser.add_argument("--fast", action="store_true", help="Fallback summaries (no Ollama for summary)")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated memory modes")
    parser.add_argument("--neutral-qa", action="store_true", default=True, help="Neutral factual QA (default on)")
    parser.add_argument("--coach-qa", action="store_true", help="Use production coach prompt")
    parser.add_argument(
        "--evidence-aligned",
        action="store_true",
        default=True,
        help="Only QA whose evidence is inside seeded sessions (default on)",
    )
    parser.add_argument(
        "--no-evidence-aligned",
        action="store_true",
        help="Disable evidence filter (legacy first-N QA)",
    )
    add_eval_db_arguments(parser)
    args = parser.parse_args()

    os.environ.setdefault("MEMORY_BUDGET_ENABLED", "false")
    os.environ.setdefault("OLLAMA_MODEL", "deepseek-r1:8b")
    os.environ.setdefault("OLLAMA_REASONING", "false")

    modes = [m.strip().upper().replace("-", "_") for m in args.modes.split(",") if m.strip()]
    modes = [logic.normalize_memory_mode(mode) for mode in modes]
    seen = set()
    modes = [m for m in modes if not (m in seen or seen.add(m))]

    fast = args.fast or os.getenv("LOCOMO_EVAL_FAST", "false").strip().lower() in ("true", "1", "yes")
    neutral_qa = not args.coach_qa
    evidence_aligned = not args.no_evidence_aligned
    _apply_fast_mode(fast)

    if not fast and not logic.check_ollama_reachable():
        print("ERROR: Ollama not reachable. Start ollama serve or pass --fast for smoke tests.")
        sys.exit(1)

    setup_eval_database(args)
    all_samples = load_locomo_samples()
    subset = select_stratified_subset(all_samples, target_n=args.samples)

    runs: List[Dict[str, Any]] = []
    for sample in subset:
        sample_id = sample.get("sample_id")
        preview = select_evidence_aligned_qa(
            sample,
            max_closed_sessions=args.max_closed_sessions,
            max_qa=args.max_qa_per_sample,
        ) if evidence_aligned else normalize_qa_pairs(sample)[: args.max_qa_per_sample]
        print(
            f"Seeding {sample_id} ({args.max_closed_sessions} sessions); "
            f"QA={len(preview)} evidence-aligned; modes={modes}...",
            flush=True,
        )
        for qa in preview:
            print(
                f"  - [{qa.get('fact_type')}] D{qa.get('evidence_session_ids')} "
                f"Q{qa['qa_index']}: {qa['question'][:70]}",
                flush=True,
            )
        runs.extend(
            run_sample_modes(
                sample,
                modes,
                max_qa_per_sample=args.max_qa_per_sample,
                max_closed_sessions=args.max_closed_sessions,
                fast_summaries=fast,
                neutral_qa=neutral_qa,
                evidence_aligned=evidence_aligned,
            )
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "generated_at": stamp,
        "eval": "locomo_memory_eval",
        "design": "locked_m2_m3_m3match_diag_prompt_v1",
        "locked_setup": {
            "modes": ["M2", "M3", "M3_MATCH"],
            "memory_mechanisms_unchanged": True,
            "changes": [
                "stricter eval QA prompt (answer if evidence present)",
                "diagnostics: memory_contains_gold / idk / diagnosis_bucket",
                "isolated eval DB + persist=False",
            ],
        },
        "model": os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
        "model_tag": _model_tag(),
        "rag_enabled": False,
        "eval_reply_english": os.getenv("EVAL_REPLY_ENGLISH", "true"),
        "neutral_qa": neutral_qa,
        "evidence_aligned": evidence_aligned,
        "memory_budget_enabled": logic.memory_budget_enabled(),
        "locomo_eval_fast": fast,
        "max_closed_sessions": args.max_closed_sessions,
        "max_qa_per_sample": args.max_qa_per_sample,
        "subset": get_subset_summary(all_samples, target_n=args.samples),
        "modes": modes,
        **eval_db_metadata(),
        "aggregate_by_mode": aggregate_by_mode(runs),
        "aggregate_by_fact_type": aggregate_by_fact_type(runs),
        "runs": runs,
    }
    output_path = write_results(payload, _model_tag())
    print(f"Wrote {output_path}")
    print("Aggregate by mode:")
    for row in payload["aggregate_by_mode"]:
        rc = row.get("relaxed_counts") or {}
        diag = row.get("diagnosis_rates") or {}
        print(
            f"  {row['memory_mode']}: F1={row['mean_qa_f1']:.4f} "
            f"tokens≈{row['mean_injection_tokens']} "
            f"C/P/I={rc.get('Correct', 0)}/{rc.get('Partial', 0)}/{rc.get('Incorrect', 0)} "
            f"idk={row.get('mean_idk_rate', 0):.2f} "
            f"mem_has_gold={row.get('mean_memory_contains_gold_rate', 0):.2f} "
            f"buckets={diag}"
        )
    print("Aggregate by fact type:")
    for row in payload["aggregate_by_fact_type"]:
        print(
            f"  {row['memory_mode']} / {row['fact_type']}: "
            f"F1={row['mean_qa_f1']:.4f} (n={row['qa_count']})"
        )


if __name__ == "__main__":
    main()

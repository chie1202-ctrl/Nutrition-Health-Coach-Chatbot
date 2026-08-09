#!/usr/bin/env python3
"""C10: Production-path RAG on vs off latency benchmark (8B, M2)."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import logic  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

C10_PROMPTS = [
    {
        "id": "P01_goal_recall",
        "message": "What was my goal again, and what food should I avoid?",
        "expected_keywords": ["shellfish"],
    },
    {
        "id": "P02_breakfast_advice",
        "message": "Last time we discussed my breakfast drink problem. What should I try now?",
        "expected_keywords": ["milk", "lactose", "soy"],
    },
    {
        "id": "P03_rag_vegetables",
        "message": "What vegetables do dietary guidelines recommend eating more of?",
        "expected_keywords": ["vegetable", "vegetables", "fiber"],
    },
    {
        "id": "P04_rag_activity",
        "message": "According to physical activity guidelines, how much moderate activity should adults get per week?",
        "expected_keywords": ["150", "minute", "moderate", "activity", "week"],
    },
    {
        "id": "P05_t2dm_breakfast",
        "message": "I'm 63 with Type 2 diabetes. What should I eat for breakfast to keep blood sugar stable?",
        "expected_keywords": ["protein", "fiber", "breakfast", "blood sugar", "diabetes"],
    },
]

EVAL_REPLY_ENGLISH_SUFFIX = "\n\nReply in English only."


def _eval_message(prompt: str) -> str:
    if os.getenv("EVAL_REPLY_ENGLISH", "true").strip().lower() in ("false", "0", "no"):
        return prompt
    return f"{prompt}{EVAL_REPLY_ENGLISH_SUFFIX}"


def _keyword_score(reply: str, keywords: List[str]) -> float:
    low = logic.strip_think_tags(reply or "").lower()
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if kw.lower() in low)
    return hits / len(keywords)


def _reset_eval_user(name: str) -> int:
    for user in logic.get_all_users():
        if user.get("name") == name:
            logic.delete_user(int(user["user_id"]))
    return logic.create_user_profile(
        name=name,
        gender="female",
        birth_date="1990-01-01",
        height_cm=165.0,
        initial_weight_kg=68.0,
        goal="lose_weight",
        diet_preference="high_protein",
        allergies=["shellfish"],
        medical_conditions=["Type 2 diabetes"],
    )


def _seed_m2_memory(user_id: int) -> None:
    """One closed session + summary so M2 prompt resembles production eval."""
    session_id, _ = logic.resolve_session(user_id, force_new=True)
    logic.save_chat(
        user_id,
        "user",
        "I want to lose 5 kg in three months. I prefer high-protein lunches and I cannot eat shellfish.",
        session_id=session_id,
    )
    logic.save_chat(
        user_id,
        "assistant",
        "Great goal. We can focus on steady calorie deficit, protein-rich lunches, and shellfish-free meals.",
        session_id=session_id,
    )
    logic.close_session(session_id, user_id, trigger_summarization=False)
    logic.summarize_session(session_id, user_id)


def _run_single(
    user_id: int,
    prompt: Dict[str, Any],
    rag_store,
    rag_label: str,
) -> Dict[str, Any]:
    message = _eval_message(prompt["message"])
    rag_retrieval_ms: Optional[float] = None
    rag_context_len = 0
    sources: List[str] = []

    if rag_store is not None:
        t0 = time.time()
        rag_context, sources = logic.retrieve_rag_context(rag_store, message, k=2)
        rag_retrieval_ms = round((time.time() - t0) * 1000, 1)
        rag_context_len = len(rag_context or "")

    t0 = time.time()
    error = None
    reply = ""
    result_sources: List[str] = []
    try:
        result = logic.process_chat_message(
            user_id,
            message,
            rag_store=rag_store,
            force_new_session=True,
            memory_mode="M2",
        )
        reply = result.get("reply") or ""
        result_sources = result.get("sources") or []
    except Exception as exc:
        error = str(exc)
    latency_ms = round((time.time() - t0) * 1000, 1)

    visible = logic.strip_think_tags(reply)
    return {
        "prompt_id": prompt["id"],
        "rag_mode": rag_label,
        "latency_ms": latency_ms,
        "rag_retrieval_ms": rag_retrieval_ms,
        "rag_context_chars": rag_context_len,
        "sources_count": len(result_sources),
        "sources": result_sources,
        "keyword_score": _keyword_score(visible, prompt.get("expected_keywords", [])),
        "reply_len": len(visible),
        "reply_preview": visible[:240],
        "error": error,
    }


def run_c10_eval() -> Dict[str, Any]:
    if not logic.check_ollama_reachable():
        raise RuntimeError("Ollama is not reachable at http://127.0.0.1:11434")

    rag_store = logic.initialize_rag()
    rag_ready = rag_store is not None

    user_id = _reset_eval_user("C10_RAG_Eval")
    _seed_m2_memory(user_id)

    runs: List[Dict[str, Any]] = []
    for prompt in C10_PROMPTS:
        runs.append(_run_single(user_id, prompt, rag_store=None, rag_label="off"))
        if rag_ready:
            runs.append(_run_single(user_id, prompt, rag_store=rag_store, rag_label="on"))
        else:
            runs.append({
                "prompt_id": prompt["id"],
                "rag_mode": "on",
                "skipped": True,
                "reason": "RAG store unavailable (initialize_rag returned None)",
            })

    logic.delete_user(user_id)

    def _avg(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [float(row[key]) for row in rows if row.get(key) is not None and not row.get("error")]
        return round(sum(values) / len(values), 1) if values else None

    off_rows = [r for r in runs if r.get("rag_mode") == "off" and not r.get("error")]
    on_rows = [r for r in runs if r.get("rag_mode") == "on" and not r.get("error") and not r.get("skipped")]

    summary = {
        "rag_off_avg_latency_ms": _avg(off_rows, "latency_ms"),
        "rag_on_avg_latency_ms": _avg(on_rows, "latency_ms"),
        "rag_off_avg_keyword": round(
            sum(r["keyword_score"] for r in off_rows) / len(off_rows), 3
        ) if off_rows else None,
        "rag_on_avg_keyword": round(
            sum(r["keyword_score"] for r in on_rows) / len(on_rows), 3
        ) if on_rows else None,
        "rag_on_avg_retrieval_ms": _avg(on_rows, "rag_retrieval_ms"),
        "rag_on_avg_sources": _avg(on_rows, "sources_count"),
        "latency_delta_ms": None,
    }
    if summary["rag_off_avg_latency_ms"] is not None and summary["rag_on_avg_latency_ms"] is not None:
        summary["latency_delta_ms"] = round(
            summary["rag_on_avg_latency_ms"] - summary["rag_off_avg_latency_ms"], 1
        )

    return {
        "eval_id": "C10_rag_latency",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "ollama_model": logic.get_ollama_chat_model_name(),
        "ollama_reasoning": os.getenv("OLLAMA_REASONING", "false"),
        "memory_mode": "M2",
        "rag_ready": rag_ready,
        "prompt_count": len(C10_PROMPTS),
        "summary": summary,
        "runs": runs,
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = run_c10_eval()
    out_path = os.path.join(RESULTS_DIR, f"rag_latency_eval_{result['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    summary = result["summary"]
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

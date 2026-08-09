"""Shared synthetic multi-session scenarios for long-context memory evals."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import logic

# (closed_sessions, turns_per_closed_session) — turns fixed so only session count varies
DEFAULT_SCENARIOS: List[Tuple[int, int]] = [
    (1, 8),
    (2, 8),
    (4, 8),
    (6, 8),
    (8, 8),
]

USER_LINE = (
    "I want to lose 5 kg over three months. I prefer high-protein lunches, "
    "avoid shellfish, and I walked 30 minutes today. I had chicken salad for lunch. "
)
ASST_LINE = (
    "Good progress on your goal. Keep a steady calorie deficit, prioritize protein at lunch, "
    "and stay shellfish-free. Consider adding vegetables at dinner and logging weight weekly. "
)

RECALL_PROMPT = (
    "What was my goal and what dietary restrictions did we discuss? "
    "Reply in English only."
)

RECALL_KEYWORDS = ["5", "kg", "shellfish"]


def reset_user(label: str) -> int:
    for user in logic.get_all_users():
        if user.get("name") == label:
            logic.delete_user(int(user["user_id"]))
    return logic.create_user_profile(
        name=label,
        gender="female",
        birth_date="1990-01-01",
        height_cm=165.0,
        initial_weight_kg=68.0,
        goal="lose_weight",
        diet_preference="balanced",
        allergies=["shellfish"],
    )


def seed_closed_sessions(
    user_id: int,
    num_sessions: int,
    turns_per_session: int,
    *,
    unique_salt: str = "",
) -> List[int]:
    closed: List[int] = []
    for session_idx in range(num_sessions):
        session_id, _ = logic.resolve_session(user_id, force_new=True)
        for turn_idx in range(turns_per_session):
            role = "user" if turn_idx % 2 == 0 else "assistant"
            content = USER_LINE if role == "user" else ASST_LINE
            # Optional salt breaks identical prefixes across latency repeats (Ollama prompt cache).
            if unique_salt and role == "user":
                content = f"[seed:{unique_salt}:s{session_idx}:t{turn_idx}] {content}"
            repeat = 1 + (turn_idx // 2)
            logic.save_chat(user_id, role, content * repeat, session_id=session_id)
        logic.close_session(session_id, user_id, trigger_summarization=False)
        if turns_per_session >= 2:
            logic.summarize_session(session_id, user_id)
            logic.maybe_rollup_memory(user_id)
        closed.append(session_id)
    return closed


def seed_scenario(
    num_closed_sessions: int,
    turns_per_session: int,
    label_prefix: str = "LongCtx",
    *,
    unique_salt: str = "",
) -> Dict[str, Any]:
    label = f"{label_prefix}_{num_closed_sessions}s_{turns_per_session}t"
    user_id = reset_user(label)
    logic.ensure_user_memory_state(user_id)
    closed = seed_closed_sessions(
        user_id,
        num_closed_sessions,
        turns_per_session,
        unique_salt=unique_salt,
    )
    active_session_id, _ = logic.resolve_session(user_id, force_new=True)
    logic.save_chat(user_id, "user", "What was my goal again?", session_id=active_session_id)

    state = logic.get_user_memory_state(user_id)
    summaries = logic.list_user_summaries(user_id, limit=20)
    session_summaries = [s for s in summaries if s.get("summary_type") == "session"]
    rollup_count = sum(1 for s in summaries if s.get("summary_type") == "rollup")

    conn = logic.get_conn()
    total_chat_rows = conn.execute(
        "SELECT COUNT(*) FROM Chat_History WHERE user_id = ?",
        (int(user_id),),
    ).fetchone()[0]
    conn.close()

    return {
        "label": label,
        "user_id": user_id,
        "active_session_id": active_session_id,
        "closed_session_ids": closed,
        "scenario": {
            "closed_sessions": num_closed_sessions,
            "turns_per_closed_session": turns_per_session,
            "total_chat_rows": total_chat_rows,
            "session_summary_count": len(session_summaries),
            "rollup_count": rollup_count,
            "cumulative_summary_chars": len(state.get("cumulative_summary") or ""),
        },
    }


def measure_memory_tokens(user_id: int, active_session_id: int, mode: str) -> Dict[str, Any]:
    ctx = logic.build_memory_context(user_id, active_session_id, memory_mode=mode)
    used = ctx.get("memory_used") or {}
    text = ctx.get("memory_text") or ""
    budget = logic.memory_budget_chars()
    return {
        "memory_mode": mode,
        "estimated_memory_tokens": used.get("estimated_memory_tokens"),
        "memory_chars": len(text),
        "memory_budget_chars": budget,
        "memory_budget_enabled": logic.memory_budget_enabled(),
        "exceeds_budget_threshold": logic.memory_budget_enabled() and len(text) >= budget,
        "truncated_by_budget": bool(used.get("truncated_by_global_budget")),
        "matched_budget_chars": used.get("matched_budget_chars"),
        "cumulative_included": used.get("cumulative_summary_included"),
        "recent_session_summaries_count": used.get("recent_session_summaries_count"),
        "full_transcript_included": used.get("full_transcript_included"),
    }


def keyword_pass_rate(text: str, keywords: List[str]) -> Dict[str, Any]:
    low = (text or "").lower()
    checks = {kw: kw.lower() in low for kw in keywords}
    total = len(keywords)
    passed = sum(1 for ok in checks.values() if ok)
    return {
        "keyword_checks": checks,
        "keyword_pass_rate": round(passed / total, 3) if total else 0.0,
    }

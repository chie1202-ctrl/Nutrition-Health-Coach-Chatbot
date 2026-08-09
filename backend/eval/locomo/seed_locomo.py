#!/usr/bin/env python3
"""Seed LOCOMO dialogue into SQLite sessions for memory-mode eval harnesses."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCOMO_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import logic  # noqa: E402

from locomo.load_locomo import parse_sessions  # noqa: E402


def reset_eval_user(name: str) -> int:
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
        diet_preference="balanced",
        allergies=[],
    )


def seed_locomo_conversation(
    sample: Dict[str, Any],
    memory_mode: str,
    *,
    user_label: Optional[str] = None,
    max_closed_sessions: Optional[int] = None,
    fast_summaries: bool = False,
) -> Dict[str, Any]:
    """Import LOCOMO sessions, close each, and apply per-mode memory updates."""
    if fast_summaries:
        logic._invoke_summary_llm = lambda _prompt: ""  # type: ignore[attr-defined]

    sample_id = str(sample.get("sample_id") or "unknown")
    mode = logic.normalize_memory_mode(memory_mode)
    label = user_label or f"LOCOMO_{sample_id}_{mode}"
    user_id = reset_eval_user(label)
    logic.ensure_user_memory_state(user_id)
    logic.clear_session_memory_index(user_id)

    sessions = parse_sessions(sample)
    if max_closed_sessions is not None:
        sessions = sessions[: max(0, int(max_closed_sessions))]

    closed_ids: List[int] = []
    for session in sessions:
        session_id, _ = logic.resolve_session(user_id, force_new=True)
        date_time = session.get("date_time")
        speaker_a = session.get("speaker_a") or "speaker_a"
        speaker_b = session.get("speaker_b") or "speaker_b"
        if date_time:
            logic.save_chat(
                user_id,
                "assistant",
                f"[Session date/time: {date_time}]",
                session_id=session_id,
            )
        for turn in session["turns"]:
            speaker = turn.get("speaker") or (
                speaker_a if turn.get("role") == "user" else speaker_b
            )
            # Keep LoCoMo speaker names visible (USER/ASSISTANT alone loses Jon/Gina identity).
            content = f"{speaker}: {turn['content']}"
            logic.save_chat(user_id, turn["role"], content, session_id=session_id)
        logic.close_session(session_id, user_id, trigger_summarization=False)
        logic.finalize_closed_session_memory(session_id, user_id, memory_mode=mode)
        closed_ids.append(session_id)

    qa_session_id, _ = logic.resolve_session(user_id, force_new=True)
    state = logic.get_user_memory_state(user_id)

    return {
        "user_id": user_id,
        "user_label": label,
        "sample_id": sample_id,
        "memory_mode": mode,
        "closed_session_ids": closed_ids,
        "qa_session_id": qa_session_id,
        "closed_session_count": len(closed_ids),
        "recursum_summary_chars": len(state.get("recursum_summary") or ""),
        "cumulative_summary_chars": len(state.get("cumulative_summary") or ""),
    }


def measure_injection_tokens(
    user_id: int,
    active_session_id: int,
    memory_mode: str,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    ctx = logic.build_memory_context(
        user_id,
        active_session_id,
        memory_mode=memory_mode,
        query=query,
    )
    used = ctx.get("memory_used") or {}
    text = ctx.get("memory_text") or ""
    return {
        "memory_mode": logic.normalize_memory_mode(memory_mode),
        "estimated_memory_tokens": used.get("estimated_memory_tokens"),
        "memory_chars": len(text),
        "memory_used": used,
    }

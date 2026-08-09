#!/usr/bin/env python3
"""Seed Adem / Steven / Emily via HTTP API and record full Q&A transcripts.

Does not import backend.logic (avoids double-loading embeddings / killing uvicorn).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

API = "http://127.0.0.1:8000"
OUT_DIR = Path(__file__).resolve().parents[1] / "backend" / "eval" / "results"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

PERSONAS = [
    {
        "key": "adem",
        "profile": {
            "name": "Adem",
            "gender": "male",
            "birth_date": "1983-03-12",
            "height_cm": 180.0,
            "weight_kg": 87.5,
            "goal": "Lose weight safely and manage stress-related overeating",
            "activity_level": "very_active",
            "diet_preference": "no_preference",
            "budget_level": "moderate",
            "medical_conditions": [],
            "allergies": [],
            "food_dislikes": [],
            "target_weight": "80 kg",
            "target_timeline": "6 months",
            "self_description": (
                "White male, age 43, married with two young children (ages 4 and 5). "
                "Former engineer who built a System One; now works as a lumberjack. "
                "Does extreme sports. Gains weight quickly when stressed because he eats a lot."
            ),
            "coach_notes": (
                "Primary goal: sustainable weight loss. High physical job + extreme sports. "
                "Key risk: stress-triggered overeating and rapid weight regain. "
                "Focus on calorie guidance, realistic exercise load, and stress-eating strategies."
            ),
        },
        "questions": [
            {"id": "A1", "text": "How can I lose weight in a safe and realistic way?"},
            {
                "id": "A2",
                "text": (
                    "Based on my height, weight and work pattern, "
                    "how many calories should I aim to eat per day?"
                ),
            },
            {
                "id": "A3",
                "text": (
                    "If I want to lose weight, how much exercise or activity "
                    "do I need each week?"
                ),
            },
            {
                "id": "A4",
                "text": (
                    "I tend to eat a lot when I am stressed. "
                    "What can I do to avoid gaining weight during stressful periods?"
                ),
            },
        ],
    },
    {
        "key": "steven",
        "profile": {
            "name": "Steven",
            "gender": "male",
            "birth_date": "1963-01-15",
            "height_cm": 190.0,
            "weight_kg": 108.3,
            "goal": "Manage Type 2 diabetes and reduce chronic pain/inflammation",
            "activity_level": "sedentary",
            "diet_preference": "mediterranean",
            "budget_level": "moderate",
            "medical_conditions": [
                "Type 2 diabetes",
                "Lumbar disc herniation (L4-L5)",
                "Chronic lower back pain",
                "Depression",
            ],
            "allergies": [],
            "food_dislikes": [],
            "target_weight": "100 kg",
            "target_timeline": "6 months",
            "self_description": (
                "Asian male, age 63, divorced and lives alone. Three adult children "
                "(ages 22, 23, 26). Office job working from home — sits at a computer "
                "all day with no regular exercise. Occasionally travels ~2 hours by train "
                "for collaborator meetings."
            ),
            "coach_notes": (
                "Primary coaching focus: T2DM glycemic management and lowering "
                "pain/inflammation. Sedentary WFH lifestyle; no commute. Limited mobility "
                "due to L4-L5 disc issues — favor low-impact, spine-safe activity. "
                "Monitor mood/depression sensitively."
            ),
        },
        "questions": [
            {
                "id": "S1",
                "text": "How should I change my diet to better manage Type 2 diabetes?",
            },
            {
                "id": "S2",
                "text": (
                    "I work from home and sit at my computer all day. "
                    "What daily routine changes would help my health?"
                ),
            },
            {
                "id": "S3",
                "text": (
                    "I have chronic L4/L5 back pain. What kinds of food or lifestyle "
                    "habits may help reduce inflammation risk?"
                ),
            },
            {
                "id": "S4",
                "text": (
                    "Can you suggest a simple one-day eating and movement plan that is "
                    "safe for my diabetes and back pain?"
                ),
            },
        ],
    },
    {
        "key": "emily",
        "profile": {
            "name": "Emily",
            "gender": "female",
            "birth_date": "1989-06-20",
            "height_cm": 160.0,
            "weight_kg": 66.6,
            "goal": "Resolve long-term constipation while managing IBS carefully",
            "activity_level": "lightly_active",
            "diet_preference": "no_preference",
            "budget_level": "moderate",
            "medical_conditions": [
                "IBS",
                "Chronic constipation (4 years)",
            ],
            "allergies": [],
            "food_dislikes": [],
            "target_weight": "",
            "target_timeline": "",
            "self_description": (
                "Pakistani female, age 37, mother of a 4-year-old daughter. "
                "Has IBS and has experienced constipation for 4 years."
            ),
            "coach_notes": (
                "Primary goal: relieve constipation. IBS present — increase fibre gradually "
                "and flag FODMAP-sensitive foods. Advise when to seek medical care for "
                "long-standing constipation rather than diet-only changes."
            ),
        },
        "questions": [
            {"id": "E1", "text": "What can I eat to help with constipation?"},
            {
                "id": "E2",
                "text": (
                    "I have IBS. What foods should I be careful with "
                    "when trying to increase fibre?"
                ),
            },
            {"id": "E3", "text": "How much water and fibre should I aim for each day?"},
            {
                "id": "E4",
                "text": (
                    "Since I have had constipation for four years, when should I seek "
                    "medical advice instead of only changing my diet?"
                ),
            },
        ],
    },
]


def http_json(method: str, path: str, body: dict | None = None, timeout: int = 60) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def wait_health(attempts: int = 90) -> dict:
    last_err = None
    for i in range(attempts):
        try:
            h = http_json("GET", "/health", timeout=5)
            if h.get("status") == "ok":
                return h
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1)
    raise RuntimeError(f"Backend not healthy after {attempts}s: {last_err}")


def upsert_persona(profile: dict) -> tuple[int, dict]:
    users = http_json("GET", "/users")
    existing = next((u for u in users if u.get("name") == profile["name"]), None)
    if existing:
        user_id = int(existing.get("user_id") or existing.get("id"))
        http_json("PUT", f"/users/{user_id}", profile)
        print(f"Updated {profile['name']} (user_id={user_id})")
    else:
        created = http_json("POST", "/users", profile)
        user = created.get("user") or created
        user_id = int(user.get("user_id") or user.get("id"))
        print(f"Created {profile['name']} (user_id={user_id})")
    metrics = http_json("GET", f"/users/{user_id}/metrics")
    return user_id, metrics


MAX_CHAT_ATTEMPTS = 2


def chat(user_id: int, message: str, force_new_session: bool = False) -> tuple[dict, float]:
    payload = {
        "user_id": user_id,
        "message": message,
        "force_new_session": force_new_session,
    }
    start = time.perf_counter()
    result = http_json("POST", "/chat", payload, timeout=300)
    return result, time.perf_counter() - start


def chat_with_retries(
    user_id: int,
    message: str,
    *,
    force_new_session: bool = False,
    max_attempts: int = MAX_CHAT_ATTEMPTS,
) -> dict:
    """Call /chat with retries for empty replies or transient 503s."""
    attempts: list[dict] = []
    last_error: str | None = None
    total_elapsed = 0.0

    for attempt in range(1, max_attempts + 1):
        force_new = force_new_session if attempt == 1 else False
        try:
            resp, elapsed = chat(user_id, message, force_new_session=force_new)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            elapsed = 0.0
            attempts.append(
                {
                    "attempt": attempt,
                    "elapsed_s": None,
                    "http_status": e.code,
                    "error": f"HTTP {e.code}: {body[:500]}",
                }
            )
            last_error = f"HTTP {e.code}: {body}"
            # Retry only transient / empty-reply style failures from backend.
            if e.code in (502, 503) and attempt < max_attempts:
                print(f"  retryable HTTP {e.code} on attempt {attempt}/{max_attempts}")
                continue
            break
        except Exception as e:  # noqa: BLE001
            attempts.append(
                {
                    "attempt": attempt,
                    "elapsed_s": None,
                    "error": str(e),
                }
            )
            last_error = str(e)
            break

        total_elapsed += elapsed
        reply = (resp.get("reply") or "").strip()
        attempt_rec = {
            "attempt": attempt,
            "elapsed_s": round(elapsed, 2),
            "sources": resp.get("sources"),
            "safety_blocked": resp.get("safety_blocked"),
            "reply_chars": len(reply),
        }
        if not reply:
            attempt_rec["error"] = "empty_reply"
            attempts.append(attempt_rec)
            last_error = "empty_reply"
            print(f"  empty_reply on attempt {attempt}/{max_attempts}")
            if attempt < max_attempts:
                continue
            break

        attempts.append(attempt_rec)
        return {
            "ok": True,
            "attempts": attempts,
            "attempt_count": attempt,
            "elapsed_s": round(total_elapsed, 2),
            "response": resp,
            "reply": resp.get("reply") or "",
            "error": None,
        }

    return {
        "ok": False,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "elapsed_s": round(total_elapsed, 2) if total_elapsed else None,
        "response": None,
        "reply": "",
        "error": last_error or "unknown_error",
    }


def run_persona(persona: dict) -> dict:
    profile = persona["profile"]
    user_id, metrics = upsert_persona(profile)
    print(
        f"  metrics: weight={metrics.get('weight_kg')} "
        f"BMI={metrics.get('bmi')} REE={metrics.get('ree')}"
    )

    turns = []
    for i, q in enumerate(persona["questions"]):
        print(f"\n[{q['id']}] {q['text']}")
        outcome = chat_with_retries(
            user_id,
            q["text"],
            force_new_session=(i == 0),
        )
        if not outcome["ok"]:
            print(f"  FAIL: {outcome['error']} attempts={outcome['attempt_count']}")
            turns.append(
                {
                    "id": q["id"],
                    "question": q["text"],
                    "error": outcome["error"],
                    "attempts": outcome["attempts"],
                    "attempt_count": outcome["attempt_count"],
                    "elapsed_s": outcome["elapsed_s"],
                    "reply": "",
                }
            )
            continue

        resp = outcome["response"] or {}
        reply = outcome["reply"]
        print(
            f"  time={outcome['elapsed_s']:.1f}s attempts={outcome['attempt_count']} "
            f"sources={resp.get('sources')} safety={resp.get('safety_blocked')} "
            f"chars={len(reply)}"
        )
        preview = reply[:240].replace("\n", " ")
        print(f"  reply preview: {preview}...")
        turns.append(
            {
                "id": q["id"],
                "question": q["text"],
                "elapsed_s": outcome["elapsed_s"],
                "attempts": outcome["attempts"],
                "attempt_count": outcome["attempt_count"],
                "sources": resp.get("sources"),
                "safety_blocked": resp.get("safety_blocked"),
                "memory_used": resp.get("memory_used"),
                "session_id": resp.get("session_id"),
                "reply": reply,
            }
        )

    return {
        "persona": persona["key"],
        "name": profile["name"],
        "user_id": user_id,
        "profile": profile,
        "metrics": {
            "weight_kg": metrics.get("weight_kg"),
            "bmi": metrics.get("bmi"),
            "ree": metrics.get("ree"),
            "bmi_label": metrics.get("bmi_label"),
        },
        "turns": turns,
    }


def to_markdown(results: list[dict], health: dict) -> str:
    lines = [
        f"# Persona Q&A Transcript — {STAMP}",
        "",
        "Live NutriCoachAI chat runs for Adem, Steven, and Emily.",
        "",
        f"- Model: `{health.get('ollama_model')}`",
        f"- Memory mode: `{health.get('memory_mode')}`",
        f"- RAG ready: `{health.get('rag_ready')}`",
        "",
    ]
    for block in results:
        lines.append(f"## {block['name']} (user_id={block['user_id']})")
        lines.append("")
        m = block.get("metrics") or {}
        lines.append(
            f"- Height: {block['profile']['height_cm']} cm | "
            f"Weight: {m.get('weight_kg')} kg | BMI: {m.get('bmi')} | "
            f"REE: {m.get('ree')} | Goal: {block['profile']['goal']}"
        )
        lines.append("")
        for t in block["turns"]:
            lines.append(f"### {t['id']}")
            lines.append("")
            lines.append(f"**Q:** {t['question']}")
            lines.append("")
            if t.get("error"):
                lines.append(f"**Error:** {t['error']}")
                if t.get("attempts"):
                    lines.append("")
                    lines.append(f"*attempts={t.get('attempt_count')} · detail={t.get('attempts')}*")
            else:
                lines.append(
                    f"*time={t.get('elapsed_s')}s · attempts={t.get('attempt_count', 1)} · "
                    f"sources={t.get('sources')} · "
                    f"safety_blocked={t.get('safety_blocked')}*"
                )
                lines.append("")
                lines.append("**A:**")
                lines.append("")
                lines.append(t.get("reply") or "")
            lines.append("")
            lines.append("---")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    health = wait_health()
    print(
        "Health:",
        json.dumps(
            {
                k: health[k]
                for k in (
                    "ollama_model",
                    "rag_ready",
                    "memory_mode",
                    "ollama_reachable",
                )
            },
            ensure_ascii=False,
        ),
    )

    results = []
    for persona in PERSONAS:
        print("\n" + "=" * 72)
        print(f"Persona: {persona['profile']['name']}")
        print("=" * 72)
        results.append(run_persona(persona))

    json_path = OUT_DIR / f"persona_qa_{STAMP}.json"
    md_path = OUT_DIR / f"persona_qa_{STAMP}.md"
    latest_json = OUT_DIR / "persona_qa_latest.json"
    latest_md = OUT_DIR / "persona_qa_latest.md"

    payload = {"timestamp": STAMP, "health": health, "results": results}
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    md = to_markdown(results, health)
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Saved JSON: {json_path}")
    print(f"Saved MD:   {md_path}")
    print(f"Latest:     {latest_md}")
    for block in results:
        ok = sum(1 for t in block["turns"] if not t.get("error") and (t.get("reply") or "").strip())
        print(f"  {block['name']}: {ok}/{len(block['turns'])} answers")


if __name__ == "__main__":
    main()

"""Seed Steven demo user from meeting agenda (profile + 3-month weight trend)."""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import logic  # noqa: E402

STEVEN_PROFILE = {
    "name": "Steven",
    "gender": "male",
    "birth_date": "1963-01-15",
    "height_cm": 190.0,
    "initial_weight_kg": 108.3,
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
        "Asian male, age 63, divorced and lives alone. Three adult children (ages 22, 23, 26). "
        "Office job working from home — sits at a computer all day with no regular exercise. "
        "Occasionally travels ~2 hours by train for collaborator meetings."
    ),
    "coach_notes": (
        "Primary coaching focus: T2DM glycemic management and lowering pain/inflammation. "
        "Sedentary WFH lifestyle; no commute. Limited mobility due to L4-L5 disc issues — "
        "favor low-impact, spine-safe activity suggestions. Monitor mood/depression sensitively."
    ),
}

WEIGHT_START_KG = 112.0
WEIGHT_END_KG = 108.3
WEIGHT_DAYS = 90


def seed_weight_history(user_id: int, start_kg: float, end_kg: float, days: int) -> None:
    for i in range(days):
        recorded_at = (datetime.now() - timedelta(days=(days - 1 - i))).strftime("%Y-%m-%d %H:%M:%S")
        progress = i / max(days - 1, 1)
        base_weight = start_kg + (end_kg - start_kg) * progress
        if i == days - 1:
            current_weight = round(end_kg, 1)
        else:
            current_weight = round(base_weight + random.uniform(-0.4, 0.4), 1)
        logic.upsert_weight_entry(user_id, current_weight, recorded_at=recorded_at)


def main() -> None:
    logic.init_db()
    existing = next((u for u in logic.get_all_users() if u.get("name") == "Steven"), None)
    if existing:
        user_id = int(existing["user_id"])
        logic.update_user_profile(
            user_id,
            STEVEN_PROFILE["name"],
            STEVEN_PROFILE["gender"],
            STEVEN_PROFILE["birth_date"],
            STEVEN_PROFILE["height_cm"],
            goal=STEVEN_PROFILE["goal"],
            activity_level=STEVEN_PROFILE["activity_level"],
            diet_preference=STEVEN_PROFILE["diet_preference"],
            budget_level=STEVEN_PROFILE["budget_level"],
            medical_conditions=STEVEN_PROFILE["medical_conditions"],
            allergies=STEVEN_PROFILE["allergies"],
            food_dislikes=STEVEN_PROFILE["food_dislikes"],
            target_weight=STEVEN_PROFILE["target_weight"],
            target_timeline=STEVEN_PROFILE["target_timeline"],
            self_description=STEVEN_PROFILE["self_description"],
            coach_notes=STEVEN_PROFILE["coach_notes"],
        )
        print(f"Updated existing Steven (user_id={user_id})")
    else:
        user_id = logic.create_user_profile(
            STEVEN_PROFILE["name"],
            STEVEN_PROFILE["gender"],
            STEVEN_PROFILE["birth_date"],
            STEVEN_PROFILE["height_cm"],
            STEVEN_PROFILE["initial_weight_kg"],
            goal=STEVEN_PROFILE["goal"],
            activity_level=STEVEN_PROFILE["activity_level"],
            diet_preference=STEVEN_PROFILE["diet_preference"],
            budget_level=STEVEN_PROFILE["budget_level"],
            medical_conditions=STEVEN_PROFILE["medical_conditions"],
            allergies=STEVEN_PROFILE["allergies"],
            food_dislikes=STEVEN_PROFILE["food_dislikes"],
            target_weight=STEVEN_PROFILE["target_weight"],
            target_timeline=STEVEN_PROFILE["target_timeline"],
            self_description=STEVEN_PROFILE["self_description"],
            coach_notes=STEVEN_PROFILE["coach_notes"],
        )
        logic.ensure_user_memory_state(user_id)
        print(f"Created Steven (user_id={user_id})")

    seed_weight_history(user_id, WEIGHT_START_KG, WEIGHT_END_KG, WEIGHT_DAYS)
    bundle = logic.get_latest_metrics_bundle(user_id)
    print(
        f"Seeded {WEIGHT_DAYS} weight entries: {WEIGHT_START_KG} kg → {WEIGHT_END_KG} kg "
        f"(current BMI {bundle.get('bmi')}, label {bundle.get('bmi_label')})"
    )


if __name__ == "__main__":
    main()

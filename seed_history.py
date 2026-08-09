"""Seed weight history for existing users via logic.py (authoritative schema)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import logic


def seed_data(username: str, start_weight: float, end_weight: float, days: int = 30) -> None:
    logic.init_db()
    user = next((item for item in logic.get_all_users() if item.get("name") == username), None)
    if not user:
        print(f"User not found: {username}")
        return

    user_id = int(user["user_id"])
    print(f"Seeding {days} days of weight data for {username} (user_id={user_id})...")

    for i in range(days):
        recorded_at = (datetime.now() - timedelta(days=(days - 1 - i))).strftime("%Y-%m-%d %H:%M:%S")
        progress = i / max(days - 1, 1)
        base_weight = start_weight + (end_weight - start_weight) * progress
        current_weight = round(base_weight + random.uniform(-0.3, 0.3), 1)
        logic.upsert_weight_entry(user_id, current_weight, recorded_at=recorded_at)

    print(f"Done seeding weight history for {username}.")


if __name__ == "__main__":
    seed_data("Steven", 112.0, 108.3, 90)
    seed_data("David", 65.0, 69.5, 30)
    seed_data("Amy", 62.0, 61.2, 30)

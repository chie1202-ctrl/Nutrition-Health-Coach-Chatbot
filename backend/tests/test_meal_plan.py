import json

import pytest

import logic


def test_extract_json_object_uses_first_valid_block():
    first = {
        "summary": "first",
        "days": [{"day": 1, "breakfast": "A", "lunch": "B", "dinner": "C", "snack": "D", "focus": "f", "notes": ""}],
    }
    second = {
        "summary": "second",
        "days": [{"day": 1, "breakfast": "A", "lunch": "B", "dinner": "C", "snack": "D", "focus": "f", "notes": ""}],
    }
    raw = json.dumps(first) + "\n```json\n" + json.dumps(second)
    parsed = json.loads(logic._extract_json_object(raw))
    assert parsed["summary"] == "first"


def test_contains_allergen_ignores_negated_mentions():
    assert logic._contains_allergen("lean beef stir-fry (no shellfish)", "shellfish") is False
    assert logic._contains_allergen("grilled shrimp salad", "shellfish") is True


def test_offline_meal_plan_has_seven_distinct_days(isolated_db):
    user_id = logic.create_user_profile(
        name="Meal User",
        gender="female",
        birth_date="1990-01-01",
        height_cm=165,
        initial_weight_kg=60,
        allergies=["shellfish"],
    )
    user = logic.get_user_profile(user_id)
    plan = logic.build_offline_meal_plan(user)
    validation = logic.validate_meal_plan(plan, user)

    assert validation["day_count"] == 7
    assert validation["distinct_main_meals"] == 7
    assert validation["valid"] is True

    blob = " ".join(
        f"{day['breakfast']} {day['lunch']} {day['dinner']} {day['snack']}".lower()
        for day in plan["days"]
    )
    assert "shellfish" not in blob
    assert "shrimp" not in blob

    logic.delete_user(user_id)


def test_validate_meal_plan_rejects_vague_and_leftover_meals(isolated_db):
    user_id = logic.create_user_profile(
        name="Quality User",
        gender="male",
        birth_date="19620101",
        height_cm=175,
        medical_conditions=["type 2 diabetes"],
        diet_preference="high_protein",
    )
    user = logic.get_user_profile(user_id)
    plan = {
        "summary": "test",
        "nutrition_targets": {"calories": 1800, "protein_g": 140, "carbs_g": 160, "fat_g": 55},
        "days": [
            {
                "day": 1,
                "breakfast": "Balanced breakfast",
                "lunch": "Leftover salmon from yesterday",
                "dinner": "Chicken breast 120 g, broccoli 150 g, brown rice 80 g",
                "snack": "Greek yogurt 150 g, blueberries 80 g",
                "focus": "test",
                "notes": "",
                "daily_totals": {"calories": 1800, "protein_g": 140, "carbs_g": 160, "fat_g": 55},
            }
        ],
    }
    validation = logic.validate_meal_plan(plan, user)
    assert validation["valid"] is False
    joined = " ".join(validation["issues"]).lower()
    assert "vague" in joined
    assert "leftover" in joined
    logic.delete_user(user_id)


def test_build_meal_plan_condition_lines_include_diabetes_rules():
    user = {"birth_date": "19620101", "medical_conditions": ["type 2 diabetes"], "diet_preference": "high_protein"}
    lines = logic.build_meal_plan_condition_requirement_lines(user)
    assert "Diabetes profile" in lines
    assert "gram portions" in lines


def test_normalize_meal_plan_preserves_nutrition_fields(isolated_db):
    user_id = logic.create_user_profile(
        name="Nutrition User",
        gender="female",
        birth_date="1990-01-01",
        height_cm=165,
    )
    user = logic.get_user_profile(user_id)
    plan = logic.normalize_meal_plan_days(
        {
            "summary": "structured",
            "nutrition_targets": {"calories": 1800, "protein_g": 140, "carbs_g": 160, "fat_g": 55},
            "days": [
                {
                    "day": 1,
                    "breakfast": "Eggs 120 g, whole-grain toast 60 g",
                    "lunch": "Salmon 130 g, quinoa 90 g, kale 100 g",
                    "dinner": "Chicken breast 120 g, broccoli 150 g, olive oil 10 g",
                    "snack": "Greek yogurt 150 g, berries 80 g",
                    "focus": "High protein",
                    "notes": "Swap salmon for cod if preferred.",
                    "daily_totals": {"calories": 1820, "protein_g": 142, "carbs_g": 155, "fat_g": 58},
                }
            ],
        },
        user,
    )
    assert plan["nutrition_targets"]["protein_g"] == 140
    assert plan["days"][0]["daily_totals"]["calories"] == 1820
    logic.delete_user(user_id)


def test_normalize_meal_plan_pads_to_seven_days(isolated_db):
    user_id = logic.create_user_profile(
        name="Pad User",
        gender="male",
        birth_date="1988-01-01",
        height_cm=175,
    )
    user = logic.get_user_profile(user_id)
    plan = logic.normalize_meal_plan_days(
        {
            "summary": "short",
            "days": [
                {
                    "day": 1,
                    "breakfast": "A",
                    "lunch": "B",
                    "dinner": "C",
                    "snack": "D",
                    "focus": "f",
                    "notes": "",
                }
            ],
        },
        user,
    )
    assert len(plan["days"]) == 7
    logic.delete_user(user_id)


def test_generate_meal_plan_requires_ollama(isolated_db, monkeypatch):
    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: False)

    user_id = logic.create_user_profile(
        name="No Ollama",
        gender="female",
        birth_date="1990-01-01",
        height_cm=160,
    )
    user = logic.get_user_profile(user_id)

    with pytest.raises(logic.OllamaUnavailableError):
        logic.generate_meal_plan(user, None, rag_store=None)

    logic.delete_user(user_id)

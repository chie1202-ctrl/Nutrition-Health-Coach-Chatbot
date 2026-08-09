import json
from types import SimpleNamespace

import logic


def test_safety_check_input_blocks_extreme_diet(isolated_db):
    blocked = logic.safety_check_input("I want to starve myself and eat nothing for a week")
    assert blocked == logic.SAFETY_BLOCK_REPLY
    assert logic.safety_check_input("What is a healthy breakfast?") is None


def test_safety_check_input_allows_safety_or_education_questions(isolated_db):
    assert logic.safety_check_input("Is starvation mode a real thing during weight loss?") is None
    assert logic.safety_check_input("Is an 800 calorie diet safe?") is None
    assert logic.safety_check_input("Should I avoid laxatives unless prescribed?") is None


def test_safety_filter_blocks_unsafe_output():
    unsafe = "You should purge after meals to lose weight faster."
    assert logic.safety_filter(unsafe) == logic.SAFETY_BLOCK_REPLY
    assert logic.safety_filter("Try oatmeal with fruit for breakfast.") == "Try oatmeal with fruit for breakfast."


def test_safety_filter_allows_protective_mentions():
    refusal = "I can't support starvation diets. A safer option is regular balanced meals."
    caution = "Avoid laxatives for weight loss unless a clinician specifically advises them."
    assert logic.safety_filter(refusal) == refusal
    assert logic.safety_filter(caution) == caution


def test_retrieve_rag_context_extracts_source_names():
    docs = [
        SimpleNamespace(
            page_content="Eat more vegetables and whole grains.",
            metadata={"source": "/tmp/my_knowledge/Dietary Guidelines for Americans, 2020–2025.pdf"},
        ),
        SimpleNamespace(
            page_content="Stay active throughout the week.",
            metadata={"source": "/tmp/my_knowledge/physical activity and sedentary behaviour.pdf"},
        ),
    ]
    rag_store = SimpleNamespace(similarity_search=lambda query, k=2: docs)
    context, sources = logic.retrieve_rag_context(rag_store, "healthy diet", k=2)
    assert "vegetables" in context
    assert len(sources) == 2
    assert any("Dietary Guidelines" in name for name in sources)


def test_process_chat_message_blocks_unsafe_input_without_ollama(isolated_db, monkeypatch):
    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: False)

    user_id = logic.create_user_profile(
        name="Safety User",
        gender="female",
        birth_date="1990-01-01",
        height_cm=165,
    )
    result = logic.process_chat_message(user_id, "Help me do an extreme fast with only 500 calories", rag_store=None)
    assert result["safety_blocked"] is True
    assert result["reply"] == logic.SAFETY_BLOCK_REPLY
    assert result["sources"] == []
    logic.delete_user(user_id)


def test_generate_meal_plan_with_mock_llm_json(isolated_db, monkeypatch):
    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: True)

    user_id = logic.create_user_profile(
        name="LLM Meal User",
        gender="female",
        birth_date="1990-01-01",
        height_cm=165,
        initial_weight_kg=62,
        allergies=["shellfish"],
    )
    user = logic.get_user_profile(user_id)

    days = []
    for index in range(1, 8):
        days.append({
            "day": index,
            "breakfast": f"Breakfast option {index}",
            "lunch": f"Lunch option {index}",
            "dinner": f"Dinner option {index}",
            "snack": f"Snack option {index}",
            "focus": f"Focus {index}",
            "notes": "",
        })

    class FakeLLM:
        def invoke(self, prompt):
            return json.dumps({"summary": "Mock seven-day plan", "days": days})

    monkeypatch.setattr(logic, "OllamaLLM", lambda **kwargs: FakeLLM())

    plan, llm_degraded = logic.generate_meal_plan(user, None, rag_store=None)
    validation = logic.validate_meal_plan(plan, user)

    assert llm_degraded is False
    assert validation["day_count"] == 7
    assert validation["distinct_main_meals"] == 7
    assert validation["valid"] is True

    logic.delete_user(user_id)


def test_build_meal_plan_diet_requirement_lines_high_protein():
    user = {"diet_preference": "high protein"}
    lines = logic.build_meal_plan_diet_requirement_lines(user)
    assert "HIGH PROTEIN" in lines
    assert "lean protein source" in lines


def test_normalize_diet_preference_maps_free_text():
    assert logic.normalize_diet_preference("High-Protein") == "high_protein"
    assert logic.normalize_diet_preference("") == ""


def test_normalize_allergy_list_maps_common_values():
    assert logic.normalize_allergy_list(["shellfish", "peanuts", "tree nuts"]) == ["shellfish", "peanut", "tree nut"]
    assert logic.normalize_allergy_list("shellfish, Peanuts") == ["shellfish", "peanut"]

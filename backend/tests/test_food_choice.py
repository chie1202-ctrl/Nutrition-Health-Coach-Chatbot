import json

import pytest

import logic


INTENT_POSITIVE = [
    ("Should I get pizza or Chinese veg meal tonight?", True),
    ("Pizza vs burger for takeaway", True),
    ("Which is better for dining out: sushi or poke bowl?", True),
    ("takeaway: pad thai or salad bowl?", True),
    ("pizza or burger tonight", True),
    ("I'm eating out — sushi or poke bowl", True),
    ("decide between burger and wrap for lunch", True),
]

INTENT_NEGATIVE = [
    ("What is a healthy breakfast?", False),
    ("Remind me what we discussed last time", False),
    ("What was my weight goal?", False),
    ("Generate my meal plan for the week", False),
    ("", False),
    ("or", False),
    ("I want to lose 5 kg", False),
]


@pytest.mark.parametrize("message,expected", INTENT_POSITIVE + INTENT_NEGATIVE)
def test_detect_food_choice_intent_matrix(message, expected):
    assert logic.detect_food_choice_intent(message) is expected


def test_count_food_choice_dims_filled():
    comparison = {
        "comparison": {
            "protein": {"option_a": "high", "option_b": "moderate"},
            "carbs": {"option_a": "high", "option_b": "low"},
            "sodium": {"option_a": "", "option_b": "low"},
            "glycemic": {"option_a": "high", "option_b": "moderate"},
        }
    }
    assert logic.count_food_choice_dims_filled(comparison) == 3


def test_validate_food_choice_complete_profile(isolated_db):
    user_id = logic.create_user_profile(
        name="Validate User",
        gender="female",
        birth_date="19620101",
        height_cm=165,
        allergies=["shellfish"],
        medical_conditions=["type 2 diabetes"],
    )
    user = logic.get_user_profile(user_id)
    comparison = {
        "option_a": "Pizza",
        "option_b": "Vegetable stir-fry",
        "comparison": {
            "protein": {"option_a": "moderate", "option_b": "higher"},
            "carbs": {"option_a": "high", "option_b": "moderate"},
            "sodium": {"option_a": "high", "option_b": "moderate"},
            "glycemic": {"option_a": "sharp rise", "option_b": "steadier"},
        },
        "recommendation": "Choose the stir-fry for better blood-sugar balance.",
        "portion_tip": "Ask for half rice.",
        "profile_notes": ["Check ingredients for allergens: shellfish."],
    }
    result = logic.validate_food_choice(comparison, user)
    assert result["valid"] is True
    assert result["scores"]["dims_filled"] == 4
    assert result["scores"]["allergy_safe"] is True
    logic.delete_user(user_id)


def test_validate_food_choice_flags_unsafe_sushi_recommendation(isolated_db):
    user_id = logic.create_user_profile(
        name="Unsafe User",
        gender="male",
        birth_date="19900101",
        height_cm=178,
        allergies=["shellfish"],
    )
    user = logic.get_user_profile(user_id)
    comparison = {
        "option_a": "Burger",
        "option_b": "Sushi Bowl",
        "comparison": {
            "protein": {"option_a": "moderate", "option_b": "higher"},
            "carbs": {"option_a": "high", "option_b": "moderate"},
            "sodium": {"option_a": "high", "option_b": "moderate"},
            "glycemic": {"option_a": "high", "option_b": "lower"},
        },
        "recommendation": "The sushi bowl is generally better for weight loss.",
        "portion_tip": "Choose one serving.",
        "profile_notes": ["Check ingredients for allergens: shellfish."],
    }
    result = logic.validate_food_choice(comparison, user)
    assert result["scores"]["allergy_safe"] is False
    assert "allergy_unsafe_recommendation" in result["issues"]
    logic.delete_user(user_id)


def test_normalize_food_choice_adds_allergy_and_diabetes_notes(isolated_db):
    user_id = logic.create_user_profile(
        name="Choice User",
        gender="female",
        birth_date="19620101",
        height_cm=165,
        allergies=["peanut"],
        medical_conditions=["type 2 diabetes"],
        diet_preference="high_protein",
    )
    user = logic.get_user_profile(user_id)
    normalized = logic.normalize_food_choice(
        {
            "option_a": "Pepperoni pizza",
            "option_b": "Tofu vegetable stir-fry",
            "comparison": {
                "protein": {"option_a": "moderate", "option_b": "higher lean protein"},
                "carbs": {"option_a": "high refined crust", "option_b": "moderate with rice"},
                "sodium": {"option_a": "high from cheese and cured meat", "option_b": "depends on sauce"},
                "glycemic": {"option_a": "sharp rise from white crust", "option_b": "steadier with veg and protein"},
            },
            "recommendation": "Choose the stir-fry for better blood-sugar balance.",
            "portion_tip": "Ask for half rice and extra vegetables.",
            "swap_suggestion": "If you pick pizza, limit to two slices and add a side salad.",
            "profile_notes": [],
        },
        user,
    )
    joined_notes = " ".join(normalized["profile_notes"]).lower()
    assert "peanut" in joined_notes
    assert "diabet" in joined_notes or "glycemic" in joined_notes
    assert normalized["comparison"]["protein"]["option_b"] == "higher lean protein"
    logic.delete_user(user_id)


def test_extract_food_choice_round_trip():
    payload = {
        "option_a": "Pizza",
        "option_b": "Salad",
        "comparison": {
            "protein": {"option_a": "a", "option_b": "b"},
            "carbs": {"option_a": "a", "option_b": "b"},
            "sodium": {"option_a": "a", "option_b": "b"},
            "glycemic": {"option_a": "a", "option_b": "b"},
        },
        "recommendation": "Salad fits better.",
        "portion_tip": "Large portion of greens.",
        "swap_suggestion": "",
        "profile_notes": [],
    }
    reply = logic.format_food_choice_reply(payload)
    stored = logic.embed_food_choice_payload(reply, payload)
    visible, embedded = logic.extract_food_choice_from_content(stored)
    assert logic.FOOD_CHOICE_MARKER_START not in visible
    assert embedded["option_a"] == "Pizza"
    assert embedded["option_b"] == "Salad"


def test_extract_compared_options_from_message():
    a, b = logic.extract_compared_options_from_message(
        "I am eating out tonight — help me compare takeaway options. beef noodles vs curry rice. Which is better for my profile?"
    )
    assert "beef" in a.lower()
    assert "curry" in b.lower()


def test_food_choice_is_usable_rejects_empty_shell():
    empty = {
        "option_a": "Option A",
        "option_b": "Option B",
        "comparison": {
            "protein": {"option_a": "", "option_b": ""},
            "carbs": {"option_a": "", "option_b": ""},
            "sodium": {"option_a": "", "option_b": ""},
            "glycemic": {"option_a": "", "option_b": ""},
        },
        "recommendation": "",
    }
    assert logic.food_choice_is_usable(empty) is False
    assert logic.food_choice_is_usable(None) is False


def test_food_choice_is_usable_requires_all_four_dimensions():
    partial = {
        "option_a": "curry rice",
        "option_b": "beef noodles",
        "comparison": {
            "protein": {"option_a": "moderate", "option_b": "higher"},
            "carbs": {"option_a": "higher", "option_b": "moderate"},
            "sodium": {"option_a": "", "option_b": ""},
            "glycemic": {"option_a": "", "option_b": ""},
        },
        "recommendation": "Beef noodles may fit weight-loss goals better.",
    }
    assert logic.food_choice_is_usable(partial) is False
    partial["comparison"]["sodium"] = {"option_a": "sauce can be high", "option_b": "broth can be high"}
    partial["comparison"]["glycemic"] = {"option_a": "rice raises glucose", "option_b": "noodles moderate"}
    assert logic.food_choice_is_usable(partial) is True


def test_normalize_food_choice_seeds_labels_from_message(isolated_db):
    user_id = logic.create_user_profile(
        name="Seed User",
        gender="female",
        birth_date="19950101",
        height_cm=165,
    )
    user = logic.get_user_profile(user_id)
    normalized = logic.normalize_food_choice(
        {
            "comparison": {
                "protein": {"option_a": "high", "option_b": "moderate"},
                "carbs": {"option_a": "moderate", "option_b": "high"},
                "sodium": {"option_a": "broth can be salty", "option_b": "sauce can be salty"},
                "glycemic": {"option_a": "noodles moderate-high", "option_b": "rice higher"},
            },
            "recommendation": "Beef noodles fit better.",
        },
        user,
        message="compare takeaway options. beef noodles vs curry rice. Which is better?",
    )
    assert "beef" in normalized["option_a"].lower()
    assert "curry" in normalized["option_b"].lower()
    assert logic.food_choice_is_usable(normalized) is True
    logic.delete_user(user_id)


def test_run_food_choice_retries_after_empty_json(isolated_db, monkeypatch):
    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: True)
    user_id = logic.create_user_profile(
        name="Retry User",
        gender="male",
        birth_date="19900101",
        height_cm=175,
    )
    user = logic.get_user_profile(user_id)
    good = {
        "option_a": "Beef noodles",
        "option_b": "Curry rice",
        "comparison": {
            "protein": {"option_a": "higher from beef", "option_b": "moderate"},
            "carbs": {"option_a": "moderate", "option_b": "higher from rice"},
            "sodium": {"option_a": "broth can be high", "option_b": "sauce can be high"},
            "glycemic": {"option_a": "moderate", "option_b": "higher with white rice"},
        },
        "recommendation": "Beef noodles usually fit a weight-loss profile better.",
        "portion_tip": "Ask for less oil and finish half if the bowl is large.",
        "swap_suggestion": "For curry rice, request brown rice or extra vegetables.",
        "profile_notes": [],
    }
    calls = {"n": 0}

    class FakeLLM:
        def invoke(self, prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return "sorry I cannot format that"
            return json.dumps(good)

    monkeypatch.setattr(logic, "OllamaLLM", lambda **kwargs: FakeLLM())
    turn = {
        "user_id": user_id,
        "message": "beef noodles vs curry rice which is better for takeaway?",
        "memory_context": {"memory_text": "", "memory_used": {"mode": "M2"}},
        "sources": [],
    }
    reply, comparison = logic.run_food_choice_comparison(turn, user, None, rag_store=None)
    assert calls["n"] >= 2
    assert comparison is not None
    assert comparison["option_a"] == "Beef noodles"
    assert "Beef noodles" in reply
    logic.delete_user(user_id)


def test_process_chat_message_food_choice_with_mock_llm(isolated_db, monkeypatch):
    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: True)

    user_id = logic.create_user_profile(
        name="Takeaway User",
        gender="male",
        birth_date="1990-01-01",
        height_cm=178,
        allergies=["shellfish"],
    )
    payload = {
        "option_a": "Two slices cheese pizza",
        "option_b": "Chicken and broccoli with brown rice",
        "comparison": {
            "protein": {"option_a": "moderate", "option_b": "higher"},
            "carbs": {"option_a": "high", "option_b": "moderate"},
            "sodium": {"option_a": "high", "option_b": "moderate"},
            "glycemic": {"option_a": "high GI crust", "option_b": "lower GI with fibre"},
        },
        "recommendation": "Brown rice bowl fits your goal better.",
        "portion_tip": "Choose steamed broccoli and sauce on the side.",
        "swap_suggestion": "If pizza, stop at two slices and add salad.",
        "profile_notes": ["No shellfish-based toppings or sauces."],
    }

    class FakeLLM:
        def invoke(self, prompt):
            return json.dumps(payload)

    monkeypatch.setattr(logic, "OllamaLLM", lambda **kwargs: FakeLLM())

    result = logic.process_chat_message(
        user_id,
        "I'm eating out — pizza or Chinese veg meal, which is better?",
        rag_store=None,
    )

    assert result.get("food_choice")
    assert result["food_choice"]["option_a"] == "Two slices cheese pizza"
    assert "shellfish" in result["reply"].lower() or any(
        "shellfish" in note.lower() for note in result["food_choice"].get("profile_notes", [])
    )
    assert logic.FOOD_CHOICE_MARKER_START in logic.get_chat_history(user_id)["messages"][-1]["content"]

    visible, embedded = logic.extract_food_choice_from_content(result["reply"])
    assert embedded is None
    stored_visible, stored_embedded = logic.extract_food_choice_from_content(
        logic.get_chat_history(user_id)["messages"][-1]["content"]
    )
    assert stored_embedded["option_b"] == "Chicken and broccoli with brown rice"
    assert logic.FOOD_CHOICE_MARKER_START not in stored_visible

    logic.delete_user(user_id)

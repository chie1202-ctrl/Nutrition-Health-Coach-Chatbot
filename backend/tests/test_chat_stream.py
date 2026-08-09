def test_chat_stream_safety_block(api_client):
    user_id = api_client.post(
        "/users",
        json={
            "name": "Stream User",
            "gender": "female",
            "birth_date": "1990-01-01",
            "height_cm": 165,
        },
    ).json()["user"]["user_id"]

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={"user_id": user_id, "message": "I want to starve and only eat 500 calories"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: done" in body
    assert "safety_blocked" in body
    assert "event: token" in body

    api_client.delete(f"/users/{user_id}")


def test_chat_stream_mock_tokens(api_client, monkeypatch):
    import logic
    from langchain_core.outputs import GenerationChunk

    class FakeStreamLLM:
        def stream(self, prompt):
            for piece in ("Hello ", "coach"):
                yield GenerationChunk(text=piece)

    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: True)
    monkeypatch.setattr(logic, "create_ollama_llm", lambda **kwargs: FakeStreamLLM())

    user_id = api_client.post(
        "/users",
        json={
            "name": "Stream Mock",
            "gender": "male",
            "birth_date": "1988-01-01",
            "height_cm": 175,
        },
    ).json()["user"]["user_id"]

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={"user_id": user_id, "message": "Hello"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert '"text": "Hello "' in body or '"text":"Hello "' in body
    assert "Hello coach" in body
    assert "event: done" in body

    api_client.delete(f"/users/{user_id}")


def test_chat_stream_empty_reply_errors_without_assistant_save(api_client, monkeypatch):
    import logic

    class FakeStreamLLM:
        def stream(self, prompt):
            yield "<" + "think" + ">hidden only</" + "think" + ">"

        def invoke(self, prompt):
            return "<" + "think" + ">still empty</" + "think" + ">"

    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: True)
    monkeypatch.setattr(logic, "create_ollama_llm", lambda **kwargs: FakeStreamLLM())

    user_id = api_client.post(
        "/users",
        json={
            "name": "Stream Empty",
            "gender": "male",
            "birth_date": "1988-01-01",
            "height_cm": 175,
        },
    ).json()["user"]["user_id"]

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={"user_id": user_id, "message": "Hello"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: error" in body
    assert "event: done" not in body

    history = logic.get_chat_history(user_id)["messages"]
    assert any(m["role"] == "user" for m in history)
    assert all(m["role"] != "assistant" for m in history)

    api_client.delete(f"/users/{user_id}")


def test_chat_stream_ollama_unavailable(api_client, monkeypatch):
    import logic

    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: False)

    user_id = api_client.post(
        "/users",
        json={
            "name": "Stream Down",
            "gender": "male",
            "birth_date": "1988-01-01",
            "height_cm": 175,
        },
    ).json()["user"]["user_id"]

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={"user_id": user_id, "message": "Hello"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: error" in body
    api_client.delete(f"/users/{user_id}")


def test_chat_stream_calorie_target_without_ollama(api_client, monkeypatch):
    import logic

    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: False)

    user_id = api_client.post(
        "/users",
        json={
            "name": "Stream Calorie",
            "gender": "female",
            "birth_date": "1991-01-01",
            "height_cm": 165,
            "weight_kg": 62,
            "goal": "lose weight",
            "activity_level": "moderately active work",
        },
    ).json()["user"]["user_id"]

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={
            "user_id": user_id,
            "message": "Based on my height, weight and work pattern, how many calories should I aim to eat per day?",
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: token" in body
    assert "event: done" in body
    assert "calorie_target" in body
    assert "estimated maintenance" in body
    assert "1000" not in body

    api_client.delete(f"/users/{user_id}")


def test_chat_stream_food_choice_mock(api_client, monkeypatch):
    import logic

    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: True)

    payload = {
        "option_a": "Pizza",
        "option_b": "Stir-fry",
        "comparison": {
            "protein": {"option_a": "moderate", "option_b": "higher"},
            "carbs": {"option_a": "high", "option_b": "moderate"},
            "sodium": {"option_a": "high", "option_b": "moderate"},
            "glycemic": {"option_a": "high", "option_b": "lower"},
        },
        "recommendation": "Stir-fry fits better.",
        "portion_tip": "Half rice portion.",
        "swap_suggestion": "",
        "profile_notes": [],
    }

    def fake_run(turn_context, user, latest, rag_store=None):
        reply = logic.embed_food_choice_payload(logic.format_food_choice_reply(payload), payload)
        return reply, payload

    monkeypatch.setattr(logic, "run_food_choice_comparison", fake_run)

    user_id = api_client.post(
        "/users",
        json={
            "name": "Stream Food Choice",
            "gender": "female",
            "birth_date": "1990-01-01",
            "height_cm": 165,
        },
    ).json()["user"]["user_id"]

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={"user_id": user_id, "message": "pizza or stir-fry for takeaway tonight?"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert '"food_choice":true' in body.replace(" ", "") or '"food_choice": true' in body
    assert "event: done" in body
    assert '"option_a":"Pizza"' in body.replace(" ", "") or '"option_a": "Pizza"' in body

    api_client.delete(f"/users/{user_id}")

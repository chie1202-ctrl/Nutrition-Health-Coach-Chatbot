def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "ollama_reachable" in payload
    assert payload["memory_feature_enabled"] is True


def test_user_crud_and_weight(api_client):
    create_response = api_client.post(
        "/users",
        json={
            "name": "API User",
            "gender": "female",
            "birth_date": "1991-11-20",
            "height_cm": 168,
            "weight_kg": 60,
            "goal": "maintain_weight",
        },
    )
    assert create_response.status_code == 200
    user_id = create_response.json()["user"]["user_id"]

    weight_response = api_client.post(
        f"/users/{user_id}/weight",
        json={"user_id": user_id, "weight_kg": 59.5},
    )
    assert weight_response.status_code == 200
    weight_payload = weight_response.json()
    assert weight_payload["weight_kg"] == 59.5
    assert weight_payload["ree"] > 0

    bundle = api_client.get(f"/users/{user_id}").json()
    assert bundle["metrics"]["weight_kg"] == 59.5
    assert bundle["metrics"]["ree"] == weight_payload["ree"]

    delete_response = api_client.delete(f"/users/{user_id}")
    assert delete_response.status_code == 200


def test_chat_and_session_close(api_client, monkeypatch):
    import logic

    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: False)

    user_id = api_client.post(
        "/users",
        json={
            "name": "Chat User",
            "gender": "male",
            "birth_date": "1988-05-05",
            "height_cm": 175,
        },
    ).json()["user"]["user_id"]

    chat_response = api_client.post(
        "/chat",
        json={"user_id": user_id, "message": "Hello coach"},
    )
    assert chat_response.status_code == 503

    meal_response = api_client.post(f"/users/{user_id}/meal-plan")
    assert meal_response.status_code == 503

    close_response = api_client.post(f"/users/{user_id}/sessions/close")
    assert close_response.status_code == 200

    memory_response = api_client.get(f"/users/{user_id}/memory")
    assert memory_response.status_code == 200

    api_client.delete(f"/users/{user_id}")

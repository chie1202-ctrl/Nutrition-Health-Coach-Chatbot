"""Manual smoke test aligned with current logic.py APIs.

Prefer the automated suite:
    cd backend && python -m pytest tests -q
"""

from logic import (
    LLM_FALLBACK_REPLY,
    create_user_profile,
    delete_user,
    get_chat_history,
    get_latest_metrics_bundle,
    get_weight_history,
    init_db,
    process_chat_message,
    save_chat,
    upsert_weight_entry,
)


def run_smoke_test() -> None:
    print("Testing database init...")
    init_db()

    print("Creating user...")
    user_id = create_user_profile(
        name="Smoke Test User",
        gender="male",
        birth_date="1995-01-01",
        height_cm=175,
        initial_weight_kg=80,
    )
    print("User created:", user_id)

    print("Adding weight...")
    result = upsert_weight_entry(user_id, 79.5)
    print("Weight upserted:", result)

    latest = get_latest_metrics_bundle(user_id)
    print("Latest metrics:", latest)

    print("Getting history...")
    history = get_weight_history(user_id)
    print("History rows:", len(history))

    print("Saving chat...")
    session_id, _ = __import__("logic").resolve_session(user_id)
    save_chat(user_id, "user", "Hello coach", session_id=session_id)
    save_chat(user_id, "assistant", "Hi!", session_id=session_id)
    chat = get_chat_history(user_id)
    print("Chat messages:", len(chat["messages"]))

    print("Processing chat round-trip...")
    response = process_chat_message(user_id, "What should I eat for lunch?", rag_store=None)
    print("Chat reply degraded:", response.get("llm_degraded"))
    if response.get("llm_degraded"):
        assert response["reply"] == LLM_FALLBACK_REPLY

    print("Deleting user...")
    delete_user(user_id)
    print("Smoke test finished.")


if __name__ == "__main__":
    run_smoke_test()

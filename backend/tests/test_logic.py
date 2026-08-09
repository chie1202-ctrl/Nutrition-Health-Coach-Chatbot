import logic
import pytest


def test_strip_think_tags_removes_closed_block():
    assert logic.strip_think_tags("Answer <think>private reasoning</think> done") == "Answer  done"


def test_strip_think_tags_removes_unclosed_block_to_end():
    assert logic.strip_think_tags("Answer\n<think>private reasoning") == "Answer"


def test_think_tag_stream_filter_hides_split_thinking_block():
    stream_filter = logic.ThinkTagStreamFilter()

    visible = [
        stream_filter.feed("Start <thi"),
        stream_filter.feed("nk>hidden"),
        stream_filter.feed(" reasoning</thi"),
        stream_filter.feed("nk> answer"),
        stream_filter.finish(),
    ]

    assert "".join(visible) == "Start  answer"


def test_think_tag_stream_filter_drops_unclosed_thinking_block():
    stream_filter = logic.ThinkTagStreamFilter()

    visible = [
        stream_filter.feed("Answer first. <think>hidden"),
        stream_filter.feed(" reasoning that never closes"),
        stream_filter.finish(),
    ]

    assert "".join(visible) == "Answer first. "


def test_strip_think_tags_think_only_becomes_empty():
    open_tag = "<" + "think" + ">"
    close_tag = "</" + "think" + ">"
    assert logic.strip_think_tags(f"{open_tag}only reasoning{close_tag}") == ""


def test_invoke_llm_visible_reply_retries_then_succeeds(monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return "<" + "think" + ">hidden only</" + "think" + ">"
            assert "final answer directly" in prompt
            return "Visible coach answer"

    fake = FakeLLM()
    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: True)
    monkeypatch.setattr(logic, "create_ollama_llm", lambda **kwargs: fake)

    visible = logic.invoke_llm_visible_reply("User question")
    assert visible == "Visible coach answer"
    assert fake.calls == 2


def test_invoke_llm_visible_reply_raises_when_always_empty(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt):
            return "<" + "think" + ">still thinking</" + "think" + ">"

    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: True)
    monkeypatch.setattr(logic, "create_ollama_llm", lambda **kwargs: FakeLLM())

    with pytest.raises(logic.EmptyLLMReplyError):
        logic.invoke_llm_visible_reply("User question", max_attempts=2)


def test_process_chat_message_retries_empty_and_saves_visible(isolated_db, monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return "<" + "think" + ">private</" + "think" + ">"
            return "Eat more vegetables and keep a modest calorie deficit."

    fake = FakeLLM()
    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: True)
    monkeypatch.setattr(logic, "create_ollama_llm", lambda **kwargs: fake)

    user_id = logic.create_user_profile(
        name="Empty Retry User",
        gender="male",
        birth_date="1983-01-01",
        height_cm=180,
        initial_weight_kg=87.5,
    )
    result = logic.process_chat_message(user_id, "How can I lose weight safely?", rag_store=None)
    assert "vegetables" in result["reply"].lower() or "calorie" in result["reply"].lower()
    assert fake.calls == 2

    history = logic.get_chat_history(user_id)["messages"]
    assistant_msgs = [m for m in history if m.get("role") == "assistant"]
    assert assistant_msgs
    assert assistant_msgs[-1]["content"].strip()
    assert assistant_msgs[-1]["content"] == result["reply"]

    logic.delete_user(user_id)


def test_process_chat_message_does_not_save_empty_assistant(isolated_db, monkeypatch):
    class FakeLLM:
        def invoke(self, prompt):
            return "<" + "think" + ">no visible answer</" + "think" + ">"

    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: True)
    monkeypatch.setattr(logic, "create_ollama_llm", lambda **kwargs: FakeLLM())

    user_id = logic.create_user_profile(
        name="Empty Fail User",
        gender="female",
        birth_date="1989-01-01",
        height_cm=160,
        initial_weight_kg=66.6,
    )
    with pytest.raises(logic.EmptyLLMReplyError):
        logic.process_chat_message(user_id, "What should I eat?", rag_store=None)

    history = logic.get_chat_history(user_id)["messages"]
    assert all(m.get("role") != "assistant" for m in history)

    logic.delete_user(user_id)


def test_init_db_creates_memory_tables(isolated_db):
    conn = logic.get_conn()
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()
    assert "Chat_Sessions" in tables
    assert "Conversation_Summaries" in tables
    assert "User_Memory_State" in tables


def test_init_db_renames_existing_bmr_column_to_ree(isolated_db):
    conn = logic.get_conn()
    conn.execute("ALTER TABLE Health_Metrics RENAME COLUMN ree TO bmr")
    conn.commit()
    conn.close()

    logic.init_db()

    conn = logic.get_conn()
    columns = {row[1] for row in conn.execute("PRAGMA table_info(Health_Metrics)").fetchall()}
    conn.close()
    assert "ree" in columns
    assert "bmr" not in columns


def test_user_weight_and_delete(isolated_db):
    user_id = logic.create_user_profile(
        name="Test User",
        gender="female",
        birth_date="1992-03-15",
        height_cm=165,
        initial_weight_kg=62,
    )
    latest = logic.get_latest_metrics_bundle(user_id)
    assert latest is not None
    assert latest["weight_kg"] == 62
    assert latest["ree"] > 0

    updated = logic.upsert_weight_entry(user_id, 61.5)
    assert updated["weight_kg"] == 61.5
    assert updated["ree"] > 0
    assert len(logic.get_weight_history(user_id)) >= 1

    logic.delete_user(user_id)
    assert logic.get_user_profile(user_id) is None


def test_build_weight_trend_block_multi_entry(isolated_db):
    user_id = logic.create_user_profile(
        name="Trend User",
        gender="male",
        birth_date="1963-01-01",
        height_cm=175,
        target_weight="105 kg",
        target_timeline="September",
    )
    logic.upsert_weight_entry(user_id, 112.0, recorded_at="2026-03-30 09:00:00")
    logic.upsert_weight_entry(user_id, 111.0, recorded_at="2026-04-01 09:00:00")
    logic.upsert_weight_entry(user_id, 109.5, recorded_at="2026-05-01 09:00:00")
    logic.upsert_weight_entry(user_id, 108.3, recorded_at="2026-06-01 09:00:00")

    user = logic.get_user_profile(user_id)
    history = logic.get_weight_history(user_id)
    block = logic.build_weight_trend_block(user, history)

    assert "112.0 kg" in block
    assert "108.3 kg" in block
    assert "Change: -3.7 kg" in block
    assert "Recent weigh-ins" in block
    assert "Target weight: 105 kg (September)" in block

    prompt = logic.build_coach_prompt(
        user=user,
        latest=logic.get_latest_metrics_bundle(user_id),
        message="Have I made progress over the last three months?",
        rag_context="",
        memory_context={"memory_text": ""},
        weight_history=history,
    )
    assert "[Weight Trend]" in prompt
    assert "Change: -3.7 kg" in prompt

    logic.delete_user(user_id)


def test_build_coach_prompt_allows_safe_calorie_targets(isolated_db):
    user_id = logic.create_user_profile(
        name="Calorie User",
        gender="female",
        birth_date="1991-01-01",
        height_cm=165,
        initial_weight_kg=62.0,
        activity_level="moderately active work",
        goal="lose weight",
    )
    user = logic.get_user_profile(user_id)
    prompt = logic.build_coach_prompt(
        user=user,
        latest=logic.get_latest_metrics_bundle(user_id),
        message="Based on my height, weight and work pattern, how many calories should I aim to eat per day?",
        rag_context="",
        memory_context={"memory_text": ""},
        weight_history=logic.get_weight_history(user_id),
    )

    assert "Calorie-target questions are allowed" in prompt
    assert "Do not refuse ordinary calorie-target questions" in prompt
    assert "300-500 kcal/day below estimated maintenance" in prompt
    assert "Do not recommend eating below REE" in prompt
    assert "do not give 1000 kcal/day as a normal starting goal" in prompt

    logic.delete_user(user_id)


def test_process_chat_message_calorie_target_uses_deterministic_route(isolated_db, monkeypatch):
    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: False)

    user_id = logic.create_user_profile(
        name="Deterministic Calorie User",
        gender="female",
        birth_date="1991-01-01",
        height_cm=165,
        initial_weight_kg=62.0,
        activity_level="moderately active work",
        goal="lose weight",
    )

    result = logic.process_chat_message(
        user_id,
        "Based on my height, weight and work pattern, how many calories should I aim to eat per day?",
        rag_store=None,
    )

    assert result["safety_blocked"] is False
    assert "calorie_target" in result
    assert result["calorie_target"]["target_low"] >= result["calorie_target"]["ree"]
    assert "1000" not in result["reply"]
    assert "estimated maintenance" in result["reply"]

    logic.delete_user(user_id)


def test_build_weight_trend_block_single_entry(isolated_db):
    user_id = logic.create_user_profile(
        name="Single Entry User",
        gender="female",
        birth_date="1990-01-01",
        height_cm=165,
        initial_weight_kg=62.0,
    )
    user = logic.get_user_profile(user_id)
    history = logic.get_weight_history(user_id)
    block = logic.build_weight_trend_block(user, history)

    assert "62" in block
    assert "Log more entries" in block

    logic.delete_user(user_id)


def test_session_close_and_chat_round_trip(isolated_db, monkeypatch):
    monkeypatch.setattr(logic, "check_ollama_reachable", lambda timeout_seconds=2.0: False)

    user_id = logic.create_user_profile(
        name="Memory User",
        gender="male",
        birth_date="1990-01-01",
        height_cm=180,
    )
    session_id, _ = logic.resolve_session(user_id)
    logic.save_chat(user_id, "user", "I want more protein at lunch", session_id=session_id)
    logic.save_chat(user_id, "assistant", "Let's focus on lean protein options.", session_id=session_id)

    close_result = logic.close_user_session(user_id)
    assert close_result["status"] == "closed"
    assert close_result["session_id"] == session_id

    with pytest.raises(logic.OllamaUnavailableError):
        logic.process_chat_message(user_id, "Remind me what we discussed", rag_store=None)

    logic.delete_user(user_id)

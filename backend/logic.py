import json
import os
import re
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_ollama import OllamaLLM
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    RAG_DEPS_AVAILABLE = True
except Exception:
    RAG_DEPS_AVAILABLE = False
    OllamaLLM = None
    DirectoryLoader = None
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None
    HuggingFaceEmbeddings = None
    Chroma = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "database", "health_coach.db")
DEFAULT_EVAL_DB_PATH = os.path.join(BASE_DIR, "database", "eval", "health_coach_eval.db")


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def resolve_db_path_from_env() -> str:
    """Resolve production/default DB path.

    ``DB_PATH`` overrides the default app database. ``EVAL_DB_PATH`` is *not*
    applied here — eval runners must call ``configure_eval_database()`` so the
    FastAPI app never accidentally opens an eval DB.
    """
    override = (os.getenv("DB_PATH") or "").strip()
    if override:
        return _expand_path(override)
    return DEFAULT_DB_PATH


DB_PATH = resolve_db_path_from_env()
PDF_DIR = os.path.join(PROJECT_ROOT, "my_knowledge")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "database", "vector_db")

PROFILE_LIST_FIELDS = ["medical_conditions", "allergies", "food_dislikes"]
PROFILE_TEXT_FIELDS = ["goal", "activity_level", "diet_preference", "budget_level", "target_weight", "target_timeline", "self_description", "coach_notes"]

MEMORY_MODES = {"M0", "M1", "M2", "M3", "M3_MATCH", "RECURSUM", "SESSION_RET"}
MEMORY_MODE_ALIASES = {
    "ZERO": "M0",
    "ANC_ZERO": "M0",
    "FULL": "M3",
    "ANC_FULL": "M3",
    "M3_U": "M3",
    "M3U": "M3",
    "M3_B": "M3_MATCH",
    "M3B": "M3_MATCH",
    "M3_BUDGET": "M3_MATCH",
    "PROD_M2": "M2",
    "LIT_RECURSUM": "RECURSUM",
    "LIT_SESSIONRET": "SESSION_RET",
}

LLM_FALLBACK_REPLY = (
    "I couldn't reach the local model, but based on your saved profile I can still help with "
    "general nutrition coaching. Try generating a 7-day meal plan for a structured recommendation."
)


class OllamaUnavailableError(Exception):
    """Raised when AI features are requested but the local Ollama runtime is not reachable."""


class EmptyLLMReplyError(OllamaUnavailableError):
    """Raised when the model returns no visible text after stripping think tags."""


EMPTY_REPLY_RETRY_SUFFIX = (
    "\n\nPlease provide the final answer directly without a thinking process."
)


def assert_ollama_ready() -> None:
    if not check_ollama_reachable() or OllamaLLM is None:
        raise OllamaUnavailableError(
            "Local AI engine (Ollama) is not running. Start Ollama and ensure the model is pulled, "
            "then try again. See README or run ./start.sh."
        )


def invoke_llm_visible_reply(
    prompt: str,
    *,
    llm: Any = None,
    max_attempts: int = 2,
) -> str:
    """Invoke the chat LLM and return visible text after stripping think tags.

    Retries once (by default) with a short direct-answer hint when the first
    visible reply is empty. Raises EmptyLLMReplyError if still empty.
    """
    assert_ollama_ready()
    client = llm or create_ollama_llm()
    attempts = max(1, int(max_attempts))
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        use_prompt = prompt if attempt == 1 else f"{prompt}{EMPTY_REPLY_RETRY_SUFFIX}"
        try:
            raw = client.invoke(use_prompt)
            text = raw if isinstance(raw, str) else str(raw or "")
            visible = strip_think_tags(text)
            if visible.strip():
                return visible
        except OllamaUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface as runtime unavailability
            last_error = exc
            raise OllamaUnavailableError(
                f"Local AI engine (Ollama) failed during chat: {exc}"
            ) from exc
    detail = "Local AI engine returned an empty reply after stripping thinking content."
    if last_error is not None:
        detail = f"{detail} Last error: {last_error}"
    raise EmptyLLMReplyError(f"{detail} Please try again.")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def session_idle_timeout_minutes() -> int:
    return _env_int("SESSION_IDLE_TIMEOUT_MINUTES", 30)


def active_session_turn_limit() -> int:
    return _env_int("ACTIVE_SESSION_TURN_LIMIT", 4)


def recent_session_summary_limit() -> int:
    return _env_int("RECENT_SESSION_SUMMARY_LIMIT", 2)


def session_summary_max_chars() -> int:
    return _env_int("SESSION_SUMMARY_MAX_CHARS", 800)


def cumulative_summary_max_chars() -> int:
    return _env_int("CUMULATIVE_SUMMARY_MAX_CHARS", 1200)


def rollup_session_threshold() -> int:
    return _env_int("ROLLUP_SESSION_THRESHOLD", 3)


def memory_budget_chars() -> int:
    return _env_int("MEMORY_BUDGET_CHARS", 3500)


def memory_budget_enabled() -> bool:
    """Global whole-blob truncate. Default off: M2 bound by per-component caps only."""
    raw = os.getenv("MEMORY_BUDGET_ENABLED", "false").strip().lower()
    return raw in ("true", "1", "yes")


def active_turn_max_chars() -> int:
    """Per-message cap for M2 active-session turns (component bound, not global axe)."""
    return _env_int("ACTIVE_TURN_MAX_CHARS", 300)


def recursum_summary_max_chars() -> int:
    return _env_int("RECURSUM_SUMMARY_MAX_CHARS", 2400)


def session_ret_max_tokens() -> int:
    return _env_int("SESSION_RET_MAX_TOKENS", 4096)


def session_ret_top_k() -> int:
    return _env_int("SESSION_RET_TOP_K", 5)


def normalize_memory_mode(memory_mode: Optional[str] = None) -> str:
    raw = (memory_mode or os.getenv("MEMORY_MODE", "M2")).upper().replace("-", "_")
    mode = MEMORY_MODE_ALIASES.get(raw, raw)
    return mode if mode in MEMORY_MODES else "M2"


def get_memory_mode() -> str:
    return normalize_memory_mode(None)


def get_summary_model_name() -> str:
    return os.getenv("SUMMARY_MODEL") or os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")


def get_ollama_chat_model_name() -> str:
    return os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")


def get_ollama_num_predict(for_summary: bool = False) -> int:
    if for_summary:
        return _env_int("SUMMARY_NUM_PREDICT", 256)
    return _env_int("OLLAMA_NUM_PREDICT", 768)


def get_meal_plan_num_predict() -> int:
    return _env_int("MEAL_PLAN_NUM_PREDICT", 3072)


def get_meal_plan_temperature() -> float:
    try:
        return float(os.getenv("MEAL_PLAN_TEMPERATURE", "0.1"))
    except (TypeError, ValueError):
        return 0.1


def get_ollama_temperature(for_summary: bool = False) -> float:
    key = "SUMMARY_TEMPERATURE" if for_summary else "OLLAMA_TEMPERATURE"
    default = "0.2" if for_summary else "0.3"
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return 0.2 if for_summary else 0.3


def get_ollama_reasoning(for_summary: bool = False) -> Optional[bool]:
    key = "SUMMARY_REASONING" if for_summary else "OLLAMA_REASONING"
    default = "false"
    raw = os.getenv(key, default).strip().lower()
    if raw in ("false", "0", "no"):
        return False
    if raw in ("true", "1", "yes"):
        return True
    return None


def create_ollama_llm(
    *,
    for_summary: bool = False,
    model: Optional[str] = None,
    num_predict: Optional[int] = None,
    temperature: Optional[float] = None,
    reasoning: Optional[bool] = None,
) -> Any:
    if OllamaLLM is None:
        raise OllamaUnavailableError(
            "Local AI engine (Ollama) dependencies are not available."
        )
    kwargs: Dict[str, Any] = {
        "model": model or (get_summary_model_name() if for_summary else get_ollama_chat_model_name()),
        "base_url": get_ollama_base_url(),
        "num_predict": num_predict if num_predict is not None else get_ollama_num_predict(for_summary=for_summary),
        "temperature": temperature if temperature is not None else get_ollama_temperature(for_summary=for_summary),
    }
    resolved_reasoning = reasoning if reasoning is not None else get_ollama_reasoning(for_summary=for_summary)
    if resolved_reasoning is not None:
        kwargs["reasoning"] = resolved_reasoning
    return OllamaLLM(**kwargs)


def strip_think_tags(text: str) -> str:
    cleaned = text or ""
    open_tag = "<" + "think" + ">"
    close_tag = "</" + "think" + ">"
    lower = cleaned.lower()
    while True:
        start = lower.find(open_tag)
        if start == -1:
            break
        end = lower.find(close_tag, start + len(open_tag))
        if end == -1:
            cleaned = cleaned[:start]
            break
        cleaned = cleaned[:start] + cleaned[end + len(close_tag):]
        lower = cleaned.lower()
    cleaned = re.sub(r"```thinking.*?```", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _tag_prefix_suffix_length(text: str, tag: str) -> int:
    max_len = min(len(text), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if tag.startswith(text[-length:].lower()):
            return length
    return 0


class ThinkTagStreamFilter:
    def __init__(self) -> None:
        self._open_tag = "<" + "think" + ">"
        self._close_tag = "</" + "think" + ">"
        self._inside_think = False
        self._pending = ""

    def feed(self, text: str) -> str:
        data = self._pending + (text or "")
        self._pending = ""
        visible: List[str] = []
        index = 0

        while index < len(data):
            lower = data.lower()
            if self._inside_think:
                end = lower.find(self._close_tag, index)
                if end == -1:
                    keep = _tag_prefix_suffix_length(data[index:], self._close_tag)
                    self._pending = data[len(data) - keep:] if keep else ""
                    return "".join(visible)
                self._inside_think = False
                index = end + len(self._close_tag)
                continue

            start = lower.find(self._open_tag, index)
            if start == -1:
                remaining = data[index:]
                keep = _tag_prefix_suffix_length(remaining, self._open_tag)
                if keep:
                    visible.append(remaining[:-keep])
                    self._pending = remaining[-keep:]
                else:
                    visible.append(remaining)
                return "".join(visible)

            visible.append(data[index:start])
            self._inside_think = True
            index = start + len(self._open_tag)

        return "".join(visible)

    def finish(self) -> str:
        if self._inside_think:
            self._pending = ""
            return ""
        visible = self._pending
        self._pending = ""
        return visible


def _extract_json_object(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = cleaned.find("{")
    if start == -1:
        return cleaned
    depth = 0
    for index, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:index + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
    end = cleaned.rfind("}")
    if end > start:
        return cleaned[start : end + 1]
    return cleaned


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def check_ollama_reachable(timeout_seconds: float = 2.0) -> bool:
    if OllamaLLM is None:
        return False
    base = get_ollama_base_url().rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout_seconds) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False


def get_runtime_health() -> Dict[str, Any]:
    ollama_reachable = check_ollama_reachable()
    return {
        "ollama_reachable": ollama_reachable,
        "ollama_model": get_ollama_chat_model_name(),
        "summary_model": get_summary_model_name(),
        "memory_mode": get_memory_mode(),
        "llm_deps_available": OllamaLLM is not None,
    }


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _truncate_keep_end(text: str, max_chars: int) -> str:
    """Keep the most recent characters (recency window for matched-budget transcript)."""
    text = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[-max_chars:]
    return "..." + text[-(max_chars - 3) :]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def get_db_path() -> str:
    return DB_PATH


def set_db_path(path: str) -> str:
    """Point subsequent ``get_conn()`` / ``init_db()`` calls at ``path``."""
    global DB_PATH
    DB_PATH = _expand_path(path)
    return DB_PATH


def configure_eval_database(
    path: Optional[str] = None,
    *,
    fresh: bool = False,
    init: bool = True,
) -> str:
    """Use an isolated SQLite file for evaluation (does not touch production users).

    Path resolution order:
      1. explicit ``path`` argument
      2. ``EVAL_DB_PATH`` environment variable
      3. ``DEFAULT_EVAL_DB_PATH`` (``backend/database/eval/health_coach_eval.db``)

    When ``fresh=True``, delete the target file first so the run starts empty.
    """
    resolved = (path or "").strip() or (os.getenv("EVAL_DB_PATH") or "").strip() or DEFAULT_EVAL_DB_PATH
    resolved = _expand_path(resolved)
    if os.path.abspath(resolved) == os.path.abspath(DEFAULT_DB_PATH):
        raise ValueError(
            "Refusing to use the production database as EVAL_DB_PATH. "
            "Pass a different path or omit --use-main-db only when intentional."
        )
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if fresh and os.path.exists(resolved):
        os.remove(resolved)
    set_db_path(resolved)
    if init:
        init_db()
    return DB_PATH


def get_conn() -> sqlite3.Connection:
    parent = os.path.dirname(DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_db_connection() -> sqlite3.Connection:
    return get_conn()


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, definition: str) -> None:
    cols = {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _migrate_bmr_to_ree(cur: sqlite3.Cursor) -> None:
    columns = {row[1] for row in cur.execute("PRAGMA table_info(Health_Metrics)").fetchall()}
    if "bmr" in columns and "ree" not in columns:
        cur.execute("ALTER TABLE Health_Metrics RENAME COLUMN bmr TO ree")


def _migrate_legacy_chat_sessions(cur: sqlite3.Cursor) -> None:
    orphan_users = cur.execute(
        """
        SELECT DISTINCT user_id
        FROM Chat_History
        WHERE session_id IS NULL
        """
    ).fetchall()
    for row in orphan_users:
        user_id = int(row["user_id"])
        bounds = cur.execute(
            """
            SELECT MIN(timestamp) AS started_at, MAX(timestamp) AS ended_at, COUNT(*) AS message_count
            FROM Chat_History
            WHERE user_id = ? AND session_id IS NULL
            """,
            (user_id,),
        ).fetchone()
        if not bounds or not bounds["message_count"]:
            continue
        started_at = bounds["started_at"] or utc_now_str()
        ended_at = bounds["ended_at"] or started_at
        user_turns = cur.execute(
            """
            SELECT COUNT(*) AS count
            FROM Chat_History
            WHERE user_id = ? AND session_id IS NULL AND role = 'user'
            """,
            (user_id,),
        ).fetchone()["count"]
        cur.execute(
            """
            INSERT INTO Chat_Sessions (user_id, status, started_at, ended_at, last_message_at, turn_count)
            VALUES (?, 'closed', ?, ?, ?, ?)
            """,
            (user_id, started_at, ended_at, ended_at, int(user_turns or 0)),
        )
        session_id = int(cur.lastrowid)
        cur.execute(
            "UPDATE Chat_History SET session_id = ? WHERE user_id = ? AND session_id IS NULL",
            (session_id, user_id),
        )
        cur.execute(
            """
            INSERT OR IGNORE INTO User_Memory_State (user_id, cumulative_summary, updated_at)
            VALUES (?, '', datetime('now', 'localtime'))
            """,
            (user_id,),
        )


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS User_Profiles (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT NOT NULL CHECK (gender IN ('male', 'female', 'other')),
            birth_date TEXT NOT NULL,
            height_cm REAL NOT NULL CHECK (height_cm > 0 AND height_cm < 300),
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        )
        """
    )
    for column, definition in {
        "goal": "TEXT DEFAULT ''",
        "activity_level": "TEXT DEFAULT ''",
        "diet_preference": "TEXT DEFAULT ''",
        "budget_level": "TEXT DEFAULT ''",
        "medical_conditions": "TEXT DEFAULT '[]'",
        "allergies": "TEXT DEFAULT '[]'",
        "food_dislikes": "TEXT DEFAULT '[]'",
        "target_weight": "TEXT DEFAULT ''",
        "target_timeline": "TEXT DEFAULT ''",
        "self_description": "TEXT DEFAULT ''",
        "coach_notes": "TEXT DEFAULT ''",
    }.items():
        _ensure_column(cur, "User_Profiles", column, definition)

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Health_Metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            recorded_at TEXT NOT NULL,
            weight_kg REAL NOT NULL CHECK (weight_kg > 0 AND weight_kg < 500),
            bmi REAL NOT NULL CHECK (bmi > 0 AND bmi < 100),
            ree REAL NOT NULL CHECK (ree > 0 AND ree < 10000),
            note TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES User_Profiles(user_id) ON DELETE CASCADE
        )
        """
    )
    _migrate_bmr_to_ree(cur)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Chat_History (
            chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES User_Profiles(user_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Meal_Plans (
            plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            goal TEXT,
            plan_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES User_Profiles(user_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Conversation_Summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id INTEGER,
            summary_type TEXT NOT NULL CHECK (summary_type IN ('session', 'rollup')),
            content TEXT NOT NULL,
            key_facts_json TEXT DEFAULT '{}',
            covers_chat_from INTEGER,
            covers_chat_to INTEGER,
            message_count INTEGER DEFAULT 0,
            model_name TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            archived INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES User_Profiles(user_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Chat_Sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('active', 'closed')),
            started_at TEXT NOT NULL,
            ended_at TEXT,
            last_message_at TEXT NOT NULL,
            turn_count INTEGER NOT NULL DEFAULT 0,
            summary_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES User_Profiles(user_id) ON DELETE CASCADE,
            FOREIGN KEY (summary_id) REFERENCES Conversation_Summaries(summary_id) ON DELETE SET NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS User_Memory_State (
            user_id INTEGER PRIMARY KEY,
            cumulative_summary TEXT NOT NULL DEFAULT '',
            last_rollup_at TEXT,
            active_session_id INTEGER,
            updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES User_Profiles(user_id) ON DELETE CASCADE,
            FOREIGN KEY (active_session_id) REFERENCES Chat_Sessions(session_id) ON DELETE SET NULL
        )
        """
    )
    _ensure_column(cur, "Chat_History", "session_id", "INTEGER")
    _ensure_column(cur, "User_Memory_State", "recursum_summary", "TEXT NOT NULL DEFAULT ''")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Session_Memory_Index (
            index_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_estimate INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            UNIQUE(user_id, session_id),
            FOREIGN KEY (user_id) REFERENCES User_Profiles(user_id) ON DELETE CASCADE,
            FOREIGN KEY (session_id) REFERENCES Chat_Sessions(session_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_health_metrics_user_day ON Health_Metrics(user_id, date(recorded_at))")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_health_metrics_user_recorded_at ON Health_Metrics(user_id, recorded_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_chat_history_user_timestamp ON Chat_History(user_id, timestamp DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_chat_history_session ON Chat_History(session_id, timestamp ASC)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_meal_plans_user_updated_at ON Meal_Plans(user_id, updated_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_status ON Chat_Sessions(user_id, status)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_last_message ON Chat_Sessions(user_id, last_message_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_conv_summaries_user_created ON Conversation_Summaries(user_id, created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_conv_summaries_session ON Conversation_Summaries(session_id)")
    _migrate_legacy_chat_sessions(cur)
    conn.commit()
    conn.close()


def utc_now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _to_json_array(value: Optional[List[str] | str]) -> str:
    if value is None:
        return json.dumps([])
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return json.dumps(parts, ensure_ascii=False)
    return json.dumps([str(part).strip() for part in value if str(part).strip()], ensure_ascii=False)


def _from_json_array(value: Any) -> List[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [part.strip() for part in str(value).split(",") if part.strip()]


def calculate_age(birth_date_str: str) -> int:
    formats = ["%Y%m%d", "%Y-%m-%d", "%d/%m/%Y"]
    birth_date = None
    for fmt in formats:
        try:
            birth_date = datetime.strptime(str(birth_date_str), fmt)
            break
        except ValueError:
            continue
    if birth_date is None:
        return 30
    today = datetime.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / ((height_cm / 100) ** 2), 2)


def calculate_ree(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    gender = gender.lower()
    if gender == "male":
        return round((10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5, 2)
    if gender == "female":
        return round((10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161, 2)
    return round((10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 78, 2)


def bmi_label(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def create_user_profile(name: str, gender: str, birth_date: str, height_cm: float, initial_weight_kg: Optional[float] = None, **profile_fields: Any) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO User_Profiles (
            name, gender, birth_date, height_cm,
            goal, activity_level, diet_preference, budget_level,
            medical_conditions, allergies, food_dislikes,
            target_weight, target_timeline, self_description, coach_notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name.strip(), gender.lower(), birth_date.strip(), float(height_cm),
            str(profile_fields.get("goal", "") or "").strip(),
            str(profile_fields.get("activity_level", "") or "").strip(),
            str(profile_fields.get("diet_preference", "") or "").strip(),
            str(profile_fields.get("budget_level", "") or "").strip(),
            _to_json_array(profile_fields.get("medical_conditions")),
            _to_json_array(normalize_allergy_list(profile_fields.get("allergies"))),
            _to_json_array(profile_fields.get("food_dislikes")),
            str(profile_fields.get("target_weight", "") or "").strip(),
            str(profile_fields.get("target_timeline", "") or "").strip(),
            str(profile_fields.get("self_description", "") or "").strip(),
            str(profile_fields.get("coach_notes", "") or "").strip(),
        ),
    )
    user_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    if initial_weight_kg is not None:
        upsert_weight_entry(user_id, float(initial_weight_kg))
    return user_id


def update_user_profile(user_id: int, name: str, gender: str, birth_date: str, height_cm: float, **profile_fields: Any) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE User_Profiles
        SET name = ?, gender = ?, birth_date = ?, height_cm = ?,
            goal = ?, activity_level = ?, diet_preference = ?, budget_level = ?,
            medical_conditions = ?, allergies = ?, food_dislikes = ?,
            target_weight = ?, target_timeline = ?, self_description = ?, coach_notes = ?,
            updated_at = datetime('now', 'localtime')
        WHERE user_id = ?
        """,
        (
            name.strip(), gender.lower(), birth_date.strip(), float(height_cm),
            str(profile_fields.get("goal", "") or "").strip(),
            str(profile_fields.get("activity_level", "") or "").strip(),
            str(profile_fields.get("diet_preference", "") or "").strip(),
            str(profile_fields.get("budget_level", "") or "").strip(),
            _to_json_array(profile_fields.get("medical_conditions")),
            _to_json_array(normalize_allergy_list(profile_fields.get("allergies"))),
            _to_json_array(profile_fields.get("food_dislikes")),
            str(profile_fields.get("target_weight", "") or "").strip(),
            str(profile_fields.get("target_timeline", "") or "").strip(),
            str(profile_fields.get("self_description", "") or "").strip(),
            str(profile_fields.get("coach_notes", "") or "").strip(),
            int(user_id),
        ),
    )
    conn.commit()
    conn.close()


def delete_user(user_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM User_Profiles WHERE user_id = ?", (int(user_id),))
    conn.commit()
    conn.close()


def _row_to_user(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    for field in PROFILE_LIST_FIELDS:
        data[field] = _from_json_array(data.get(field))
    return data


def get_all_users() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM User_Profiles ORDER BY name COLLATE NOCASE").fetchall()
    conn.close()
    return [_row_to_user(row) for row in rows]


def get_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM User_Profiles WHERE user_id = ?", (int(user_id),)).fetchone()
    conn.close()
    return _row_to_user(row) if row else None


def _get_user_meta_for_calculation(user_id: int) -> Tuple[float, str, str]:
    user = get_user_profile(user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")
    return float(user["height_cm"]), str(user["gender"]), str(user["birth_date"])


def upsert_weight_entry(user_id: int, weight_kg: float, recorded_at: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
    if recorded_at is None:
        recorded_at = utc_now_str()
    height_cm, gender, birth_date = _get_user_meta_for_calculation(user_id)
    age = calculate_age(birth_date)
    bmi = calculate_bmi(float(weight_kg), height_cm)
    ree = calculate_ree(float(weight_kg), height_cm, age, gender)
    conn = get_conn()
    cur = conn.cursor()
    existing = cur.execute(
        "SELECT metric_id FROM Health_Metrics WHERE user_id = ? AND date(recorded_at) = date(?) LIMIT 1",
        (int(user_id), recorded_at),
    ).fetchone()
    if existing:
        metric_id = int(existing["metric_id"])
        cur.execute(
            "UPDATE Health_Metrics SET recorded_at = ?, weight_kg = ?, bmi = ?, ree = ?, note = ?, updated_at = datetime('now', 'localtime') WHERE metric_id = ?",
            (recorded_at, float(weight_kg), bmi, ree, note, metric_id),
        )
    else:
        cur.execute(
            "INSERT INTO Health_Metrics (user_id, recorded_at, weight_kg, bmi, ree, note) VALUES (?, ?, ?, ?, ?, ?)",
            (int(user_id), recorded_at, float(weight_kg), bmi, ree, note),
        )
        metric_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return {"metric_id": metric_id, "user_id": int(user_id), "recorded_at": recorded_at, "weight_kg": float(weight_kg), "bmi": bmi, "ree": ree, "bmi_label": bmi_label(bmi), "note": note}


def add_new_weight_entry(user_id: int, weight_kg: float, *_args, **_kwargs) -> Dict[str, Any]:
    return upsert_weight_entry(user_id, weight_kg)


def update_weight_record(metric_id: int, user_id: int, new_weight_kg: float, recorded_at: str, note: Optional[str] = None) -> Dict[str, Any]:
    height_cm, gender, birth_date = _get_user_meta_for_calculation(user_id)
    age = calculate_age(birth_date)
    bmi = calculate_bmi(float(new_weight_kg), height_cm)
    ree = calculate_ree(float(new_weight_kg), height_cm, age, gender)
    conn = get_conn()
    conn.execute(
        "UPDATE Health_Metrics SET recorded_at = ?, weight_kg = ?, bmi = ?, ree = ?, note = ?, updated_at = datetime('now', 'localtime') WHERE metric_id = ? AND user_id = ?",
        (recorded_at, float(new_weight_kg), bmi, ree, note, int(metric_id), int(user_id)),
    )
    conn.commit()
    conn.close()
    return {"metric_id": int(metric_id), "user_id": int(user_id), "recorded_at": recorded_at, "weight_kg": float(new_weight_kg), "bmi": bmi, "ree": ree, "bmi_label": bmi_label(bmi), "note": note}


def delete_weight_record(metric_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM Health_Metrics WHERE metric_id = ?", (int(metric_id),))
    conn.commit()
    conn.close()


def get_latest_metrics_bundle(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        "SELECT metric_id, user_id, recorded_at, weight_kg, bmi, ree, note FROM Health_Metrics WHERE user_id = ? ORDER BY datetime(recorded_at) DESC, metric_id DESC LIMIT 1",
        (int(user_id),),
    ).fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    data["bmi_label"] = bmi_label(float(data["bmi"]))
    data["record_date"] = str(data["recorded_at"]).split(" ")[0]
    return data


def get_latest_metrics(user_id: int) -> Optional[Tuple[float, float, float]]:
    latest = get_latest_metrics_bundle(user_id)
    if not latest:
        return None
    return (float(latest["weight_kg"]), float(latest["bmi"]), float(latest["ree"]))


def get_weight_history(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT metric_id, user_id, recorded_at, date(recorded_at) AS record_date, weight_kg, bmi, ree, note FROM Health_Metrics WHERE user_id = ? ORDER BY datetime(recorded_at) ASC, metric_id ASC",
        (int(user_id),),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def ensure_user_memory_state(user_id: int) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT OR IGNORE INTO User_Memory_State (user_id, cumulative_summary, updated_at)
        VALUES (?, '', datetime('now', 'localtime'))
        """,
        (int(user_id),),
    )
    conn.commit()
    conn.close()


def get_user_memory_state(user_id: int) -> Dict[str, Any]:
    ensure_user_memory_state(user_id)
    conn = get_conn()
    row = conn.execute("SELECT * FROM User_Memory_State WHERE user_id = ?", (int(user_id),)).fetchone()
    conn.close()
    return dict(row) if row else {
        "user_id": user_id,
        "cumulative_summary": "",
        "recursum_summary": "",
        "active_session_id": None,
    }


def get_active_session(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        """
        SELECT * FROM Chat_Sessions
        WHERE user_id = ? AND status = 'active'
        ORDER BY datetime(last_message_at) DESC, session_id DESC
        LIMIT 1
        """,
        (int(user_id),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _create_session(user_id: int) -> int:
    now = utc_now_str()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO User_Memory_State (user_id, cumulative_summary, updated_at)
        VALUES (?, '', datetime('now', 'localtime'))
        """,
        (int(user_id),),
    )
    cur.execute(
        """
        INSERT INTO Chat_Sessions (user_id, status, started_at, last_message_at, turn_count)
        VALUES (?, 'active', ?, ?, 0)
        """,
        (int(user_id), now, now),
    )
    session_id = int(cur.lastrowid)
    cur.execute(
        """
        UPDATE User_Memory_State
        SET active_session_id = ?, updated_at = datetime('now', 'localtime')
        WHERE user_id = ?
        """,
        (session_id, int(user_id)),
    )
    conn.commit()
    conn.close()
    return session_id


def _session_message_count(session_id: int) -> int:
    conn = get_conn()
    count = conn.execute(
        "SELECT COUNT(*) AS count FROM Chat_History WHERE session_id = ?",
        (int(session_id),),
    ).fetchone()["count"]
    conn.close()
    return int(count or 0)


def close_session(session_id: int, user_id: int, trigger_summarization: bool = True) -> Optional[int]:
    conn = get_conn()
    cur = conn.cursor()
    session = cur.execute(
        "SELECT * FROM Chat_Sessions WHERE session_id = ? AND user_id = ?",
        (int(session_id), int(user_id)),
    ).fetchone()
    if not session or session["status"] == "closed":
        conn.close()
        return None
    now = utc_now_str()
    cur.execute(
        """
        UPDATE Chat_Sessions
        SET status = 'closed', ended_at = ?, last_message_at = ?
        WHERE session_id = ?
        """,
        (now, now, int(session_id)),
    )
    memory = cur.execute("SELECT active_session_id FROM User_Memory_State WHERE user_id = ?", (int(user_id),)).fetchone()
    if memory and memory["active_session_id"] == session_id:
        cur.execute(
            """
            UPDATE User_Memory_State
            SET active_session_id = NULL, updated_at = datetime('now', 'localtime')
            WHERE user_id = ?
            """,
            (int(user_id),),
        )
    conn.commit()
    conn.close()

    if not trigger_summarization:
        return None
    if _session_message_count(session_id) < 2:
        return None

    def _run_summary() -> None:
        try:
            summarize_session(session_id, user_id)
            maybe_rollup_memory(user_id)
        except Exception:
            pass

    thread = threading.Thread(target=_run_summary, daemon=True)
    thread.start()
    return session_id


def resolve_session(user_id: int, force_new: bool = False) -> Tuple[int, Optional[int]]:
    ensure_user_memory_state(user_id)
    closed_session_id: Optional[int] = None
    active = get_active_session(user_id)

    if force_new and active:
        closed_session_id = close_session(int(active["session_id"]), user_id)
        active = None

    if active:
        last_at = active.get("last_message_at") or active.get("started_at")
        try:
            last_dt = datetime.strptime(str(last_at), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            last_dt = datetime.now() - timedelta(minutes=session_idle_timeout_minutes() + 1)
        idle_limit = timedelta(minutes=session_idle_timeout_minutes())
        if datetime.now() - last_dt > idle_limit:
            closed_session_id = close_session(int(active["session_id"]), user_id)
            active = None

    if active:
        return int(active["session_id"]), closed_session_id

    new_session_id = _create_session(user_id)
    return new_session_id, closed_session_id


def get_chat_history(user_id: int) -> Dict[str, Any]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT chat_id, role, content, timestamp, session_id
        FROM Chat_History
        WHERE user_id = ?
        ORDER BY datetime(timestamp) ASC, chat_id ASC
        """,
        (int(user_id),),
    ).fetchall()
    sessions = conn.execute(
        """
        SELECT s.session_id, s.status, s.started_at, s.ended_at, s.turn_count, s.summary_id,
               cs.content AS summary_content
        FROM Chat_Sessions s
        LEFT JOIN Conversation_Summaries cs ON cs.summary_id = s.summary_id
        WHERE s.user_id = ?
        ORDER BY datetime(s.started_at) DESC, s.session_id DESC
        """,
        (int(user_id),),
    ).fetchall()
    conn.close()
    messages = [dict(row) for row in rows]
    session_list = []
    for row in sessions:
        data = dict(row)
        preview = _truncate(data.pop("summary_content", "") or "", 120)
        data["summary_preview"] = preview
        session_list.append(data)
    return {"messages": messages, "sessions": session_list}


def get_session_messages(session_id: int, limit: Optional[int] = None, exclude_current_user_message: bool = False) -> List[Dict[str, Any]]:
    conn = get_conn()
    query = """
        SELECT chat_id, role, content, timestamp
        FROM Chat_History
        WHERE session_id = ?
        ORDER BY datetime(timestamp) ASC, chat_id ASC
    """
    rows = conn.execute(query, (int(session_id),)).fetchall()
    conn.close()
    messages = [dict(row) for row in rows]
    if exclude_current_user_message and messages and messages[-1]["role"] == "user":
        messages = messages[:-1]
    if limit is not None and limit > 0:
        messages = messages[-limit:]
    return messages


def save_chat(user_id: int, role: str, content: str, session_id: Optional[int] = None) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO Chat_History (user_id, role, content, timestamp, session_id)
        VALUES (?, ?, ?, datetime('now', 'localtime'), ?)
        """,
        (int(user_id), role, content, int(session_id) if session_id is not None else None),
    )
    chat_id = int(cur.lastrowid)
    if session_id is not None:
        if role == "user":
            cur.execute(
                """
                UPDATE Chat_Sessions
                SET last_message_at = datetime('now', 'localtime'), turn_count = turn_count + 1
                WHERE session_id = ? AND user_id = ?
                """,
                (int(session_id), int(user_id)),
            )
        else:
            cur.execute(
                """
                UPDATE Chat_Sessions
                SET last_message_at = datetime('now', 'localtime')
                WHERE session_id = ? AND user_id = ?
                """,
                (int(session_id), int(user_id)),
            )
    conn.commit()
    conn.close()
    return chat_id


def list_user_sessions(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT s.session_id, s.status, s.started_at, s.ended_at, s.last_message_at,
               s.turn_count, s.summary_id, cs.content AS summary_content
        FROM Chat_Sessions s
        LEFT JOIN Conversation_Summaries cs ON cs.summary_id = s.summary_id
        WHERE s.user_id = ?
        ORDER BY datetime(s.started_at) DESC, s.session_id DESC
        LIMIT ?
        """,
        (int(user_id), int(limit)),
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        data = dict(row)
        data["summary_preview"] = _truncate(data.pop("summary_content", "") or "", 120)
        result.append(data)
    return result


def list_user_summaries(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT summary_id, user_id, session_id, summary_type, content, message_count,
               model_name, created_at, archived
        FROM Conversation_Summaries
        WHERE user_id = ?
        ORDER BY datetime(created_at) DESC, summary_id DESC
        LIMIT ?
        """,
        (int(user_id), int(limit)),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_user_memory_bundle(user_id: int) -> Dict[str, Any]:
    state = get_user_memory_state(user_id)
    active = get_active_session(user_id)
    recent_summaries = list_user_summaries(user_id, limit=recent_session_summary_limit() + rollup_session_threshold())
    session_summaries = [
        item for item in recent_summaries
        if item.get("summary_type") == "session" and not item.get("archived")
    ][:recent_session_summary_limit()]
    return {
        "user_id": user_id,
        "cumulative_summary": state.get("cumulative_summary") or "",
        "last_rollup_at": state.get("last_rollup_at"),
        "active_session": active,
        "recent_session_summaries": session_summaries,
    }


def _invoke_summary_llm(prompt: str) -> str:
    if OllamaLLM is None:
        return ""
    try:
        llm = create_ollama_llm(for_summary=True)
        return strip_think_tags((llm.invoke(prompt) or "").strip())
    except Exception:
        return ""


def _fallback_session_summary(messages: List[Dict[str, Any]]) -> str:
    user_snippets = []
    for msg in messages:
        if msg.get("role") == "user":
            user_snippets.append(_truncate(str(msg.get("content", "")), 100))
    if not user_snippets:
        return ""
    joined = " | ".join(user_snippets[-3:])
    return _truncate(f"Session notes (fallback): {joined}", session_summary_max_chars())


def summarize_session(session_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    existing = conn.execute(
        "SELECT summary_id FROM Chat_Sessions WHERE session_id = ? AND user_id = ? AND summary_id IS NOT NULL",
        (int(session_id), int(user_id)),
    ).fetchone()
    conn.close()
    if existing:
        return None

    messages = get_session_messages(session_id)
    if len(messages) < 2:
        return None

    user = get_user_profile(user_id)
    if not user:
        return None

    state = get_user_memory_state(user_id)
    transcript_lines = [f"{msg['role'].upper()}: {msg['content']}" for msg in messages]
    transcript = "\n".join(transcript_lines)
    profile_constraints = build_profile_summary(user, get_latest_metrics_bundle(user_id))
    summary_prompt = f"""Summarize this health coaching conversation for future sessions.
Do not invent medical facts. Preserve goals, preferences, constraints, progress notes, advice already given, and open questions.
Use these section headings:
## Coaching Context
## Goals and Motivation
## Progress and Metrics Mentioned
## Preferences and Constraints Discussed
## Advice Already Given
## Open Questions / Next Steps

Keep the summary under {session_summary_max_chars()} characters.

Existing long-term memory:
{state.get('cumulative_summary') or '(none)'}

Profile constraints (authoritative for allergies/conditions):
{profile_constraints}

Conversation transcript:
{transcript}
"""
    content = _invoke_summary_llm(summary_prompt)
    if len(content) < 40:
        content = _fallback_session_summary(messages)
    if not content:
        return None

    content = _truncate(content, session_summary_max_chars())
    covers_from = messages[0]["chat_id"]
    covers_to = messages[-1]["chat_id"]
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO Conversation_Summaries (
            user_id, session_id, summary_type, content, covers_chat_from, covers_chat_to,
            message_count, model_name
        )
        VALUES (?, ?, 'session', ?, ?, ?, ?, ?)
        """,
        (
            int(user_id), int(session_id), content, int(covers_from), int(covers_to),
            len(messages), get_summary_model_name(),
        ),
    )
    summary_id = int(cur.lastrowid)
    cur.execute(
        "UPDATE Chat_Sessions SET summary_id = ? WHERE session_id = ? AND user_id = ?",
        (summary_id, int(session_id), int(user_id)),
    )
    conn.commit()
    conn.close()
    return {
        "summary_id": summary_id,
        "session_id": session_id,
        "content": content,
        "message_count": len(messages),
    }


def maybe_rollup_memory(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT summary_id, content, created_at
        FROM Conversation_Summaries
        WHERE user_id = ? AND summary_type = 'session' AND archived = 0
        ORDER BY datetime(created_at) ASC, summary_id ASC
        """,
        (int(user_id),),
    ).fetchall()
    conn.close()
    if len(rows) <= rollup_session_threshold():
        return None
    return rollup_memory(user_id)


def rollup_memory(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT summary_id, content
        FROM Conversation_Summaries
        WHERE user_id = ? AND summary_type = 'session' AND archived = 0
        ORDER BY datetime(created_at) ASC, summary_id ASC
        LIMIT 2
        """,
        (int(user_id),),
    ).fetchall()
    conn.close()
    if len(rows) < 2:
        return None

    state = get_user_memory_state(user_id)
    old_blocks = [row["content"] for row in rows]
    rollup_prompt = f"""Merge the existing long-term coaching memory with the two session summaries below.
Produce one concise cumulative memory for future coaching sessions.
Use the same section headings as a session summary.
Do not invent facts. Keep under {cumulative_summary_max_chars()} characters.

Existing cumulative memory:
{state.get('cumulative_summary') or '(none)'}

Session summary A:
{old_blocks[0]}

Session summary B:
{old_blocks[1]}
"""
    merged = _invoke_summary_llm(rollup_prompt)
    if len(merged) < 40:
        merged = _truncate(
            f"{state.get('cumulative_summary', '').strip()}\n\n{old_blocks[0]}\n\n{old_blocks[1]}".strip(),
            cumulative_summary_max_chars(),
        )
    merged = _truncate(merged, cumulative_summary_max_chars())

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO Conversation_Summaries (user_id, summary_type, content, message_count, model_name)
        VALUES (?, 'rollup', ?, 0, ?)
        """,
        (int(user_id), merged, get_summary_model_name()),
    )
    rollup_id = int(cur.lastrowid)
    for row in rows:
        cur.execute("UPDATE Conversation_Summaries SET archived = 1 WHERE summary_id = ?", (int(row["summary_id"]),))
    cur.execute(
        """
        UPDATE User_Memory_State
        SET cumulative_summary = ?, last_rollup_at = datetime('now', 'localtime'),
            updated_at = datetime('now', 'localtime')
        WHERE user_id = ?
        """,
        (merged, int(user_id)),
    )
    conn.commit()
    conn.close()
    return {"rollup_id": rollup_id, "cumulative_summary": merged}


def regenerate_session_summary(session_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    session = conn.execute(
        "SELECT summary_id FROM Chat_Sessions WHERE session_id = ? AND user_id = ?",
        (int(session_id), int(user_id)),
    ).fetchone()
    if not session:
        conn.close()
        return None
    if session["summary_id"]:
        conn.execute("DELETE FROM Conversation_Summaries WHERE summary_id = ?", (int(session["summary_id"]),))
        conn.execute(
            "UPDATE Chat_Sessions SET summary_id = NULL WHERE session_id = ? AND user_id = ?",
            (int(session_id), int(user_id)),
        )
    conn.commit()
    conn.close()
    return summarize_session(session_id, user_id)


def _format_session_transcript(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _bm25_scores(query: str, documents: List[str]) -> List[float]:
    import math
    from collections import Counter

    query_tokens = _tokenize_for_bm25(query)
    if not documents:
        return []
    if not query_tokens:
        return [0.0] * len(documents)

    doc_tokens = [_tokenize_for_bm25(doc) for doc in documents]
    n_docs = len(documents)
    avgdl = sum(len(tokens) for tokens in doc_tokens) / n_docs
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1

    scores: List[float] = []
    k1, b = 1.5, 0.75
    for tokens in doc_tokens:
        score = 0.0
        dl = len(tokens)
        tf_map = Counter(tokens)
        for term in query_tokens:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            denom = tf + k1 * (1 - b + b * dl / avgdl) if avgdl else tf + k1
            score += idf * (tf * (k1 + 1)) / denom
        scores.append(score)
    return scores


def clear_session_memory_index(user_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM Session_Memory_Index WHERE user_id = ?", (int(user_id),))
    conn.commit()
    conn.close()


def index_closed_session_for_retrieval(session_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    messages = get_session_messages(session_id)
    if not messages:
        return None
    content = _format_session_transcript(messages)
    token_estimate = _estimate_tokens(content)
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO Session_Memory_Index (user_id, session_id, content, token_estimate, created_at)
        VALUES (?, ?, ?, ?, datetime('now', 'localtime'))
        ON CONFLICT(user_id, session_id) DO UPDATE SET
            content = excluded.content,
            token_estimate = excluded.token_estimate,
            created_at = excluded.created_at
        """,
        (int(user_id), int(session_id), content, int(token_estimate)),
    )
    conn.commit()
    conn.close()
    return {"session_id": session_id, "token_estimate": token_estimate, "content_chars": len(content)}


def update_recursum_memory(session_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    messages = get_session_messages(session_id)
    if len(messages) < 2:
        return None
    transcript = _format_session_transcript(messages)
    state = get_user_memory_state(user_id)
    previous = (state.get("recursum_summary") or "").strip()
    prompt = f"""Update the rolling conversation summary by merging the previous summary with the new session transcript.
Preserve factual details needed for future question answering. Do not invent facts.
Keep under {recursum_summary_max_chars()} characters.

Previous summary:
{previous or '(none)'}

New session transcript:
{transcript}
"""
    updated = _invoke_summary_llm(prompt)
    if len(updated) < 20:
        merged = f"{previous}\n\n{transcript}".strip() if previous else transcript
        updated = _truncate(merged, recursum_summary_max_chars())
    updated = _truncate(updated, recursum_summary_max_chars())

    conn = get_conn()
    conn.execute(
        """
        UPDATE User_Memory_State
        SET recursum_summary = ?, updated_at = datetime('now', 'localtime')
        WHERE user_id = ?
        """,
        (updated, int(user_id)),
    )
    conn.commit()
    conn.close()
    return {"session_id": session_id, "recursum_summary_chars": len(updated)}


def finalize_closed_session_memory(session_id: int, user_id: int, memory_mode: Optional[str] = None) -> None:
    mode = normalize_memory_mode(memory_mode)
    if mode == "RECURSUM":
        update_recursum_memory(session_id, user_id)
    elif mode == "SESSION_RET":
        index_closed_session_for_retrieval(session_id, user_id)
    elif mode in {"M1", "M2"}:
        if _session_message_count(session_id) >= 2:
            summarize_session(session_id, user_id)
            if mode == "M2":
                maybe_rollup_memory(user_id)


def retrieve_sessions_for_memory(
    user_id: int,
    query: Optional[str],
    exclude_session_id: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT index_id, session_id, content, token_estimate, created_at
        FROM Session_Memory_Index
        WHERE user_id = ? AND session_id != ?
        ORDER BY datetime(created_at) ASC, index_id ASC
        """,
        (int(user_id), int(exclude_session_id)),
    ).fetchall()
    conn.close()
    indexed = [dict(row) for row in rows]
    if not indexed:
        return "", []

    documents = [row["content"] for row in indexed]
    query_text = (query or "").strip()
    if query_text:
        scores = _bm25_scores(query_text, documents)
        ranked = sorted(zip(indexed, scores), key=lambda item: item[1], reverse=True)
    else:
        ranked = [(row, 0.0) for row in reversed(indexed)]

    selected: List[Dict[str, Any]] = []
    blocks: List[str] = []
    used_tokens = 0
    budget = session_ret_max_tokens()
    top_k = session_ret_top_k()
    for row, score in ranked[:top_k]:
        block_tokens = int(row.get("token_estimate") or _estimate_tokens(row["content"]))
        if used_tokens and used_tokens + block_tokens > budget:
            continue
        if block_tokens > budget and not blocks:
            trimmed = _truncate(row["content"], budget * 4)
            block_tokens = _estimate_tokens(trimmed)
            blocks.append(trimmed)
            selected.append({**row, "bm25_score": score, "trimmed": True})
            used_tokens += block_tokens
            break
        blocks.append(row["content"])
        selected.append({**row, "bm25_score": score, "trimmed": False})
        used_tokens += block_tokens
        if used_tokens >= budget:
            break

    return "\n\n---\n\n".join(blocks), selected


def build_memory_context(
    user_id: int,
    session_id: int,
    memory_mode: Optional[str] = None,
    query: Optional[str] = None,
    match_chars: Optional[int] = None,
) -> Dict[str, Any]:
    mode = normalize_memory_mode(memory_mode)

    state = get_user_memory_state(user_id)
    cumulative = (state.get("cumulative_summary") or "").strip()
    recent_summaries: List[Dict[str, Any]] = []
    active_turns: List[Dict[str, Any]] = []
    full_transcript = ""

    if mode in {"M1", "M2"}:
        if mode == "M1":
            recent_summaries = []
        else:
            conn = get_conn()
            rows = conn.execute(
                """
                SELECT cs.summary_id, cs.content, cs.created_at, cs.session_id
                FROM Conversation_Summaries cs
                WHERE cs.user_id = ? AND cs.summary_type = 'session' AND cs.archived = 0
                  AND (cs.session_id IS NULL OR cs.session_id != ?)
                ORDER BY datetime(cs.created_at) DESC, cs.summary_id DESC
                LIMIT ?
                """,
                (int(user_id), int(session_id), recent_session_summary_limit()),
            ).fetchall()
            conn.close()
            recent_summaries = [dict(row) for row in rows]

        active_turns = get_session_messages(
            session_id,
            limit=active_session_turn_limit(),
            exclude_current_user_message=True,
        ) if mode == "M2" else []

    recursum_summary = (state.get("recursum_summary") or "").strip()
    session_ret_text = ""
    session_ret_selected: List[Dict[str, Any]] = []

    if mode == "RECURSUM":
        active_turns = get_session_messages(
            session_id,
            limit=active_session_turn_limit(),
            exclude_current_user_message=True,
        )

    if mode == "SESSION_RET":
        session_ret_text, session_ret_selected = retrieve_sessions_for_memory(
            user_id, query, int(session_id)
        )
        active_turns = get_session_messages(
            session_id,
            limit=active_session_turn_limit(),
            exclude_current_user_message=True,
        )

    if mode in {"M3", "M3_MATCH"}:
        conn = get_conn()
        rows = conn.execute(
            """
            SELECT role, content, timestamp
            FROM Chat_History
            WHERE user_id = ? AND session_id != ?
            ORDER BY datetime(timestamp) ASC, chat_id ASC
            """,
            (int(user_id), int(session_id)),
        ).fetchall()
        conn.close()
        full_transcript = "\n".join(f"{row['role'].upper()}: {row['content']}" for row in rows)

    long_term_block = cumulative if mode in {"M1", "M2"} and cumulative else "（尚無長期記憶）"
    if mode == "M0":
        long_term_block = "（尚無長期記憶）"

    recent_lines = []
    for item in recent_summaries:
        date_label = str(item.get("created_at", "")).split(" ")[0] or "recent"
        excerpt = _truncate(item.get("content", ""), session_summary_max_chars())
        recent_lines.append(f"- Session {date_label}: {excerpt}")
    recent_block = "\n".join(recent_lines) if recent_lines else "（尚無近期 session 摘要）"

    turn_cap = active_turn_max_chars()
    turn_lines = []
    for msg in active_turns:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = _truncate(str(msg.get("content") or ""), turn_cap)
        turn_lines.append(f"{role}: {content}")
    turns_block = "\n".join(turn_lines) if turn_lines else "（當前 session 尚無先前對話）"

    matched_budget_chars: Optional[int] = None
    if mode == "M3" and full_transcript:
        memory_text = f"[Full Conversation Transcript]\n{full_transcript}"
    elif mode == "M3_MATCH":
        # Recency window sized to match_chars (typically len of M2 memory_text).
        if match_chars is None:
            m2_ctx = build_memory_context(user_id, session_id, memory_mode="M2", query=query)
            matched_budget_chars = len(m2_ctx.get("memory_text") or "")
        else:
            matched_budget_chars = max(0, int(match_chars))
        body = _truncate_keep_end(full_transcript, matched_budget_chars)
        memory_text = f"[Matched-Budget Recent Transcript]\n{body}"
    elif mode == "RECURSUM":
        recursum_block = recursum_summary or "（尚無 rolling summary）"
        memory_parts = [
            f"[Rolling Conversation Summary]\n{recursum_block}",
            f"[Current Conversation — recent turns only]\n{turns_block}",
        ]
        memory_text = "\n\n".join(memory_parts)
    elif mode == "SESSION_RET":
        retrieved_block = session_ret_text or "（尚無檢索到的 session）"
        memory_parts = [
            f"[Retrieved Session Memory]\n{retrieved_block}",
            f"[Current Conversation — recent turns only]\n{turns_block}",
        ]
        memory_text = "\n\n".join(memory_parts)
    else:
        memory_parts = [
            f"[Long-Term Coaching Memory]\n{long_term_block}",
            f"[Recent Session Summaries]\n{recent_block}",
            f"[Current Conversation — recent turns only]\n{turns_block}",
        ]
        memory_text = "\n\n".join(memory_parts)

    if mode == "M0":
        memory_text = ""

    estimated_tokens = _estimate_tokens(memory_text)
    truncated_by_global_budget = False
    if (
        memory_budget_enabled()
        and mode not in {"M3", "M3_MATCH", "SESSION_RET"}
        and len(memory_text) > memory_budget_chars()
    ):
        memory_text = _truncate(memory_text, memory_budget_chars())
        estimated_tokens = _estimate_tokens(memory_text)
        truncated_by_global_budget = True

    return {
        "memory_text": memory_text,
        "memory_used": {
            "memory_mode": mode,
            "cumulative_summary_included": bool(cumulative) and mode in {"M1", "M2"},
            "recent_session_summaries_count": len(recent_summaries) if mode == "M2" else 0,
            "active_session_turns_included": len(active_turns) if mode in {"M2", "RECURSUM", "SESSION_RET"} else 0,
            "full_transcript_included": mode == "M3" and bool(full_transcript),
            "matched_budget_transcript": mode == "M3_MATCH",
            "matched_budget_chars": matched_budget_chars if mode == "M3_MATCH" else None,
            "recursum_summary_included": mode == "RECURSUM" and bool(recursum_summary),
            "session_ret_sessions_count": len(session_ret_selected) if mode == "SESSION_RET" else 0,
            "session_ret_budget_tokens": session_ret_max_tokens() if mode == "SESSION_RET" else None,
            "estimated_memory_tokens": estimated_tokens,
            "memory_budget_enabled": memory_budget_enabled(),
            "truncated_by_global_budget": truncated_by_global_budget,
        },
    }


def weight_trend_recent_limit() -> int:
    return _env_int("WEIGHT_TREND_RECENT_LIMIT", 5)


def _metric_record_date(entry: Dict[str, Any]) -> str:
    raw = entry.get("record_date") or entry.get("recorded_at") or ""
    return str(raw).split(" ")[0] or "unknown"


def build_weight_trend_block(
    user: Optional[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]],
) -> str:
    entries = list(history or [])
    if not entries:
        return "No weight history logged yet."

    if len(entries) == 1:
        only = entries[0]
        return (
            f"Single weigh-in on {_metric_record_date(only)}: {only['weight_kg']} kg. "
            "Log more entries to establish a trend."
        )

    start = entries[0]
    current = entries[-1]
    start_kg = float(start["weight_kg"])
    current_kg = float(current["weight_kg"])
    delta_kg = round(current_kg - start_kg, 1)
    delta_sign = "+" if delta_kg > 0 else ""

    lines = [
        f"Period: {_metric_record_date(start)} to {_metric_record_date(current)} ({len(entries)} entries)",
        f"Start weight: {start_kg} kg → Current: {current_kg} kg",
        f"Change: {delta_sign}{delta_kg} kg",
    ]

    recent = entries[-weight_trend_recent_limit():]
    recent_parts = [f"{_metric_record_date(item)} {float(item['weight_kg'])} kg" for item in recent]
    lines.append(f"Recent weigh-ins (last {len(recent)}): {', '.join(recent_parts)}")

    if user:
        target_weight = str(user.get("target_weight") or "").strip()
        target_timeline = str(user.get("target_timeline") or "").strip()
        if target_weight:
            target_line = f"Target weight: {target_weight}"
            if target_timeline:
                target_line += f" ({target_timeline})"
            lines.append(target_line)

    return "\n".join(lines)


def build_metrics_block(latest: Optional[Dict[str, Any]]) -> str:
    if not latest:
        return "Weight / BMI / REE: unknown"
    return (
        f"Weight: {latest['weight_kg']} kg\n"
        f"BMI: {latest['bmi']} ({latest.get('bmi_label', '')})\n"
        f"REE: {latest['ree']} kcal/day"
    )


def build_coach_prompt(
    user: Dict[str, Any],
    latest: Optional[Dict[str, Any]],
    message: str,
    rag_context: str,
    memory_context: Dict[str, Any],
    weight_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    profile_summary = build_profile_summary(user, latest)
    metrics_block = build_metrics_block(latest)
    memory_text = memory_context.get("memory_text", "")
    memory_section = f"\n\n{memory_text}" if memory_text else ""
    trend_section = ""
    if weight_history is not None:
        trend_block = build_weight_trend_block(user, weight_history)
        trend_section = f"\n\n[Weight Trend]\n{trend_block}"
    return (
        "You are a professional, supportive, and safety-aware AI nutrition coach. "
        "Use the user profile, metrics, coaching memory, and knowledge context below. "
        "Memory summaries may be incomplete; allergies, medical conditions, and diet restrictions in the User Profile are authoritative. "
        "Structured weight history in [Weight Trend] is authoritative for progress questions. "
        "Do not give unsafe medical claims. If relevant, offer a next step such as generating a 7-day meal plan.\n"
        "Calorie-target questions are allowed as general wellness coaching when height, weight, REE, and activity/work pattern are available. "
        "Do not refuse ordinary calorie-target questions solely because they are personalized. "
        "For calorie targets, estimate maintenance from REE plus activity/work pattern first, then suggest a modest weight-loss target around 300-500 kcal/day below estimated maintenance. "
        "Do not recommend eating below REE as a general target, and do not give 1000 kcal/day as a normal starting goal; mention clinician/dietitian support for very-low-calorie or medically supervised plans only.\n"
        "Be warm but concise. "
        "Answer first with the concrete recommendation (numbers, types, or decisions when relevant). "
        "Then include a brief Why section with at most 2 short bullets. "
        "Do not show step-by-step reasoning, long analysis, or a thinking process before the answer. "
        "Avoid long personalized openers, disclaimers, or recap before the answer. "
        "Keep the full response under 5 short bullets unless the user asks for detail.\n\n"
        f"[User Profile]\n{profile_summary}\n\n"
        f"[Latest Metrics]\n{metrics_block}"
        f"{trend_section}"
        f"{memory_section}\n\n"
        f"[Knowledge Context — RAG]\n{rag_context or '(none)'}\n\n"
        f"[Current User Message]\n{message}"
    )


CALORIE_TARGET_INTENT_RE = re.compile(
    r"\b(?:calorie|calories|kcal|energy intake|eat per day|daily intake|maintenance|tdee)\b",
    re.IGNORECASE,
)


def detect_calorie_target_intent(message: str) -> bool:
    text = message or ""
    if not CALORIE_TARGET_INTENT_RE.search(text):
        return False
    return bool(re.search(r"\b(?:how many|aim|target|goal|should|recommend|need|per day|daily)\b", text, re.IGNORECASE))


def estimate_activity_factor(user: Dict[str, Any], message: str = "") -> Tuple[float, str]:
    text = " ".join(
        str(value or "")
        for value in (
            user.get("activity_level"),
            user.get("self_description"),
            user.get("coach_notes"),
            message,
        )
    ).lower()
    if re.search(r"\b(?:very active|manual|physical job|warehouse|construction|labou?r|on my feet all day|heavy work|athlete|hard training)\b", text):
        return 1.7, "very active/manual work"
    if re.search(r"\b(?:moderate|moderately|active job|standing|walking|server|nurse|retail|exercise regularly)\b", text):
        return 1.5, "moderately active"
    if re.search(r"\b(?:light|lightly|some walking|desk.*walk|office.*walk)\b", text):
        return 1.35, "lightly active"
    if re.search(r"\b(?:sedentary|desk|sitting|inactive|little exercise)\b", text):
        return 1.2, "sedentary"
    return 1.35, "lightly active estimate"


def _round_calorie(value: float) -> int:
    return int(round(float(value) / 50.0) * 50)


def _is_weight_loss_goal(user: Dict[str, Any], message: str) -> bool:
    text = " ".join(str(value or "") for value in (user.get("goal"), message)).lower()
    return bool(re.search(r"\b(?:lose|loss|fat loss|cut|slim|weight down|減重|瘦身)\b", text))


def build_calorie_target_reply(
    user: Dict[str, Any],
    latest: Optional[Dict[str, Any]],
    message: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not detect_calorie_target_intent(message):
        return None
    if not latest or latest.get("ree") is None:
        reply = (
            "Calorie goal:\n"
            "I need your current weight before estimating a daily calorie target.\n\n"
            "Why:\n"
            "- Your height and date of birth are not enough to estimate REE reliably.\n"
            "- Once weight is logged, I can estimate maintenance first and then suggest a safe starting range."
        )
        return reply, {"reason": "missing_latest_metrics"}

    ree = float(latest["ree"])
    factor, activity_label = estimate_activity_factor(user, message)
    maintenance = ree * factor
    weight_loss = _is_weight_loss_goal(user, message)
    gender = str(user.get("gender") or "").lower()
    general_floor = 1500 if gender == "male" else 1200
    safe_floor = max(ree, float(general_floor))

    if weight_loss:
        raw_low = maintenance - 500
        raw_high = maintenance - 300
        target_low = max(safe_floor, raw_low)
        target_high = max(target_low + 100, raw_high)
        target_high = min(target_high, maintenance)
        goal_line = f"Aim for about {_round_calorie(target_low)}-{_round_calorie(target_high)} kcal/day as a starting weight-loss range."
        why_deficit = "This uses a modest deficit from estimated maintenance without setting a target below your REE."
    else:
        target_low = maintenance - 100
        target_high = maintenance + 100
        goal_line = f"Aim for about {_round_calorie(target_low)}-{_round_calorie(target_high)} kcal/day to start, then adjust from your trend."
        why_deficit = "This is an estimated maintenance range, not an aggressive weight-loss target."

    reply = (
        "Calorie goal:\n"
        f"{goal_line}\n\n"
        "Why:\n"
        f"- Your REE is about {_round_calorie(ree)} kcal/day; with {activity_label}, estimated maintenance is about {_round_calorie(maintenance)} kcal/day.\n"
        f"- {why_deficit} Track weight, hunger, energy, and work performance for 2-4 weeks before adjusting."
    )
    meta = {
        "ree": _round_calorie(ree),
        "activity_factor": factor,
        "activity_label": activity_label,
        "estimated_maintenance": _round_calorie(maintenance),
        "target_low": _round_calorie(target_low),
        "target_high": _round_calorie(target_high),
        "weight_loss_target": weight_loss,
    }
    return reply, meta


FOOD_CHOICE_MARKER_START = "<!--nutricoach-food-choice"
FOOD_CHOICE_MARKER_END = "-->"
FOOD_CHOICE_COMPARISON_DIMENSIONS = ("protein", "carbs", "sodium", "glycemic")

FOOD_CHOICE_COMPARISON_SIGNALS = (
    r"\bvs\.?\b",
    r"\bversus\b",
    r"which is better",
    r"which one (?:is|should|would)",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"better choice",
    r"healthier\b.*\bor\b",
    r"\bor\b.*\bhealthier\b",
)

FOOD_CHOICE_DINING_SIGNALS = (
    r"\btakeaway\b",
    r"\btake away\b",
    r"\btake-out\b",
    r"\btake out\b",
    r"\bdining out\b",
    r"\beat out\b",
    r"\brestaurant\b",
    r"\bfast food\b",
    r"\bdelivery\b",
    r"\bpizza\b",
    r"\bburger\b",
    r"\bchinese\b",
    r"\bsushi\b",
    r"\bchipotle\b",
    r"\bmeal option",
    r"\bfood choice",
    r"\bmenu\b",
)


def get_food_choice_num_predict() -> int:
    return _env_int("FOOD_CHOICE_NUM_PREDICT", 1536)


def get_food_choice_temperature() -> float:
    try:
        return float(os.getenv("FOOD_CHOICE_TEMPERATURE", "0.1"))
    except (TypeError, ValueError):
        return 0.1


def detect_food_choice_intent(message: str) -> bool:
    low = (message or "").lower().strip()
    if not low:
        return False
    has_comparison = any(re.search(pattern, low) for pattern in FOOD_CHOICE_COMPARISON_SIGNALS)
    has_dining = any(re.search(pattern, low) for pattern in FOOD_CHOICE_DINING_SIGNALS)
    has_or_with_dining = bool(re.search(r"\bor\b", low)) and has_dining
    choosing = bool(re.search(r"\b(option|choice|pick|decide|between)\b", low))
    return has_comparison or has_or_with_dining or (has_dining and choosing)


def build_food_choice_requirement_lines(user: Dict[str, Any]) -> str:
    lines = [
        "- Compare typical restaurant or takeaway portions, not idealised home-cooked versions.",
        "- Keep advice general wellness coaching — not medical prescriptions.",
        "- If allergies are listed, flag allergen risks and never recommend unsafe options.",
    ]
    allergies = user.get("allergies") or []
    if allergies:
        lines.append(f"- Strict allergies: {', '.join(allergies)} — mention cross-contact or hidden allergens when relevant.")
    if user_has_diabetes(user):
        lines.extend([
            "- Diabetes profile: emphasise glycemic impact, carb load, and portion control.",
            "- Favour lower-GI options and suggest limiting refined carbs or sugary sauces.",
        ])
    diet = normalize_diet_preference(user.get("diet_preference"))
    if diet == "high_protein":
        lines.append("- High-protein profile: highlight which option delivers more lean protein per portion.")
    conditions = user.get("medical_conditions") or []
    if conditions:
        lines.append(f"- Medical conditions to respect: {', '.join(conditions)}.")
    return "\n".join(lines)


def build_food_choice_prompt(
    user: Dict[str, Any],
    latest: Optional[Dict[str, Any]],
    message: str,
    rag_context: str,
    memory_context: Dict[str, Any],
    weight_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    profile_summary = build_profile_summary(user, latest)
    metrics_block = build_metrics_block(latest)
    memory_text = memory_context.get("memory_text", "")
    memory_section = f"\n\n{memory_text}" if memory_text else ""
    trend_section = ""
    if weight_history is not None:
        trend_block = build_weight_trend_block(user, weight_history)
        trend_section = f"\n\n[Weight Trend]\n{trend_block}"
    requirements = build_food_choice_requirement_lines(user)
    dimensions = ", ".join(FOOD_CHOICE_COMPARISON_DIMENSIONS)
    return (
        "You are a professional, supportive AI nutrition coach helping someone choose between two dining-out or takeaway meal options.\n"
        "Use the user profile, metrics, coaching memory, and knowledge context below.\n"
        "Memory summaries may be incomplete; allergies, medical conditions, and diet restrictions in the User Profile are authoritative.\n"
        "Respond with ONLY valid JSON (no markdown fences, no commentary) using this schema:\n"
        "{\n"
        '  "option_a": "short label for first option",\n'
        '  "option_b": "short label for second option",\n'
        '  "comparison": {\n'
        '    "protein": {"option_a": "brief note", "option_b": "brief note"},\n'
        '    "carbs": {"option_a": "brief note", "option_b": "brief note"},\n'
        '    "sodium": {"option_a": "brief note", "option_b": "brief note"},\n'
        '    "glycemic": {"option_a": "brief note", "option_b": "brief note"}\n'
        "  },\n"
        '  "recommendation": "1-2 sentences naming the better fit for this user and why",\n'
        '  "portion_tip": "practical portion guidance for the recommended option",\n'
        '  "swap_suggestion": "one concrete swap to improve the less-favoured option",\n'
        '  "profile_notes": ["optional bullet about allergy, diabetes, or diet preference"]\n'
        "}\n"
        f"Cover these comparison dimensions: {dimensions}.\n"
        f"[Profile-aware rules]\n{requirements}\n\n"
        f"[User Profile]\n{profile_summary}\n\n"
        f"[Latest Metrics]\n{metrics_block}"
        f"{trend_section}"
        f"{memory_section}\n\n"
        f"[Knowledge Context — RAG]\n{rag_context or '(none)'}\n\n"
        f"[Current User Message]\n{message}"
    )


def _normalize_food_choice_dimension_block(raw: Any) -> Dict[str, str]:
    block = raw if isinstance(raw, dict) else {}
    return {
        "option_a": str(block.get("option_a") or "").strip(),
        "option_b": str(block.get("option_b") or "").strip(),
    }


def extract_compared_options_from_message(message: str) -> Tuple[str, str]:
    """Best-effort parse of two meal labels from a comparison question."""
    text = re.sub(r"\s+", " ", (message or "").strip())
    if not text:
        return "", ""
    patterns = (
        r"(?i)compare\s+takeaway\s+options\.\s*(.+?)\s+vs\.?\s+(.+?)(?:\.\s*which|\?|$)",
        r"(?i)(?:options?|between)\s+(.+?)\s+(?:vs\.?|versus|and|or)\s+(.+?)(?:\.\s*which|\?|$)",
        r"(?i)(.+?)\s+vs\.?\s+(.+?)(?:\.\s*which|\?|$)",
        r"(?i)(.+?)\s+versus\s+(.+?)(?:\.\s*which|\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        left = re.sub(r"^(?:help me compare takeaway options\.\s*)", "", match.group(1).strip(), flags=re.I)
        right = match.group(2).strip()
        right = re.sub(r"\.\s*which is better.*$", "", right, flags=re.I).strip(" .?")
        left = left.strip(" .?")
        if left and right and left.lower() != right.lower():
            return left[:80], right[:80]
    return "", ""


def food_choice_is_usable(comparison: Optional[Dict[str, Any]]) -> bool:
    """Require all comparison dimensions plus a recommendation before showing a card."""
    if not isinstance(comparison, dict) or not comparison:
        return False
    option_a = str(comparison.get("option_a") or "").strip()
    option_b = str(comparison.get("option_b") or "").strip()
    if not option_a or not option_b:
        return False
    if option_a.lower() in {"option a", "option_a"} and option_b.lower() in {"option b", "option_b"}:
        return False
    dims_filled = count_food_choice_dims_filled(comparison)
    has_recommendation = bool(str(comparison.get("recommendation") or "").strip())
    return dims_filled >= len(FOOD_CHOICE_COMPARISON_DIMENSIONS) and has_recommendation


def normalize_food_choice(
    data: Dict[str, Any],
    user: Dict[str, Any],
    message: str = "",
) -> Dict[str, Any]:
    comparison_raw = data.get("comparison") if isinstance(data.get("comparison"), dict) else {}
    comparison = {
        dimension: _normalize_food_choice_dimension_block(comparison_raw.get(dimension))
        for dimension in FOOD_CHOICE_COMPARISON_DIMENSIONS
    }
    profile_notes = [
        str(item).strip()
        for item in (data.get("profile_notes") or [])
        if str(item).strip()
    ]
    allergies = user.get("allergies") or []
    if allergies and not any("allerg" in note.lower() for note in profile_notes):
        profile_notes.append(f"Check ingredients for allergens: {', '.join(allergies)}.")
    if user_has_diabetes(user) and not any("diabet" in note.lower() or "glycemic" in note.lower() for note in profile_notes):
        profile_notes.append("Diabetes profile: favour the option with steadier blood-sugar impact and smaller refined-carb portions.")
    guessed_a, guessed_b = extract_compared_options_from_message(message)
    option_a = str(data.get("option_a") or "").strip() or guessed_a or "Option A"
    option_b = str(data.get("option_b") or "").strip() or guessed_b or "Option B"
    return {
        "option_a": option_a,
        "option_b": option_b,
        "comparison": comparison,
        "recommendation": str(data.get("recommendation") or "").strip(),
        "portion_tip": str(data.get("portion_tip") or "").strip(),
        "swap_suggestion": str(data.get("swap_suggestion") or "").strip(),
        "profile_notes": profile_notes,
    }


FOOD_CHOICE_ALLERGEN_EXTRA = {
    "shellfish": ("sushi", "poke", "sashimi", "calamari", "mussels", "oyster"),
    "fish": ("sushi", "poke", "sashimi"),
}


def count_food_choice_dims_filled(comparison: Dict[str, Any]) -> int:
    comp = comparison.get("comparison") if isinstance(comparison.get("comparison"), dict) else {}
    filled = 0
    for dimension in FOOD_CHOICE_COMPARISON_DIMENSIONS:
        block = comp.get(dimension) if isinstance(comp.get(dimension), dict) else {}
        if str(block.get("option_a") or "").strip() and str(block.get("option_b") or "").strip():
            filled += 1
    return filled


def _food_choice_text_has_allergen(text: str, allergen: str) -> bool:
    if _contains_allergen(text, allergen):
        return True
    allergen_low = str(allergen).lower().strip()
    extras = FOOD_CHOICE_ALLERGEN_EXTRA.get(allergen_low, ())
    low = (text or "").lower()
    for token in extras:
        for match in re.finditer(re.escape(token), low):
            prefix = low[max(0, match.start() - 24):match.start()]
            suffix = low[match.end():match.end() + 40]
            if re.search(r"\b(?:no|without|avoid|excluding|free from)\s*$", prefix):
                continue
            if re.search(r"^\s+(?:bowl|roll)?\s+is\s+not\s+(?:allowed|recommended)", suffix):
                continue
            return True
    return False


def _food_choice_recommendation_allergy_safe(comparison: Dict[str, Any], user: Dict[str, Any]) -> bool:
    allergies = user.get("allergies") or []
    if not allergies:
        return True
    recommendation = str(comparison.get("recommendation") or "")
    option_a = str(comparison.get("option_a") or "")
    option_b = str(comparison.get("option_b") or "")
    rec_low = recommendation.lower()
    caution_patterns = (
        r"\bavoid\b",
        r"\bnot\s+recommend",
        r"\bdo\s+not\b",
        r"\bshould\s+not\b",
        r"\bcaution\b",
        r"\bwarn\b",
        r"\ballerg",
        r"\bunsafe\b",
        r"\brisk\b",
        r"\bcheck\s+ingredients\b",
    )
    for allergen in allergies:
        if not _food_choice_text_has_allergen(recommendation, str(allergen)):
            favored_option = ""
            if option_a and option_a.lower() in rec_low:
                favored_option = option_a
            elif option_b and option_b.lower() in rec_low:
                favored_option = option_b
            if favored_option and _food_choice_text_has_allergen(favored_option, str(allergen)):
                if not any(re.search(pattern, rec_low) for pattern in caution_patterns):
                    return False
            continue
        if any(re.search(pattern, rec_low) for pattern in caution_patterns):
            continue
        return False
    return True


def validate_food_choice(comparison: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    dims_filled = count_food_choice_dims_filled(comparison)
    for dimension in FOOD_CHOICE_COMPARISON_DIMENSIONS:
        block = (comparison.get("comparison") or {}).get(dimension) or {}
        if not str(block.get("option_a") or "").strip() or not str(block.get("option_b") or "").strip():
            issues.append(f"empty_dimension:{dimension}")

    recommendation = str(comparison.get("recommendation") or "").strip()
    portion_tip = str(comparison.get("portion_tip") or "").strip()
    profile_notes = comparison.get("profile_notes") or []
    if not recommendation:
        issues.append("missing_recommendation")
    if not portion_tip:
        issues.append("missing_portion_tip")

    allergies = user.get("allergies") or []
    profile_notes_text = " ".join(str(note) for note in profile_notes).lower()
    allergy_note_present = not allergies or any("allerg" in profile_notes_text for note in profile_notes)
    if allergies and not allergy_note_present:
        issues.append("missing_allergy_note")

    diabetes_signal = False
    if user_has_diabetes(user):
        comp = comparison.get("comparison") if isinstance(comparison.get("comparison"), dict) else {}
        glycemic = comp.get("glycemic") if isinstance(comp.get("glycemic"), dict) else {}
        glycemic_text = f"{glycemic.get('option_a', '')} {glycemic.get('option_b', '')}".lower()
        diabetes_signal = bool(glycemic_text.strip()) or any(
            "diabet" in str(note).lower() or "glycemic" in str(note).lower() for note in profile_notes
        )
        if not diabetes_signal:
            issues.append("missing_diabetes_signal")
    else:
        diabetes_signal = True

    allergy_safe = _food_choice_recommendation_allergy_safe(comparison, user)
    if not allergy_safe:
        issues.append("allergy_unsafe_recommendation")

    scores = {
        "dims_filled": dims_filled,
        "has_recommendation": bool(recommendation),
        "has_portion_tip": bool(portion_tip),
        "profile_notes_present": bool(profile_notes),
        "allergy_note_present": allergy_note_present,
        "diabetes_signal": diabetes_signal,
        "allergy_safe": allergy_safe,
    }
    return {
        "valid": not issues,
        "issues": issues,
        "scores": scores,
    }


def format_food_choice_reply(comparison: Dict[str, Any]) -> str:
    lines = [
        f"**{comparison['option_a']}** vs **{comparison['option_b']}**",
        "",
        comparison.get("recommendation") or "Here is a balanced comparison for your profile.",
    ]
    if comparison.get("portion_tip"):
        lines.extend(["", f"**Portion tip:** {comparison['portion_tip']}"])
    if comparison.get("swap_suggestion"):
        lines.append(f"**Swap idea:** {comparison['swap_suggestion']}")
    for note in comparison.get("profile_notes") or []:
        lines.append(f"**Profile note:** {note}")
    return "\n".join(lines).strip()


def embed_food_choice_payload(reply_text: str, comparison: Dict[str, Any]) -> str:
    payload = json.dumps(comparison, ensure_ascii=False)
    return f"{reply_text.rstrip()}\n\n{FOOD_CHOICE_MARKER_START}\n{payload}\n{FOOD_CHOICE_MARKER_END}"


def extract_food_choice_from_content(content: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    text = content or ""
    start = text.find(FOOD_CHOICE_MARKER_START)
    if start == -1:
        return text, None
    end = text.find(FOOD_CHOICE_MARKER_END, start)
    if end == -1:
        return text, None
    payload_raw = text[start + len(FOOD_CHOICE_MARKER_START):end].strip()
    visible = text[:start].rstrip()
    try:
        parsed = json.loads(payload_raw)
    except json.JSONDecodeError:
        return visible or text, None
    if not isinstance(parsed, dict):
        return visible or text, None
    return visible, parsed


def _invoke_food_choice_llm(prompt: str) -> str:
    llm = create_ollama_llm(
        num_predict=get_food_choice_num_predict(),
        temperature=get_food_choice_temperature(),
        reasoning=False,
    )
    return strip_think_tags(llm.invoke(prompt))


def _parse_food_choice_llm_json(raw: str) -> Dict[str, Any]:
    filtered = safety_filter(raw)
    if filtered == SAFETY_BLOCK_REPLY:
        return {"__safety_blocked__": True}
    try:
        parsed = json.loads(_extract_json_object(filtered))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_food_choice_repair_prompt(
    user: Dict[str, Any],
    latest: Optional[Dict[str, Any]],
    message: str,
    issues: List[str],
    option_a: str = "",
    option_b: str = "",
) -> str:
    profile_summary = build_profile_summary(user, latest)
    guessed_a, guessed_b = extract_compared_options_from_message(message)
    label_a = option_a or guessed_a or "first meal"
    label_b = option_b or guessed_b or "second meal"
    issue_text = "\n".join(f"- {item}" for item in (issues or ["incomplete or invalid JSON"]))
    return (
        "You are a nutrition coach. Your previous JSON meal comparison was invalid or incomplete.\n"
        "Return ONLY valid JSON (no markdown fences, no commentary) with this exact schema:\n"
        "{\n"
        f'  "option_a": "{label_a}",\n'
        f'  "option_b": "{label_b}",\n'
        '  "comparison": {\n'
        '    "protein": {"option_a": "brief note", "option_b": "brief note"},\n'
        '    "carbs": {"option_a": "brief note", "option_b": "brief note"},\n'
        '    "sodium": {"option_a": "brief note", "option_b": "brief note"},\n'
        '    "glycemic": {"option_a": "brief note", "option_b": "brief note"}\n'
        "  },\n"
        '  "recommendation": "1-2 sentences naming the better fit and why",\n'
        '  "portion_tip": "practical portion guidance",\n'
        '  "swap_suggestion": "one concrete swap",\n'
        '  "profile_notes": ["optional allergy or diet note"]\n'
        "}\n"
        "Fill ALL four comparison dimensions with non-empty notes for both options.\n"
        f"Issues to fix:\n{issue_text}\n\n"
        f"[User Profile]\n{profile_summary}\n\n"
        f"[Current User Message]\n{message}"
    )


def run_food_choice_comparison(
    turn_context: Dict[str, Any],
    user: Dict[str, Any],
    latest: Optional[Dict[str, Any]],
    rag_store=None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    weight_history = get_weight_history(int(turn_context["user_id"]))
    rag_context, sources = retrieve_rag_context(rag_store, turn_context["message"])
    if sources:
        turn_context["sources"] = sources
    message = turn_context["message"]
    prompt = build_food_choice_prompt(
        user,
        latest,
        message,
        rag_context,
        turn_context["memory_context"],
        weight_history=weight_history,
    )

    last_issues: List[str] = ["invalid_or_empty_json"]
    comparison: Optional[Dict[str, Any]] = None
    for attempt in range(1, 4):
        attempt_prompt = prompt
        if attempt > 1:
            option_a = str((comparison or {}).get("option_a") or "")
            option_b = str((comparison or {}).get("option_b") or "")
            attempt_prompt = build_food_choice_repair_prompt(
                user,
                latest,
                message,
                last_issues,
                option_a=option_a,
                option_b=option_b,
            )
        try:
            raw = _invoke_food_choice_llm(attempt_prompt)
        except Exception as exc:
            last_issues = [f"llm_error:{exc}"]
            continue
        parsed = _parse_food_choice_llm_json(raw)
        if parsed.get("__safety_blocked__"):
            return SAFETY_BLOCK_REPLY, None
        candidate = normalize_food_choice(parsed, user, message=message)
        validation = validate_food_choice(candidate, user)
        # Require all four dimensions + recommendation; incomplete cards trigger repair retries.
        if food_choice_is_usable(candidate):
            reply_text = format_food_choice_reply(candidate)
            stored_reply = embed_food_choice_payload(reply_text, candidate)
            return stored_reply, candidate
        comparison = candidate
        last_issues = list(validation.get("issues") or ["unusable_comparison"])
        if count_food_choice_dims_filled(candidate) < len(FOOD_CHOICE_COMPARISON_DIMENSIONS):
            last_issues = list(dict.fromkeys([*last_issues, "incomplete_dimensions_need_all_four"]))
        if not parsed:
            last_issues = ["invalid_or_empty_json"]

    fallback = (
        "I couldn't build a reliable structured comparison just now "
        "(the model returned incomplete data). Please try again in a moment — "
        "for example: \"pizza vs salad\" or use Compare takeaway options."
    )
    return fallback, None


def build_eval_qa_prompt(memory_text: str, question: str) -> str:
    """Neutral factual-QA prompt for memory evals (no nutrition-coach domain guard)."""
    memory_block = (memory_text or "").strip() or "(no conversation memory)"
    return (
        "You are a careful factual question-answering assistant.\n"
        "Answer ONLY using the conversation memory below.\n"
        "If the memory contains the information needed to answer, you MUST answer with that fact. "
        "Do not say you do not know when the answer is present in the memory.\n"
        "Say you do not know ONLY when the memory truly lacks the needed fact.\n"
        "Do not refuse the question as out-of-domain.\n"
        "Do not give nutrition coaching, wellness advice, or medical guidance.\n"
        "Session messages may include headers like [Session date/time: ...]. "
        "When the dialogue uses relative time (yesterday, last week, this month), "
        "resolve absolute dates using those session date/time headers when possible.\n"
        "Prefer concrete dates, names, and short phrases that match the question.\n"
        "Keep the answer concise (one short sentence or phrase). Reply in English only.\n\n"
        f"[Conversation Memory]\n{memory_block}\n\n"
        f"[Question]\n{question}"
    )


def process_eval_qa_message(
    user_id: int,
    question: str,
    *,
    memory_mode: Optional[str] = None,
    match_chars: Optional[int] = None,
    force_new_session: bool = True,
    persist: bool = False,
) -> Dict[str, Any]:
    """Eval-only QA turn: inject memory_mode context with a neutral prompt (RAG off)."""
    user = get_user_profile(user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")

    session_id, closed_session_id = resolve_session(user_id, force_new=force_new_session)
    memory_context = build_memory_context(
        user_id,
        session_id,
        memory_mode=memory_mode,
        query=question,
        match_chars=match_chars,
    )
    prompt = build_eval_qa_prompt(memory_context.get("memory_text") or "", question)

    assert_ollama_ready()
    if persist:
        save_chat(user_id, "user", question, session_id=session_id)
    llm = create_ollama_llm()
    raw_reply = strip_think_tags(llm.invoke(prompt))
    final_reply = safety_filter(raw_reply)
    if persist:
        save_chat(user_id, "assistant", final_reply, session_id=session_id)
    return {
        "reply": final_reply,
        "session_id": session_id,
        "closed_session_id": closed_session_id,
        "memory_used": memory_context.get("memory_used") or {},
        "memory_text": memory_context.get("memory_text") or "",
        "memory_chars": len(memory_context.get("memory_text") or ""),
        "safety_blocked": False,
        "sources": [],
        "eval_neutral_prompt": True,
        "persisted_eval_turn": bool(persist),
    }


def prepare_chat_turn(
    user_id: int,
    message: str,
    rag_store=None,
    force_new_session: bool = False,
    memory_mode: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    user = get_user_profile(user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")

    session_id, closed_session_id = resolve_session(user_id, force_new=force_new_session)
    latest = get_latest_metrics_bundle(user_id)
    memory_context = build_memory_context(user_id, session_id, memory_mode=memory_mode, query=message)

    blocked_reply = safety_check_input(message)
    if blocked_reply:
        return {
            "reply": blocked_reply,
            "session_id": session_id,
            "closed_session_id": closed_session_id,
            "memory_used": memory_context["memory_used"],
            "summarization_pending": closed_session_id is not None and _session_message_count(closed_session_id) >= 2,
            "ollama_reachable": check_ollama_reachable(),
            "llm_degraded": False,
            "safety_blocked": True,
            "sources": [],
        }, None

    rag_context, sources = retrieve_rag_context(rag_store, message)
    weight_history = get_weight_history(user_id)
    prompt = build_coach_prompt(
        user, latest, message, rag_context, memory_context, weight_history=weight_history
    )
    turn_context = {
        "user_id": user_id,
        "message": message,
        "session_id": session_id,
        "closed_session_id": closed_session_id,
        "memory_context": memory_context,
        "sources": sources,
        "prompt": prompt,
    }
    return None, turn_context


def _chat_result_from_turn(
    turn_context: Dict[str, Any],
    reply: str,
    *,
    safety_blocked: bool = False,
    food_choice: Optional[Dict[str, Any]] = None,
    calorie_target: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    closed_session_id = turn_context["closed_session_id"]
    session_id = turn_context["session_id"]
    visible_reply, embedded_choice = extract_food_choice_from_content(reply)
    result = {
        "reply": visible_reply or reply,
        "session_id": session_id,
        "closed_session_id": closed_session_id,
        "memory_used": turn_context["memory_context"]["memory_used"],
        "summarization_pending": closed_session_id is not None and _session_message_count(closed_session_id) >= 2,
        "ollama_reachable": True,
        "llm_degraded": False,
        "safety_blocked": safety_blocked,
        "sources": turn_context.get("sources") or [],
    }
    choice = food_choice if food_choice_is_usable(food_choice) else None
    if choice is None and food_choice_is_usable(embedded_choice):
        choice = embedded_choice
    if choice:
        result["food_choice"] = choice
    if calorie_target:
        result["calorie_target"] = calorie_target
    return result


def process_chat_message(
    user_id: int,
    message: str,
    rag_store=None,
    force_new_session: bool = False,
    memory_mode: Optional[str] = None,
) -> Dict[str, Any]:
    blocked, turn_context = prepare_chat_turn(
        user_id,
        message,
        rag_store=rag_store,
        force_new_session=force_new_session,
        memory_mode=memory_mode,
    )
    if blocked:
        save_chat(user_id, "user", message, session_id=blocked["session_id"])
        save_chat(user_id, "assistant", blocked["reply"], session_id=blocked["session_id"])
        return blocked

    user = get_user_profile(user_id)
    latest = get_latest_metrics_bundle(user_id)

    calorie_result = build_calorie_target_reply(user, latest, message)
    if calorie_result:
        final_reply, calorie_target = calorie_result
        save_chat(user_id, "user", message, session_id=turn_context["session_id"])
        save_chat(user_id, "assistant", final_reply, session_id=turn_context["session_id"])
        return _chat_result_from_turn(turn_context, final_reply, calorie_target=calorie_target)

    if detect_food_choice_intent(message):
        assert_ollama_ready()
        try:
            stored_reply, food_choice = run_food_choice_comparison(turn_context, user, latest, rag_store=rag_store)
        except Exception as exc:
            raise OllamaUnavailableError(
                f"Local AI engine (Ollama) failed during food-choice comparison: {exc}"
            ) from exc
        final_reply = safety_filter(stored_reply)
        if not (final_reply or "").strip():
            raise EmptyLLMReplyError(
                "Local AI engine returned an empty food-choice reply. Please try again."
            )
        save_chat(user_id, "user", message, session_id=turn_context["session_id"])
        save_chat(user_id, "assistant", final_reply, session_id=turn_context["session_id"])
        return _chat_result_from_turn(turn_context, final_reply, food_choice=food_choice or None)

    reply = invoke_llm_visible_reply(turn_context["prompt"])
    final_reply = safety_filter(reply)
    if not (final_reply or "").strip():
        raise EmptyLLMReplyError(
            "Local AI engine returned an empty reply after safety filtering. Please try again."
        )
    save_chat(user_id, "user", message, session_id=turn_context["session_id"])
    save_chat(user_id, "assistant", final_reply, session_id=turn_context["session_id"])
    return _chat_result_from_turn(turn_context, final_reply)


def iter_chat_sse_events(
    user_id: int,
    message: str,
    rag_store=None,
    force_new_session: bool = False,
    memory_mode: Optional[str] = None,
):
    blocked, turn_context = prepare_chat_turn(
        user_id,
        message,
        rag_store=rag_store,
        force_new_session=force_new_session,
        memory_mode=memory_mode,
    )
    if blocked:
        save_chat(user_id, "user", message, session_id=blocked["session_id"])
        save_chat(user_id, "assistant", blocked["reply"], session_id=blocked["session_id"])
        yield {"event": "meta", "data": {
            "session_id": blocked["session_id"],
            "sources": blocked.get("sources") or [],
            "memory_used": blocked.get("memory_used"),
            "safety_blocked": True,
        }}
        yield {"event": "token", "data": {"text": blocked["reply"]}}
        yield {"event": "done", "data": blocked}
        return

    user = get_user_profile(user_id)
    latest = get_latest_metrics_bundle(user_id)

    calorie_result = build_calorie_target_reply(user, latest, message)
    if calorie_result:
        final_reply, calorie_target = calorie_result
        save_chat(user_id, "user", message, session_id=turn_context["session_id"])
        yield {"event": "meta", "data": {
            "session_id": turn_context["session_id"],
            "sources": turn_context.get("sources") or [],
            "memory_used": turn_context["memory_context"]["memory_used"],
            "safety_blocked": False,
            "calorie_target": True,
        }}
        yield {"event": "token", "data": {"text": final_reply}}
        save_chat(user_id, "assistant", final_reply, session_id=turn_context["session_id"])
        yield {"event": "done", "data": _chat_result_from_turn(turn_context, final_reply, calorie_target=calorie_target)}
        return

    if detect_food_choice_intent(message):
        assert_ollama_ready()
        save_chat(user_id, "user", message, session_id=turn_context["session_id"])
        yield {"event": "meta", "data": {
            "session_id": turn_context["session_id"],
            "sources": turn_context.get("sources") or [],
            "memory_used": turn_context["memory_context"]["memory_used"],
            "safety_blocked": False,
            "food_choice": True,
        }}
        try:
            stored_reply, food_choice = run_food_choice_comparison(turn_context, user, latest, rag_store=rag_store)
        except Exception as exc:
            yield {"event": "error", "data": {"detail": f"Local AI engine (Ollama) failed during food-choice comparison: {exc}"}}
            return
        final_reply = safety_filter(stored_reply)
        if not (final_reply or "").strip():
            yield {
                "event": "error",
                "data": {
                    "detail": "Local AI engine returned an empty food-choice reply. Please try again."
                },
            }
            return
        visible_reply, _ = extract_food_choice_from_content(final_reply)
        yield {"event": "token", "data": {"text": visible_reply or final_reply}}
        save_chat(user_id, "assistant", final_reply, session_id=turn_context["session_id"])
        yield {"event": "done", "data": _chat_result_from_turn(turn_context, final_reply, food_choice=food_choice or None)}
        return

    assert_ollama_ready()
    save_chat(user_id, "user", message, session_id=turn_context["session_id"])
    yield {"event": "meta", "data": {
        "session_id": turn_context["session_id"],
        "sources": turn_context.get("sources") or [],
        "memory_used": turn_context["memory_context"]["memory_used"],
        "safety_blocked": False,
    }}

    accumulated: List[str] = []
    visible_any = False
    try:
        llm = create_ollama_llm()
        stream_filter = ThinkTagStreamFilter()
        for chunk in llm.stream(turn_context["prompt"]):
            if isinstance(chunk, str):
                text = chunk
            else:
                text = getattr(chunk, "text", None) or getattr(chunk, "content", None) or str(chunk)
            if not text:
                continue
            accumulated.append(text)
            visible_text = stream_filter.feed(text)
            if visible_text:
                visible_any = True
                yield {"event": "token", "data": {"text": visible_text}}
        trailing_visible = stream_filter.finish()
        if trailing_visible:
            visible_any = True
            yield {"event": "token", "data": {"text": trailing_visible}}
    except Exception as exc:
        yield {"event": "error", "data": {"detail": f"Local AI engine (Ollama) failed during chat: {exc}"}}
        return

    raw_reply = "".join(accumulated)
    final_reply = safety_filter(strip_think_tags(raw_reply))
    if not (final_reply or "").strip():
        # Do not persist an empty assistant message. If nothing was streamed yet,
        # attempt one non-stream recovery invoke with a direct-answer hint.
        if not visible_any:
            try:
                recovered = invoke_llm_visible_reply(
                    f"{turn_context['prompt']}{EMPTY_REPLY_RETRY_SUFFIX}",
                    max_attempts=1,
                )
                final_reply = safety_filter(recovered)
            except OllamaUnavailableError as exc:
                yield {"event": "error", "data": {"detail": str(exc)}}
                return
            if (final_reply or "").strip():
                yield {"event": "token", "data": {"text": final_reply}}
                save_chat(user_id, "assistant", final_reply, session_id=turn_context["session_id"])
                yield {"event": "done", "data": _chat_result_from_turn(turn_context, final_reply)}
                return
        yield {
            "event": "error",
            "data": {
                "detail": (
                    "Local AI engine returned an empty reply after stripping thinking content. "
                    "Please try again."
                )
            },
        }
        return

    save_chat(user_id, "assistant", final_reply, session_id=turn_context["session_id"])
    yield {"event": "done", "data": _chat_result_from_turn(turn_context, final_reply)}


def close_user_session(user_id: int) -> Dict[str, Any]:
    active = get_active_session(user_id)
    if not active:
        return {"status": "no_active_session", "session_id": None, "summarization_pending": False}
    session_id = int(active["session_id"])
    closed = close_session(session_id, user_id)
    return {
        "status": "closed",
        "session_id": session_id,
        "summarization_pending": closed is not None,
    }


def initialize_rag():
    if not RAG_DEPS_AVAILABLE or not os.path.isdir(PDF_DIR):
        return None
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    try:
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        vector_store = Chroma(collection_name="health_knowledge", persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
        existing_count = vector_store._collection.count() if hasattr(vector_store, "_collection") else 0
        if existing_count == 0:
            loader = DirectoryLoader(PDF_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False)
            docs = loader.load()
            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(docs)
                vector_store.add_documents(chunks)
        return vector_store
    except Exception:
        return None


INPUT_UNSAFE_PATTERNS = [
    r"\bhow\s+(?:do|can|to)\s+i\b.{0,80}\b(?:purge|make myself vomit|vomit after meals)\b",
    r"\b(?:help me|teach me|show me|give me|make me|i want to|i plan to|can i|should i)\b.{0,80}\b(?:purge|purging|make myself vomit|vomit after meals)\b",
    r"\b(?:help me|teach me|show me|give me|make me|i want to|i plan to|can i|should i)\b.{0,80}\b(?:starve|stop eating|eat nothing|water fast|dry fast|extreme fast)\b",
    r"\b(?:only|under|below|less than)\s+(?:500|800)\s*(?:kcal|calorie|calories)\b",
    r"\b(?:500|800)\s*(?:kcal|calorie|calories)\b.{0,80}\b(?:diet|fast|meal plan|plan|per day|a day|daily|only)\b",
    r"\b(?:use|take|abuse)\b.{0,80}\blaxatives?\b.{0,80}\b(?:lose weight|weight loss|slim|purge)\b",
    r"\b(?:self-harm|self harm)\b",
]

OUTPUT_UNSAFE_PATTERNS = [
    r"\b(?:you should|you can|try|start|use|take|do|follow|continue|aim for|limit yourself to|keep it to)\b.{0,100}\b(?:purge|purging|make yourself vomit|vomit after meals)\b",
    r"\b(?:you should|you can|try|start|do|follow|continue|aim for|limit yourself to|keep it to)\b.{0,100}\b(?:starve|stop eating|eat nothing|water fast|dry fast|extreme fast)\b",
    r"\b(?:you should|you can|try|start|use|take|do|follow|continue|aim for|limit yourself to|keep it to)\b.{0,100}\blaxatives?\b.{0,80}\b(?:lose weight|weight loss|slim|purge)\b",
    r"\b(?:only|under|below|less than|limit yourself to|keep it to)\s+(?:500|800)\s*(?:kcal|calorie|calories)\b",
    r"\b(?:500|800)\s*(?:kcal|calorie|calories)\b.{0,80}\b(?:diet|fast|meal plan|plan|per day|a day|daily|only)\b",
]

SAFETY_EDUCATION_RE = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|can|could|should|what|why|how)\b"
    r".{0,140}\b(?:safe|unsafe|dangerous|healthy|risk|risks|harmful|real|avoid|concern|concerns)\b"
)

SAFETY_BLOCK_REPLY = (
    "I can't support unsafe or extreme dieting requests. "
    "NutriCoachAI offers general wellness coaching only — not medical advice. "
    "Please speak with a doctor or registered dietitian for personalized, safe guidance."
)


def safety_check_input(message: str) -> Optional[str]:
    text = (message or "").lower()
    if SAFETY_EDUCATION_RE.search(text):
        return None
    if any(re.search(pattern, text) for pattern in INPUT_UNSAFE_PATTERNS):
        return SAFETY_BLOCK_REPLY
    return None


def safety_filter(response: str) -> str:
    text = (response or "").lower()
    if any(re.search(pattern, text) for pattern in OUTPUT_UNSAFE_PATTERNS):
        return SAFETY_BLOCK_REPLY
    return response


def retrieve_rag_context(rag_store, message: str, k: int = 2) -> Tuple[str, List[str]]:
    if rag_store is None:
        return "", []
    try:
        docs = rag_store.similarity_search(message, k=k)
        sources: List[str] = []
        seen = set()
        for doc in docs:
            raw = doc.metadata.get("source") or doc.metadata.get("file_path") or ""
            name = os.path.basename(str(raw)) if raw else ""
            if name and name not in seen:
                seen.add(name)
                sources.append(name)
        context = "\n\n".join(doc.page_content[:1200] for doc in docs)
        return context, sources
    except Exception:
        return "", []


def build_profile_summary(user: Dict[str, Any], latest: Optional[Dict[str, Any]]) -> str:
    lines = [
        f"Name: {user.get('name', 'Unknown')}",
        f"Gender: {user.get('gender', 'Unknown')}",
        f"Birth date: {user.get('birth_date', 'Unknown')}",
        f"Height: {user.get('height_cm', 'Unknown')} cm",
        f"Current weight: {latest['weight_kg']} kg" if latest else "Current weight: unknown",
        f"BMI: {latest['bmi']} ({latest['bmi_label']})" if latest else "BMI: unknown",
        f"REE: {latest['ree']} kcal/day" if latest else "REE: unknown",
    ]
    for field in PROFILE_TEXT_FIELDS:
        value = str(user.get(field, "") or "").strip()
        if value:
            lines.append(f"{field.replace('_', ' ').title()}: {value}")
    for field in PROFILE_LIST_FIELDS:
        values = user.get(field) or []
        if values:
            lines.append(f"{field.replace('_', ' ').title()}: {', '.join(values)}")
    return "\n".join(lines)


OFFLINE_MEAL_DAY_TEMPLATES: List[Dict[str, str]] = [
    {
        "breakfast": "Greek yogurt with oats, berries, and chia seeds",
        "lunch": "Grilled chicken breast, brown rice, and steamed broccoli",
        "dinner": "Baked tofu stir-fry with bell peppers and snap peas",
        "snack": "Apple slices with almond butter",
        "focus": "High protein, balanced carbs",
    },
    {
        "breakfast": "Vegetable omelette with whole-grain toast",
        "lunch": "Turkey and avocado salad with mixed greens",
        "dinner": "Baked cod with quinoa and roasted carrots",
        "snack": "Carrot sticks with hummus",
        "focus": "Lean protein variety",
    },
    {
        "breakfast": "Overnight oats with banana and walnuts",
        "lunch": "Lentil soup with side salad and olive oil dressing",
        "dinner": "Chicken fajita bowl with peppers, beans, and brown rice",
        "snack": "Low-sugar yogurt with cucumber",
        "focus": "Fiber-forward meals",
    },
    {
        "breakfast": "Smoothie with spinach, banana, soy milk, and flaxseed",
        "lunch": "Tuna salad wrap on whole-wheat tortilla (no shellfish if allergic)",
        "dinner": "Lean beef and vegetable stew with barley",
        "snack": "Handful of unsalted nuts",
        "focus": "Micronutrient diversity",
    },
    {
        "breakfast": "Cottage cheese with pineapple and pumpkin seeds",
        "lunch": "Chickpea Buddha bowl with tahini dressing",
        "dinner": "Grilled pork tenderloin with sweet potato and green beans",
        "snack": "Pear with a small cheese portion",
        "focus": "Plant and animal protein mix",
    },
    {
        "breakfast": "Whole-grain pancakes with berries (light syrup optional)",
        "lunch": "Egg salad on mixed greens with cherry tomatoes",
        "dinner": "Shrimp-free seafood alternative: white fish with asparagus and wild rice",
        "snack": "Rice cakes with peanut butter",
        "focus": "Steady energy meals",
    },
    {
        "breakfast": "Tofu scramble with mushrooms and whole-grain toast",
        "lunch": "Chicken noodle soup with extra vegetables",
        "dinner": "Stuffed bell peppers with lean ground turkey and quinoa",
        "snack": "Orange and a few olives",
        "focus": "Weekend prep-friendly options",
    },
]

ALLERGEN_ALTERNATIVES = {
    "shellfish": ("shrimp", "prawn", "crab", "lobster", "shellfish"),
    "fish": ("cod", "tuna", "salmon", "white fish", "fish"),
    "milk": ("milk", "yogurt", "cheese", "cottage cheese"),
    "dairy": ("milk", "yogurt", "cheese", "cottage cheese"),
    "peanut": ("peanut", "peanuts"),
    "peanuts": ("peanut", "peanuts"),
    "tree nut": ("almond", "walnut", "nuts", "tree nut"),
    "tree_nut": ("almond", "walnut", "nuts", "tree nut"),
    "tree_nuts": ("almond", "walnut", "nuts", "tree nut"),
    "egg": ("omelette", "egg salad", "egg", "eggs"),
    "eggs": ("omelette", "egg salad", "egg", "eggs"),
    "soy": ("soy", "tofu", "tempeh", "edamame"),
    "wheat": ("wheat", "gluten", "bread", "pasta"),
    "gluten": ("wheat", "gluten", "bread", "pasta"),
    "sesame": ("sesame", "tahini"),
}


ALLERGY_ALIASES = {
    "shellfish": "shellfish",
    "fish": "fish",
    "peanut": "peanut",
    "peanuts": "peanut",
    "tree_nut": "tree nut",
    "tree_nuts": "tree nut",
    "tree_nut_allergy": "tree nut",
    "egg": "egg",
    "eggs": "egg",
    "milk": "milk",
    "dairy": "dairy",
    "soy": "soy",
    "wheat": "wheat",
    "gluten": "wheat",
    "sesame": "sesame",
}


def normalize_allergy_list(value: Any) -> List[str]:
    if value is None:
        return []
    raw_items: List[str]
    if isinstance(value, list):
        raw_items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raw_items = [part.strip() for part in str(value).split(",") if part.strip()]
    normalized: List[str] = []
    seen = set()
    for item in raw_items:
        slug = re.sub(r"\s+", "_", item.strip().lower().replace("-", "_"))
        canonical = ALLERGY_ALIASES.get(slug, item.strip().lower())
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


VAGUE_MEAL_RE = re.compile(
    r"^(balanced|light|healthy|simple|basic|nutritious|wholesome)\s+(breakfast|lunch|dinner|snack|meal|day)\.?$",
    re.IGNORECASE,
)
LEFTOVER_MEAL_RE = re.compile(r"\b(left\s*-?\s*over|leftover|yesterday'?s?|previous day)\b", re.IGNORECASE)
PORTION_RE = re.compile(r"\d+\s*g\b", re.IGNORECASE)
GENERIC_FRUIT_RE = re.compile(r"\b(fruit|fruits)\b", re.IGNORECASE)
BANANA_RE = re.compile(r"\bbananas?\b", re.IGNORECASE)
MEAL_FIELDS = ("breakfast", "lunch", "dinner", "snack")
LOW_GI_FRUIT_HINTS = ("berry", "berries", "blueberr", "strawberr", "apple", "pear", "kiwi", "orange", "plum")


def _profile_text_blob(user: Dict[str, Any]) -> str:
    parts = [
        str(user.get("goal", "")),
        str(user.get("self_description", "")),
        str(user.get("coach_notes", "")),
        str(user.get("target_timeline", "")),
    ]
    parts.extend(str(item) for item in (user.get("medical_conditions") or []))
    return " ".join(parts).lower()


def user_has_diabetes(user: Dict[str, Any]) -> bool:
    blob = _profile_text_blob(user)
    return any(token in blob for token in ("diabetes", "diabetic", "type 2", "type2", "blood sugar", "glycemic", "a1c"))


def user_has_depression(user: Dict[str, Any]) -> bool:
    blob = _profile_text_blob(user)
    return any(token in blob for token in ("depression", "depressive", "low mood", "mental health"))


def user_has_inflammation_focus(user: Dict[str, Any]) -> bool:
    blob = _profile_text_blob(user)
    return any(
        token in blob
        for token in (
            "inflammation",
            "inflammatory",
            "anti-inflammatory",
            "chronic pain",
            "back pain",
            "disc herniation",
            "herniation",
            "lumbar",
        )
    )


def user_has_musculoskeletal_focus(user: Dict[str, Any]) -> bool:
    blob = _profile_text_blob(user)
    return any(token in blob for token in ("disc herniation", "herniation", "lumbar", "back pain", "spine"))


def user_is_older_adult(user: Dict[str, Any]) -> bool:
    birth_date = str(user.get("birth_date") or "").strip()
    if len(birth_date) != 8 or not birth_date.isdigit():
        return False
    try:
        born = datetime.strptime(birth_date, "%Y%m%d")
    except ValueError:
        return False
    age = (datetime.now() - born).days / 365.25
    return age >= 60


def _normalize_nutrition_block(block: Any) -> Dict[str, int]:
    if not isinstance(block, dict):
        return {}
    normalized: Dict[str, int] = {}
    for key in ("calories", "protein_g", "carbs_g", "fat_g"):
        value = block.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            normalized[key] = int(round(float(value)))
        except (TypeError, ValueError):
            continue
    return normalized


def _meal_quality_issues(day: Dict[str, Any], user: Dict[str, Any], banana_count: List[int]) -> List[str]:
    issues: List[str] = []
    day_no = day.get("day")
    for field in MEAL_FIELDS:
        text = str(day.get(field, "")).strip()
        if not text:
            issues.append(f"day {day_no} missing {field}")
            continue
        if VAGUE_MEAL_RE.match(text):
            issues.append(f"day {day_no} {field} is too vague: '{text}'")
        if LEFTOVER_MEAL_RE.search(text):
            issues.append(f"day {day_no} {field} must be a complete meal, not leftovers: '{text}'")
        if GENERIC_FRUIT_RE.search(text) and not any(hint in text.lower() for hint in LOW_GI_FRUIT_HINTS):
            issues.append(f"day {day_no} {field} must name a specific fruit, not generic 'fruit': '{text}'")

    portion_hits = sum(1 for field in MEAL_FIELDS if PORTION_RE.search(str(day.get(field, ""))))
    if portion_hits < 3:
        issues.append(f"day {day_no} needs gram portions in at least 3 of 4 meals")

    if user_has_diabetes(user):
        for field in MEAL_FIELDS:
            text = str(day.get(field, ""))
            if BANANA_RE.search(text):
                banana_count[0] += 1
                if banana_count[0] > 1:
                    issues.append("diabetes profile allows banana at most once across the week")
                if not re.search(r"\b(half|1/2|\d+\s*g)\b", text, re.IGNORECASE):
                    issues.append(f"day {day_no} {field} banana must include portion such as half banana or grams")

    return issues


def _plan_quality_issues(plan: Dict[str, Any], user: Dict[str, Any]) -> List[str]:
    days = plan.get("days") if isinstance(plan.get("days"), list) else []
    has_structured_nutrition = bool(_normalize_nutrition_block(plan.get("nutrition_targets"))) or any(
        _normalize_nutrition_block(day.get("daily_totals"))
        for day in days
        if isinstance(day, dict)
    )
    if not has_structured_nutrition:
        return []
    issues: List[str] = []
    banana_count = [0]
    fish_or_omega_days = 0
    leafy_days = 0
    mood_support_days = 0

    for day in days:
        if not isinstance(day, dict):
            continue
        issues.extend(_meal_quality_issues(day, user, banana_count))
        blob = _meal_text_blob(day).lower()
        if any(token in blob for token in ("salmon", "sardine", "cod", "mackerel", "trout", "fish", "omega")):
            fish_or_omega_days += 1
        if any(token in blob for token in ("spinach", "kale", "broccoli", "chard", "greens", "leafy")):
            leafy_days += 1
        if any(token in blob for token in ("salmon", "sardine", "walnut", "pumpkin seed", "oat", "leafy", "spinach", "kale", "egg")):
            mood_support_days += 1

    if user_has_inflammation_focus(user) and fish_or_omega_days < 4:
        issues.append("anti-inflammatory profile needs omega-3 rich fish on at least 4 days")
    if user_has_inflammation_focus(user) and leafy_days < 5:
        issues.append("anti-inflammatory profile needs leafy greens on at least 5 days")

    if user_has_depression(user) and mood_support_days < 5:
        issues.append("depression profile should include omega-3, folate, or mood-supporting foods on at least 5 days")

    if normalize_diet_preference(user.get("diet_preference")) == "high_protein":
        low_protein_days = 0
        for day in days:
            blob = _meal_text_blob(day).lower()
            if not any(token in blob for token in ("chicken", "turkey", "fish", "salmon", "cod", "egg", "yogurt", "tofu", "bean", "lentil", "cottage", "pork", "beef")):
                low_protein_days += 1
        if low_protein_days > 1:
            issues.append("high-protein profile needs a clear protein source every day")

    targets = _normalize_nutrition_block(plan.get("nutrition_targets"))
    if not targets.get("calories"):
        issues.append("nutrition_targets.calories is required")
    if not targets.get("protein_g"):
        issues.append("nutrition_targets.protein_g is required")

    days_with_totals = 0
    for day in days:
        totals = _normalize_nutrition_block(day.get("daily_totals"))
        if totals.get("calories") and totals.get("protein_g"):
            days_with_totals += 1
    if days_with_totals < 5:
        issues.append("at least 5 days need daily_totals with calories and protein_g")

    return issues


def build_meal_plan_condition_requirement_lines(user: Dict[str, Any]) -> str:
    lines = [
        "- Every meal must name specific foods with gram portions (example: chicken breast 120 g, brown rice 80 g, broccoli 150 g).",
        "- Never use vague placeholders such as 'Balanced breakfast' or 'Light snack'.",
        "- Never suggest leftovers, yesterday's meals, or 'leftover' wording.",
        "- Name specific fruits (berries, apple, pear, kiwi) instead of generic 'fruit'.",
        "- Include one swap/substitution idea in each day notes field.",
        "- Provide top-level nutrition_targets and per-day daily_totals with calories, protein_g, carbs_g, fat_g.",
    ]
    if user.get("allergies"):
        lines.append("- Treat allergies as strict: avoid allergen ingredients and obvious cross-contact foods.")
    if user_has_diabetes(user):
        lines.extend([
            "- Diabetes profile: favor low-GI carbs (oats, quinoa, brown rice, berries) and limit white bread or white rice.",
            "- Limit banana to at most one day per week and always specify portion (half banana or grams).",
            "- Spread carbohydrates across meals; avoid stacking multiple high-GI fruits in one day.",
        ])
    if normalize_diet_preference(user.get("diet_preference")) == "high_protein":
        lines.extend([
            "- High-protein profile: aim for about 25-30 g protein per main meal.",
            "- Prioritize eggs, Greek yogurt, fish, poultry, tofu, legumes, and lean dairy where allowed.",
        ])
    if user_is_older_adult(user):
        lines.append("- Older adult profile: emphasize protein at each meal to support muscle maintenance.")
    if user_has_inflammation_focus(user):
        lines.extend([
            "- Anti-inflammatory focus: include leafy greens daily and omega-3 rich fish at least 4 days per week.",
            "- Prefer fish or poultry over pork for inflammation goals.",
        ])
    if user_has_depression(user):
        lines.extend([
            "- Depression profile: include omega-3 fish, leafy greens, oats, pumpkin seeds, or eggs across the week.",
            "- Mention vitamin D or folate supporting foods when appropriate (fortified dairy, eggs, greens, fish).",
        ])
    if user_has_musculoskeletal_focus(user):
        lines.extend([
            "- Back/disc profile: support recovery with adequate protein, calcium-rich foods, and magnesium sources (leafy greens, seeds, legumes).",
            "- Keep notes practical for low-impact routines.",
        ])
    return "\n".join(f"{line}\n" for line in lines)


def _meal_text_blob(day: Dict[str, Any]) -> str:
    return " ".join(str(day.get(key, "")) for key in ("breakfast", "lunch", "dinner", "snack")).lower()


def _contains_allergen(text: str, allergen: str) -> bool:
    allergen_low = allergen.lower().strip()
    if not allergen_low:
        return False
    tokens = (allergen_low,) + ALLERGEN_ALTERNATIVES.get(allergen_low, ())
    for token in tokens:
        if not token:
            continue
        for match in re.finditer(re.escape(token), text, re.IGNORECASE):
            prefix = text[max(0, match.start() - 24):match.start()]
            suffix = text[match.end():match.end() + 40]
            if re.search(r"\b(?:no|without|avoid|excluding|free from)\s*$", prefix, re.IGNORECASE):
                continue
            if re.search(r"^\s+(?:stir-fry|salad)?\s+is\s+not\s+allowed", suffix, re.IGNORECASE):
                continue
            return True
    return False


def count_distinct_main_meals(days: List[Dict[str, Any]]) -> int:
    signatures = set()
    for day in days:
        signatures.add((str(day.get("lunch", "")).strip().lower(), str(day.get("dinner", "")).strip().lower()))
    return len(signatures)


def validate_meal_plan(plan: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    days = plan.get("days") if isinstance(plan.get("days"), list) else []
    issues: List[str] = []
    if len(days) != 7:
        issues.append(f"expected 7 days, got {len(days)}")
    distinct = count_distinct_main_meals(days)
    if distinct < 5:
        issues.append(f"only {distinct} distinct lunch/dinner pairs (need >= 5)")
    for day in days:
        blob = _meal_text_blob(day)
        for allergen in user.get("allergies") or []:
            if _contains_allergen(blob, str(allergen)):
                issues.append(f"allergen '{allergen}' may appear on day {day.get('day')}")
    issues.extend(_plan_quality_issues(plan, user))
    return {
        "valid": not issues,
        "issues": issues,
        "day_count": len(days),
        "distinct_main_meals": distinct,
    }


def _swap_allergens_in_text(text: str, allergies: List[str]) -> str:
    updated = text
    for allergen in allergies:
        allergen_low = str(allergen).lower()
        if allergen_low in ("shellfish", "fish"):
            updated = updated.replace("Shrimp-free seafood alternative: white fish", "Baked chicken breast")
            updated = updated.replace("Baked cod", "Baked chicken breast")
            updated = updated.replace("Tuna salad", "Chickpea salad")
            updated = updated.replace("tuna", "chickpea")
            updated = updated.replace("cod", "chicken")
            updated = updated.replace("salmon", "chicken")
            updated = updated.replace("shellfish", "poultry")
        if allergen_low in ("milk", "dairy"):
            updated = updated.replace("Greek yogurt", "soy yogurt")
            updated = updated.replace("yogurt", "soy yogurt")
            updated = updated.replace("cheese", "dairy-free cheese")
            updated = updated.replace("milk", "soy milk")
            updated = updated.replace("Cottage cheese", "Silken tofu")
        if allergen_low in ("peanut", "peanuts"):
            updated = updated.replace("peanut butter", "sunflower seed butter")
            updated = updated.replace("Peanut butter", "Sunflower seed butter")
            updated = updated.replace("peanuts", "pumpkin seeds")
        if allergen_low in ("tree nut", "tree_nuts"):
            updated = updated.replace("almond butter", "sunflower seed butter")
            updated = updated.replace("walnuts", "pumpkin seeds")
            updated = updated.replace("almond", "pumpkin seed")
    return updated


def build_offline_meal_plan(user: Dict[str, Any]) -> Dict[str, Any]:
    goal = str(user.get("goal") or "healthy eating").replace("_", " ")
    allergies = [str(item) for item in (user.get("allergies") or [])]
    days = []
    for index, template in enumerate(OFFLINE_MEAL_DAY_TEMPLATES, start=1):
        days.append({
            "day": index,
            "breakfast": _swap_allergens_in_text(template["breakfast"], allergies),
            "lunch": _swap_allergens_in_text(template["lunch"], allergies),
            "dinner": _swap_allergens_in_text(template["dinner"], allergies),
            "snack": _swap_allergens_in_text(template["snack"], allergies),
            "focus": template["focus"],
            "notes": f"Supports {goal}. Adjust portions for activity level and medical guidance.",
        })
    return {
        "summary": f"A varied 7-day meal plan for {user.get('name', 'the user')} (template fallback while the local model output was invalid).",
        "days": days,
    }


def normalize_meal_plan_days(plan: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    raw_days = plan.get("days") if isinstance(plan.get("days"), list) else []
    normalized: List[Dict[str, Any]] = []
    for index, day in enumerate(raw_days[:7], start=1):
        if not isinstance(day, dict):
            continue
        entry = {
            "day": index,
            "breakfast": str(day.get("breakfast", "")).strip(),
            "lunch": str(day.get("lunch", "")).strip(),
            "dinner": str(day.get("dinner", "")).strip(),
            "snack": str(day.get("snack", "")).strip(),
            "focus": str(day.get("focus", "")).strip() or "Balanced day",
            "notes": str(day.get("notes", "")).strip(),
        }
        daily_totals = _normalize_nutrition_block(day.get("daily_totals"))
        if daily_totals:
            entry["daily_totals"] = daily_totals
        normalized.append(entry)
    while len(normalized) < 7:
        filler = OFFLINE_MEAL_DAY_TEMPLATES[len(normalized) % len(OFFLINE_MEAL_DAY_TEMPLATES)]
        allergies = [str(item) for item in (user.get("allergies") or [])]
        normalized.append({
            "day": len(normalized) + 1,
            "breakfast": _swap_allergens_in_text(filler["breakfast"], allergies),
            "lunch": _swap_allergens_in_text(filler["lunch"], allergies),
            "dinner": _swap_allergens_in_text(filler["dinner"], allergies),
            "snack": _swap_allergens_in_text(filler["snack"], allergies),
            "focus": filler["focus"],
            "notes": "Template fallback meal while the local model output was invalid.",
        })
    summary = str(plan.get("summary") or f"A 7-day meal plan for {user.get('name', 'the user')}'s profile.")
    result: Dict[str, Any] = {"summary": summary, "days": normalized}
    nutrition_targets = _normalize_nutrition_block(plan.get("nutrition_targets"))
    if nutrition_targets:
        result["nutrition_targets"] = nutrition_targets
    return result


def _meal_plan_failure_dir() -> str:
    return os.path.join(BASE_DIR, "eval", "results", "meal_plan_raw_failures")


def _save_meal_plan_failure_raw(raw: str, reason: str, attempt: int) -> None:
    if os.getenv("MEAL_PLAN_CAPTURE_FAILURES", "1").strip().lower() in ("0", "false", "no"):
        return
    out_dir = _meal_plan_failure_dir()
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_reason = re.sub(r"[^a-z0-9_]+", "_", reason.lower()).strip("_") or "unknown"
    path = os.path.join(out_dir, f"{stamp}_attempt{attempt}_{safe_reason}.txt")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(raw or "")
    except OSError:
        pass


DIET_PREFERENCE_PROMPT_RULES: Dict[str, List[str]] = {
    "high_protein": [
        "Diet preference is HIGH PROTEIN.",
        "Include a clear lean protein source in breakfast, lunch, dinner, and snack.",
        "Prefer eggs, Greek yogurt, chicken, turkey, fish, tofu, tempeh, legumes, and lean dairy where allowed.",
        "Each day focus should mention protein balance.",
    ],
    "vegetarian": [
        "Diet preference is VEGETARIAN: no meat, poultry, or fish.",
        "Use eggs, dairy, legumes, tofu, tempeh, nuts, and seeds for protein where appropriate.",
    ],
    "vegan": [
        "Diet preference is VEGAN: no meat, poultry, fish, eggs, or dairy.",
        "Use legumes, tofu, tempeh, nuts, seeds, and plant milks for protein and variety.",
    ],
    "low_carb": [
        "Diet preference is LOW CARB.",
        "Limit bread, pasta, rice, and sugary snacks; emphasize vegetables, protein, and healthy fats.",
    ],
    "mediterranean": [
        "Diet preference is MEDITERRANEAN.",
        "Emphasize vegetables, olive oil, legumes, whole grains, fish, and moderate lean poultry.",
    ],
    "pescatarian": [
        "Diet preference is PESCATARIAN: no meat or poultry.",
        "Use fish, seafood alternatives where safe, eggs, dairy, legumes, and plant proteins.",
    ],
    "balanced": [
        "Diet preference is BALANCED.",
        "Include vegetables, whole grains, lean protein, and healthy fats across the week.",
    ],
}


def normalize_diet_preference(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw:
        return ""
    slug = re.sub(r"\s+", "_", raw)
    if slug in DIET_PREFERENCE_PROMPT_RULES:
        return slug
    if "high" in slug and "protein" in slug:
        return "high_protein"
    if "low" in slug and "carb" in slug:
        return "low_carb"
    if "mediterranean" in slug:
        return "mediterranean"
    if "pescatarian" in slug:
        return "pescatarian"
    if slug.startswith("vegan"):
        return "vegan"
    if "vegetarian" in slug:
        return "vegetarian"
    if slug in {"balanced", "balance", "general", "none", "no_preference"}:
        return "balanced"
    return slug


def build_meal_plan_diet_requirement_lines(user: Dict[str, Any]) -> str:
    preference = normalize_diet_preference(user.get("diet_preference"))
    rules = DIET_PREFERENCE_PROMPT_RULES.get(preference)
    if not rules:
        if preference:
            label = preference.replace("_", " ")
            return f"- Diet preference: {label}. Respect this preference in every meal.\n"
        return ""
    joined = "\n".join(f"- {rule}" for rule in rules)
    return f"{joined}\n"


def _build_meal_plan_prompt(user: Dict[str, Any], profile_summary: str, context: str, repair_note: str = "") -> str:
    allergies = [str(item).strip() for item in (user.get("allergies") or []) if str(item).strip()]
    dislikes = [str(item).strip() for item in (user.get("food_dislikes") or []) if str(item).strip()]
    avoid_bits: List[str] = []
    if allergies:
        avoid_bits.append(f"allergens: {', '.join(allergies)}")
    if dislikes:
        avoid_bits.append(f"dislikes: {', '.join(dislikes)}")
    avoid_line = ""
    if avoid_bits:
        avoid_line = (
            f"- STRICTLY avoid { '; '.join(avoid_bits) }. "
            "Never include shrimp, crab, lobster, or related ingredients when shellfish is listed. "
            "Do not write allergen names anywhere in the JSON (summary, meals, focus, or notes).\n"
        )
    repair_block = f"\nFix these issues from the previous attempt:\n{repair_note}\n" if repair_note else ""
    diet_block = build_meal_plan_diet_requirement_lines(user)
    condition_block = build_meal_plan_condition_requirement_lines(user)
    return f"""You are a professional but safe nutrition coach. Create a detailed but practical 7-day meal plan as valid JSON.
Return JSON only. No markdown fences, no commentary, no text before or after the JSON.
Use this exact structure:
{{
  "summary": "...",
  "nutrition_targets": {{"calories": 1800, "protein_g": 140, "carbs_g": 160, "fat_g": 55}},
  "days": [
    {{
      "day": 1,
      "breakfast": "...",
      "lunch": "...",
      "dinner": "...",
      "snack": "...",
      "focus": "...",
      "notes": "Swap salmon for cod if preferred.",
      "daily_totals": {{"calories": 1820, "protein_g": 142, "carbs_g": 155, "fat_g": 58}}
    }}
  ]
}}
Requirements:
- Include exactly 7 day objects numbered 1 through 7.
- Each day must have different lunch and dinner ideas (no copy-paste across days).
- Each meal must list specific foods with gram portions; max 22 words per meal field.
- Never name a forbidden ingredient as the meal, even in examples or warnings.
{avoid_line}{diet_block}{condition_block}- Respect medical conditions and diet preferences.
Profile:
{profile_summary}
Knowledge context:
{context}
{repair_block}"""


def _build_meal_plan_repair_prompt(user: Dict[str, Any], profile_summary: str, issues: List[str]) -> str:
    allergies = ", ".join(str(item) for item in (user.get("allergies") or [])) or "none"
    issue_text = "; ".join(issues) if issues else "invalid JSON or schema"
    diet_block = build_meal_plan_diet_requirement_lines(user)
    condition_block = build_meal_plan_condition_requirement_lines(user)
    return f"""Return ONLY valid JSON for a 7-day meal plan. No markdown fences.
Structure: {{"summary":"...","nutrition_targets":{{"calories":1800,"protein_g":140,"carbs_g":160,"fat_g":55}},"days":[{{"day":1,"breakfast":"...","lunch":"...","dinner":"...","snack":"...","focus":"...","notes":"...","daily_totals":{{"calories":1820,"protein_g":142,"carbs_g":155,"fat_g":58}}}}, ... 7 days]}}
Rules:
- 7 unique lunch/dinner pairs.
- Allergies to avoid: {allergies}. No shrimp, crab, lobster, or shellfish ingredients.
- Do not write allergen names anywhere in the JSON.
- Use gram portions in meals; max 22 words per meal field.
{diet_block}{condition_block}- Previous failure: {issue_text}
Profile:
{profile_summary}
"""


def generate_meal_plan(user: Dict[str, Any], latest: Optional[Dict[str, Any]], rag_store=None) -> Tuple[Dict[str, Any], bool]:
    assert_ollama_ready()
    profile_summary = build_profile_summary(user, latest)
    context = ""
    if rag_store is not None:
        try:
            docs = rag_store.similarity_search(
                f"nutrition meal planning for {user.get('goal', 'general health')} {', '.join(user.get('medical_conditions', []))}",
                k=2,
            )
            context = "\n\n".join(doc.page_content[:1000] for doc in docs)
        except Exception:
            context = ""

    prompt = _build_meal_plan_prompt(user, profile_summary, context)
    llm_degraded = False
    llm = create_ollama_llm(
        num_predict=get_meal_plan_num_predict(),
        temperature=get_meal_plan_temperature(),
    )
    last_issues: List[str] = []
    last_raw = ""
    for attempt in range(1, 4):
        attempt_prompt = prompt if attempt < 3 else _build_meal_plan_repair_prompt(user, profile_summary, last_issues)
        if attempt == 2 and last_issues:
            attempt_prompt = _build_meal_plan_prompt(user, profile_summary, context, repair_note="\n".join(f"- {issue}" for issue in last_issues))
        try:
            response = llm.invoke(attempt_prompt)
            last_raw = response or ""
            cleaned = _extract_json_object(strip_think_tags(last_raw))
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and isinstance(parsed.get("days"), list):
                candidate = normalize_meal_plan_days(parsed, user)
                validation = validate_meal_plan(candidate, user)
                if validation["valid"]:
                    return candidate, False
                last_issues = list(validation.get("issues") or [])
                _save_meal_plan_failure_raw(last_raw, "allergy_validation" if any("allergen" in i for i in last_issues) else "validation", attempt)
        except json.JSONDecodeError as exc:
            last_issues = [f"invalid JSON: {exc}"]
            _save_meal_plan_failure_raw(last_raw, "invalid_json", attempt)
        except Exception as exc:
            last_issues = [f"generation error: {exc}"]
            _save_meal_plan_failure_raw(last_raw, "invoke_error", attempt)

    plan = build_offline_meal_plan(user)
    llm_degraded = True

    return plan, llm_degraded


def save_meal_plan(user_id: int, plan: Dict[str, Any], goal: Optional[str] = None) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO Meal_Plans (user_id, goal, plan_json, created_at, updated_at) VALUES (?, ?, ?, datetime('now', 'localtime'), datetime('now', 'localtime'))",
        (int(user_id), goal or "", json.dumps(plan, ensure_ascii=False)),
    )
    plan_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return {"plan_id": plan_id, "user_id": int(user_id), "goal": goal or "", "plan": plan}


def get_latest_meal_plan(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        "SELECT plan_id, user_id, goal, plan_json, created_at, updated_at FROM Meal_Plans WHERE user_id = ? ORDER BY datetime(updated_at) DESC, plan_id DESC LIMIT 1",
        (int(user_id),),
    ).fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    try:
        data["plan"] = json.loads(data.pop("plan_json"))
    except Exception:
        data["plan"] = {"summary": "Unable to parse meal plan.", "days": []}
    return data

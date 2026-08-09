"""OBSOLETE: legacy schema initializer superseded by logic.init_db().

Do not run this script for validation or migrations. Use:
    python -m pytest tests -q
"""

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "health_coach.db")


def get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = get_conn(db_path)
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
        CREATE UNIQUE INDEX IF NOT EXISTS ux_health_metrics_user_day
        ON Health_Metrics(user_id, date(recorded_at))
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_health_metrics_user_recorded_at
        ON Health_Metrics(user_id, recorded_at DESC)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chat_history_user_timestamp
        ON Chat_History(user_id, timestamp DESC)
        """
    )

    conn.commit()
    conn.close()
    print(f"Database initialized successfully: {db_path}")


if __name__ == "__main__":
    init_db()

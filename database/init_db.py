"""Initialize SQLite database and seed demo data."""

import os
import sqlite3

from werkzeug.security import generate_password_hash


def get_connection(database_path: str) -> sqlite3.Connection:
    """Return a SQLite connection for the given database path."""
    conn = sqlite3.connect(database_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_database(database_path: str) -> None:
    """Create tables if they do not already exist."""
    conn = get_connection(database_path)
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            password_hash TEXT,
            email TEXT UNIQUE,
            role TEXT NOT NULL DEFAULT 'user',
            full_name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            detection_id INTEGER,
            recording_name TEXT NOT NULL,
            video_path TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            duration REAL,
            detected_objects TEXT,
            confidence_score REAL DEFAULT 0.0,
            accident_status TEXT NOT NULL DEFAULT 'non_accident',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            detection_type TEXT NOT NULL,
            media_type TEXT,
            severity TEXT NOT NULL DEFAULT 'low',
            confidence REAL DEFAULT 0.0,
            class_name TEXT,
            image_path TEXT,
            frame_path TEXT,
            screenshot_path TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            start_time TEXT,
            end_time TEXT,
            duration REAL,
            total_frames INTEGER DEFAULT 0,
            average_fps REAL DEFAULT 0.0,
            total_objects INTEGER DEFAULT 0,
            object_counts TEXT,
            output_video_path TEXT,
            session_summary TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            theme TEXT NOT NULL DEFAULT 'light',
            notifications_enabled INTEGER NOT NULL DEFAULT 1,
            language TEXT NOT NULL DEFAULT 'en'
        );
        """
    )
    _ensure_detection_session_columns(cursor)
    _ensure_detection_ownership(cursor)
    _ensure_user_auth_columns(cursor)
    _ensure_recordings_table(cursor)
    conn.commit()
    conn.close()


def _ensure_user_auth_columns(cursor: sqlite3.Cursor) -> None:
    """Ensure the users table supports hashed passwords and legacy records."""
    existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(users)").fetchall()}
    if "password_hash" not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "email" not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if "full_name" not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT")


def _ensure_recordings_table(cursor: sqlite3.Cursor) -> None:
    """Create the webcam recordings table and backfill it from completed detections."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            detection_id INTEGER,
            recording_name TEXT NOT NULL,
            video_path TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            duration REAL,
            detected_objects TEXT,
            confidence_score REAL DEFAULT 0.0,
            accident_status TEXT NOT NULL DEFAULT 'non_accident',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )


def _ensure_detection_session_columns(cursor: sqlite3.Cursor) -> None:
    """Add webcam session columns to existing detections tables."""
    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(detections)").fetchall()
    }
    required_columns = {
        "start_time": "TEXT",
        "end_time": "TEXT",
        "duration": "REAL",
        "total_frames": "INTEGER DEFAULT 0",
        "average_fps": "REAL DEFAULT 0.0",
        "total_objects": "INTEGER DEFAULT 0",
        "object_counts": "TEXT",
        "output_video_path": "TEXT",
        "session_summary": "TEXT",
        "user_id": "INTEGER",
        "class_name": "TEXT",
        "media_type": "TEXT",
        "image_path": "TEXT",
        "frame_path": "TEXT",
        "screenshot_path": "TEXT",
    }

    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE detections ADD COLUMN {column_name} {column_type}")


def _ensure_detection_ownership(cursor: sqlite3.Cursor) -> None:
    """Backfill missing detection owners and prevent future ownerless inserts."""
    existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(detections)").fetchall()}
    if "user_id" not in existing_columns:
        cursor.execute("ALTER TABLE detections ADD COLUMN user_id INTEGER")

    owner_id = _default_detection_owner_id(cursor)
    if owner_id is not None:
        cursor.execute("UPDATE detections SET user_id = ? WHERE user_id IS NULL", (owner_id,))

    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS detections_require_user_id_insert
        BEFORE INSERT ON detections
        WHEN NEW.user_id IS NULL
        BEGIN
            SELECT RAISE(ABORT, 'detections.user_id is required');
        END;
        """
    )
    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS detections_require_user_id_update
        BEFORE UPDATE OF user_id ON detections
        WHEN NEW.user_id IS NULL
        BEGIN
            SELECT RAISE(ABORT, 'detections.user_id is required');
        END;
        """
    )


def _default_detection_owner_id(cursor: sqlite3.Cursor) -> int | None:
    """Return a stable default user id for seeded detections."""
    admin_row = cursor.execute(
        "SELECT id FROM users WHERE lower(role) = 'admin' ORDER BY id LIMIT 1"
    ).fetchone()
    if admin_row:
        return admin_row[0]

    user_row = cursor.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()
    return user_row[0] if user_row else None


def seed_demo_data(database_path: str) -> None:
    """Insert starter data if tables are empty."""
    conn = get_connection(database_path)
    cursor = conn.cursor()

    _seed_or_migrate_users(cursor)
    owner_id = _default_detection_owner_id(cursor)

    if owner_id is not None and cursor.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO detections (user_id, title, detection_type, severity, confidence, status) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (owner_id, "Rear-end collision", "accident", "high", 0.94, "completed"),
                (owner_id, "Road obstruction", "non_accident", "low", 0.82, "completed"),
            ],
        )

    if cursor.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 0:
        cursor.execute("INSERT INTO reports (report_name) VALUES (?)", ("Summary Report",))

    if cursor.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO settings (theme, notifications_enabled, language) VALUES (?, ?, ?)",
            ("light", 1, "en"),
        )

    conn.commit()
    conn.close()


def _seed_or_migrate_users(cursor: sqlite3.Cursor) -> None:
    """Seed default users and backfill password hashes for legacy rows."""
    users = cursor.execute("SELECT id, username, password, password_hash, role FROM users").fetchall()
    for user in users:
        password_hash = user[3]
        if not password_hash:
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (generate_password_hash(user[2]), user[0]),
            )

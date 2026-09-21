import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "backend" / "data" / "annotations.db"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", BASE_DIR / "public" / "samples" / "metadata.json"))


@contextmanager
def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database():
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                ground_truth_label TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE
            );
            """
        )

        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        connection.executemany(
            """
            INSERT INTO images (filename, ground_truth_label)
            VALUES (:filename, :label)
            ON CONFLICT(filename) DO UPDATE SET ground_truth_label = excluded.ground_truth_label
            """,
            metadata,
        )


def image_rows():
    with get_connection() as connection:
        return connection.execute(
            "SELECT id, filename, ground_truth_label FROM images ORDER BY id"
        ).fetchall()


def annotation_rows(image_id: int):
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, image_id, label, x, y, width, height, created_at
            FROM annotations
            WHERE image_id = ?
            ORDER BY id
            """,
            (image_id,),
        ).fetchall()


def image_exists(image_id: int):
    with get_connection() as connection:
        return connection.execute(
            "SELECT 1 FROM images WHERE id = ?", (image_id,)
        ).fetchone() is not None


def create_annotation(image_id: int, label: str, x: float, y: float, width: float, height: float):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO annotations (image_id, label, x, y, width, height)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (image_id, label, x, y, width, height),
        )
        return connection.execute(
            """
            SELECT id, image_id, label, x, y, width, height, created_at
            FROM annotations WHERE id = ?
            """,
            (cursor.lastrowid,),
        ).fetchone()


def delete_annotation(annotation_id: int):
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM annotations WHERE id = ?", (annotation_id,)
        )
        return cursor.rowcount > 0

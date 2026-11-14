"""Store prediction history locally without student names or IDs."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.data import PROJECT_ROOT
from src.predict import normalize_batch_columns
from src.train import MODEL_FEATURES

HISTORY_PATH = PROJECT_ROOT / "outputs" / "engineering_prediction_history.db"


def _connect() -> sqlite3.Connection:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at_utc TEXT NOT NULL,
            mode TEXT NOT NULL CHECK (mode IN ('early', 'updated')),
            model_version TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            estimated_grade REAL NOT NULL CHECK (estimated_grade >= 0 AND estimated_grade <= 120)
        )
        """
    )
    return connection


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """Commit successful operations and always close the SQLite handle."""
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def record_prediction(
    mode: str,
    inputs: dict,
    estimated_grade: float,
    algorithm: str,
    model_version: str,
) -> None:
    """Save feature values and result, deliberately excluding student identifiers."""
    if mode not in MODEL_FEATURES:
        raise ValueError(f"Unknown prediction mode: {mode}")
    selected_inputs = {
        feature: (None if inputs[feature] is None else float(inputs[feature]))
        for feature in MODEL_FEATURES[mode]
    }
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history
                (created_at_utc, mode, model_version, algorithm, inputs_json, estimated_grade)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                mode,
                model_version,
                algorithm,
                json.dumps(selected_inputs, sort_keys=True),
                float(estimated_grade),
            ),
        )


def record_batch_predictions(
    rows: pd.DataFrame,
    predictions: pd.DataFrame,
    mode: str,
    algorithm: str,
    model_version: str,
) -> None:
    """Save batch results without carrying through student IDs or extra CSV columns."""
    if mode not in MODEL_FEATURES:
        raise ValueError(f"Unknown prediction mode: {mode}")
    if len(rows) != len(predictions):
        raise ValueError("The number of student rows does not match the number of predictions.")
    features = MODEL_FEATURES[mode]
    normalized_rows = normalize_batch_columns(rows)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for position, (_, prediction) in enumerate(predictions.iterrows()):
        selected_inputs = {
            feature: float(normalized_rows.iloc[position][feature]) for feature in features
        }
        records.append((
            created_at,
            mode,
            model_version,
            algorithm,
            json.dumps(selected_inputs, sort_keys=True),
            float(prediction["estimated_final_score"]),
        ))
    with _connection() as connection:
        connection.executemany(
            """
            INSERT INTO prediction_history
                (created_at_utc, mode, model_version, algorithm, inputs_json, estimated_grade)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )


def get_history(limit: int = 250) -> pd.DataFrame:
    """Return the newest locally stored predictions."""
    with _connection() as connection:
        return pd.read_sql_query(
            """
            SELECT prediction_id, created_at_utc, mode, model_version,
                   algorithm, inputs_json, estimated_grade
            FROM prediction_history
            ORDER BY prediction_id DESC
            LIMIT ?
            """,
            connection,
            params=(limit,),
        )


def clear_history() -> None:
    """Remove all stored prediction records."""
    with _connection() as connection:
        connection.execute("DELETE FROM prediction_history")

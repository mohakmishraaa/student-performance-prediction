"""Load saved models and estimate a course Final mark."""

import unicodedata
from pathlib import Path

import joblib
import pandas as pd

from src.train import MODEL_FEATURES, MODEL_PATH

CSV_FEATURE_ALIASES = {feature: f"{feature}_score" for feature in MODEL_FEATURES["updated"]}
SCORE_MAX = 120
MAX_BATCH_ROWS = 1_000
MAX_BATCH_FILE_BYTES = 5 * 1024 * 1024


def normalize_batch_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Trim/case-normalize supported headers and map CSV aliases to features."""
    normalized = rows.copy()
    headers = [str(column).strip().lstrip("\ufeff").strip().lower() for column in normalized.columns]
    duplicates = sorted({name for name in headers if headers.count(name) > 1})
    if duplicates:
        raise ValueError(f"CSV has duplicate column names after normalization: {', '.join(duplicates)}")
    normalized.columns = headers
    alias_to_feature = {alias: feature for feature, alias in CSV_FEATURE_ALIASES.items()}
    alias_to_feature.update({"mid-sem": "midterm", "mid-sem_score": "midterm"})
    normalized = normalized.rename(columns=alias_to_feature)
    duplicates = sorted({name for name in normalized.columns if list(normalized.columns).count(name) > 1})
    if duplicates:
        raise ValueError(f"CSV provides the same mark more than once: {', '.join(duplicates)}")
    return normalized


def spreadsheet_safe_csv(frame: pd.DataFrame) -> str:
    """Export a CSV with text cells escaped against spreadsheet formula execution."""
    safe = frame.copy()

    def escape(value, column_name):
        if not isinstance(value, str):
            return value
        first_visible = value
        while first_visible:
            character = first_visible[0]
            if character.isspace() or unicodedata.category(character).startswith("C"):
                first_visible = first_visible[1:]
            else:
                break
        if first_visible.startswith(("=", "+", "-", "@")):
            return "'" + value
        if column_name == "student_id" and value.strip().isdigit() and len(value.strip()) > 1 and value.strip().startswith("0"):
            return "'" + value
        return value

    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(lambda value: escape(value, str(column).strip().lower()))
    return safe.to_csv(index=False)


def predict_grade(values: dict, mode: str = "updated", model_path: Path = MODEL_PATH) -> float:
    """Return the estimated Final mark on the source dataset's 0–120 scale."""
    if mode not in MODEL_FEATURES:
        raise ValueError(f"Unknown prediction mode: {mode}")
    features = MODEL_FEATURES[mode]
    missing = [feature for feature in features if feature not in values]
    if missing:
        raise ValueError(f"Missing input features: {missing}")
    row = pd.DataFrame([{feature: values[feature] for feature in features}])
    estimate = float(joblib.load(model_path)[mode].predict(row)[0])
    return max(0.0, min(float(SCORE_MAX), estimate))


def predict_batch(rows: pd.DataFrame, mode: str = "updated", model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """Validate batch input rows and append estimated Final marks."""
    if mode not in MODEL_FEATURES:
        raise ValueError(f"Unknown prediction mode: {mode}")
    if rows.empty:
        raise ValueError("The CSV has no student rows.")
    if len(rows) > MAX_BATCH_ROWS:
        raise ValueError(f"The CSV has {len(rows):,} rows; the limit is {MAX_BATCH_ROWS:,} rows per upload.")

    rows = normalize_batch_columns(rows)
    features = MODEL_FEATURES[mode]
    allowed_columns = set(MODEL_FEATURES["updated"]) | {"student_id"}
    unexpected = sorted(set(rows.columns) - allowed_columns)
    if unexpected:
        raise ValueError(
            "Unsupported CSV column(s): " + ", ".join(unexpected)
            + ". Use only assessment score columns and the optional student_id column."
        )
    rows = rows.reset_index(drop=True)
    missing = [feature for feature in features if feature not in rows.columns]
    if missing:
        names = [CSV_FEATURE_ALIASES[feature] for feature in missing]
        raise ValueError(f"CSV is missing required columns: {', '.join(names)}")
    inputs = rows.loc[:, features].apply(pd.to_numeric, errors="coerce")
    invalid = set()
    for feature in features:
        bad = inputs[feature].isna() | inputs[feature].lt(0) | inputs[feature].gt(SCORE_MAX)
        invalid.update((inputs.index[bad] + 2).tolist())
    if invalid:
        examples = ", ".join(map(str, sorted(invalid)[:10]))
        extra = "…" if len(invalid) > 10 else ""
        raise ValueError(f"Invalid or missing marks in CSV row(s) {examples}{extra}. Marks must be numeric and between 0 and {SCORE_MAX}.")
    output = inputs.copy().rename(columns=CSV_FEATURE_ALIASES)
    if "student_id" in rows.columns:
        output.insert(0, "student_id", rows["student_id"].astype("string"))
    output["estimated_final_score"] = joblib.load(model_path)[mode].predict(inputs[features]).clip(0, SCORE_MAX).round(1)
    return output

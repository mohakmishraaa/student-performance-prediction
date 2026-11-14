"""Load the public Chemical Engineering course grades dataset."""

import csv
from io import StringIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "engineering-course-grades.csv"


def load_data(download_if_missing: bool = True, data_path: Path | None = None) -> pd.DataFrame:
    """Load local engineering course marks; download when the copy is absent."""
    path = Path(data_path) if data_path is not None else DATA_PATH
    if path.exists():
        raw = path.read_text(encoding="utf-8-sig")
    elif download_if_missing and data_path is None:
        from urllib.request import urlopen

        with urlopen("https://openmv.net/file/class-grades.csv", timeout=30) as response:
            raw = response.read().decode("utf-8-sig")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw, encoding="utf-8")
    else:
        raise FileNotFoundError(f"Engineering grades dataset not found at {path}.")

    # The published CSV contains rows with one extra trailing empty field.
    records = list(csv.reader(StringIO(raw)))
    header = records[0]
    normalized = []
    for row in records[1:]:
        while len(row) > len(header) and not row[-1].strip():
            row.pop()
        if len(row) != len(header):
            raise ValueError("The engineering CSV contains a row with an unexpected number of columns.")
        normalized.append(row)
    frame = pd.DataFrame(normalized, columns=header)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    expected = {"prefix", "assignment", "tutorial", "midterm", "takehome", "final"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Engineering dataset is missing columns: {sorted(missing)}")
    return frame


if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data)} Chemical Engineering course records from {DATA_PATH}")
    print(f"Columns: {', '.join(data.columns)}")

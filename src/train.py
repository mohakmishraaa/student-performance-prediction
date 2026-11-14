"""Train models that estimate a Chemical Engineering course final mark."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data import PROJECT_ROOT, load_data

MODEL_FEATURES = {
    "early": ["assignment", "tutorial"],
    "updated": ["assignment", "tutorial", "midterm", "takehome"],
}
FEATURES = MODEL_FEATURES["updated"]
TARGET = "final"
MODEL_PATH = PROJECT_ROOT / "outputs" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"
EVALUATION_PATH = PROJECT_ROOT / "outputs" / "evaluation_predictions.csv"
CV_FOLDS = 5
CV_REPEATS = 3


def train_model(data: pd.DataFrame | None = None, data_path: Path | None = None,
                output_dir: Path = PROJECT_ROOT / "outputs") -> dict:
    """Fit early and updated models and report metrics on a held-out split."""
    data = data if data is not None else load_data(data_path=data_path)
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data = data.dropna(subset=[TARGET]).copy()
    y = data[TARGET].astype(float)
    train_indices, test_indices = train_test_split(data.index, test_size=0.2, random_state=42)
    candidates = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=40, min_samples_leaf=3, random_state=42),
    }
    model_simplicity = {
        "Mean baseline": 0,
        "Linear Regression": 1,
        "Decision Tree": 2,
        "Random Forest": 3,
    }
    models, model_metrics, evaluation_records = {}, {}, []

    for mode, features in MODEL_FEATURES.items():
        X_train, X_test = data.loc[train_indices, features], data.loc[test_indices, features]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        preprocessing = ColumnTransformer(
            [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), features)],
            remainder="drop",
        )
        fold_splitter = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=42)
        candidate_results, candidate_models, cv_scores_by_name = [], {}, {}
        for name, regressor in candidates.items():
            candidate = Pipeline([("preprocessing", preprocessing), ("regressor", regressor)])
            cv_scores = -cross_val_score(candidate, X_train, y_train, cv=fold_splitter,
                                         scoring="neg_mean_absolute_error", n_jobs=1)
            cv_scores_by_name[name] = cv_scores
            candidate_results.append({
                "name": name,
                "cv_mae": float(cv_scores.mean()),
                "cv_mae_std": float(cv_scores.std(ddof=1)),
                # Repeated fold scores are correlated. Use folds, not all 15 scores,
                # as the effective count for this explicitly heuristic one-SE rule.
                "cv_mae_se": float(cv_scores.std(ddof=1) / np.sqrt(CV_FOLDS)),
            })
            candidate_models[name] = candidate
        best_result = min(candidate_results, key=lambda result: result["cv_mae"])
        one_se_limit = best_result["cv_mae"] + best_result["cv_mae_se"]
        eligible_results = [result for result in candidate_results
                            if result["cv_mae"] <= one_se_limit]
        selected_name = min(
            eligible_results,
            key=lambda result: (model_simplicity[result["name"]], result["cv_mae"]),
        )["name"]
        model = candidate_models[selected_name]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test).clip(0, 120)
        baseline_mae = float(mean_absolute_error(y_test, [float(y_train.mean())] * len(y_test)))
        for row_index, actual, predicted in zip(test_indices, y_test, predictions):
            evaluation_records.append({"mode": mode, "dataset_row": int(row_index) + 1,
                "actual_grade": float(actual), "predicted_grade": float(predicted),
                "absolute_error": abs(float(actual) - float(predicted))})
        delta_by_name = {
            name: cv_scores_by_name[name] - cv_scores_by_name["Mean baseline"]
            for name in cv_scores_by_name
        }
        model_metrics[mode] = {
            "selected_algorithm": selected_name, "cv_folds": CV_FOLDS, "cv_repeats": CV_REPEATS,
            "cv_mae_se_method": "fold standard deviation divided by sqrt(5); heuristic for repeated CV",
            "selection_rule": "one-standard-error simplicity rule",
            "candidate_comparison": [{
                **result,
                "cv_mae": round(result["cv_mae"], 3),
                "cv_mae_std": round(result["cv_mae_std"], 3),
                "cv_mae_se": round(result["cv_mae_se"], 3),
                "cv_delta_vs_baseline": round(float(delta_by_name[result["name"]].mean()), 3),
                "cv_delta_std_vs_baseline": round(float(delta_by_name[result["name"]].std(ddof=1)), 3),
                "cv_delta_se_vs_baseline": round(float(
                    delta_by_name[result["name"]].std(ddof=1) / np.sqrt(CV_FOLDS)
                ), 3),
                "selected": result["name"] == selected_name,
            } for result in sorted(candidate_results, key=lambda result: result["cv_mae"])],
            "features": features, "test_rows": int(len(y_test)),
            "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
            "baseline_mae": round(baseline_mae, 3),
            "mae_change_vs_baseline": round(
                float(mean_absolute_error(y_test, predictions)) - baseline_mae, 3
            ),
            "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 3),
            "r2": round(float(r2_score(y_test, predictions)), 3),
            "evaluation_training_rows": int(len(y_train)),
            "deployment_training_rows": int(len(y)),
        }
        # Preserve an untouched holdout evaluation above, then fit the selected
        # algorithm on every labeled record for the model used by the app.
        deployment_model = candidate_models[selected_name]
        deployment_model.fit(data.loc[:, features], y)
        models[mode] = deployment_model
        if selected_name == "Linear Regression":
            model_metrics[mode]["linear_coefficients"] = {
                feature: round(float(value), 4)
                for feature, value in zip(features, deployment_model.named_steps["regressor"].coef_)
            }
    metrics = {"task": "regression", "target": TARGET,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "training_rows": int(len(data)), "score_min": 0, "score_max": 120,
        "dataset": "McMaster Chemical Engineering course grades (OpenMV)", "models": model_metrics}
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(evaluation_records).to_csv(output_dir / "evaluation_predictions.csv", index=False)
    if output_dir.resolve() == (PROJECT_ROOT / "outputs").resolve():
        active_source = output_dir / "active_model.json"
        if active_source.exists():
            active_source.unlink()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    result = train_model(data_path=args.data_path, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))
    print(f"Model saved to {args.output_dir / 'model.joblib'}")

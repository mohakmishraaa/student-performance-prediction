"""Streamlit interface for the Student Performance Predictor."""

import hashlib
import json
import os

import pandas as pd
import streamlit as st

from src.predict import (
    CSV_FEATURE_ALIASES,
    MAX_BATCH_FILE_BYTES,
    MAX_BATCH_ROWS,
    SCORE_MAX,
    predict_batch,
    predict_grade,
    spreadsheet_safe_csv,
)
from src.history import clear_history, get_history, record_batch_predictions, record_prediction
from src.train import (
    EVALUATION_PATH,
    METRICS_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    train_model,
)

FEATURE_LABELS = {
    "assignment": "Assignment average",
    "tutorial": "Tutorial average",
    "midterm": "Mid-Sem mark",
    "takehome": "Take-home assessment",
}
FEATURE_HELP = {
    "assignment": "Enter the average mark for assignments completed so far in this course.",
    "tutorial": "Enter the average mark for tutorials completed so far in this course.",
    "midterm": "Enter the mark from the course Midterm assessment (labelled Mid-Sem here).",
    "takehome": "Enter the mark from the take-home assessment, or uncheck its availability if it has not happened yet.",
}
PUBLIC_DEPLOYMENT = os.getenv("STUDENT_PREDICTOR_PUBLIC", "").strip().lower() in {"1", "true", "yes"}
LOCAL_HISTORY_ENABLED = (
    not PUBLIC_DEPLOYMENT
    and os.getenv("STUDENT_PREDICTOR_LOCAL_HISTORY", "").strip().lower() in {"1", "true", "yes"}
)
DEMO_PRESETS = {
    "early": {
        "Lower marks": {"assignment": 65, "tutorial": 70},
        "Typical marks": {"assignment": 86, "tutorial": 90},
        "Higher marks": {"assignment": 96, "tutorial": 103},
    },
    "updated": {
        "Lower marks": {"assignment": 65, "tutorial": 70, "midterm": 55, "takehome": 60},
        "Typical marks": {"assignment": 86, "tutorial": 90, "midterm": 69, "takehome": 88},
        "Higher marks": {"assignment": 96, "tutorial": 103, "midterm": 90, "takehome": 100},
    },
}


def prediction_scatter_svg(rows):
    """Build a small predicted-versus-actual plot without chart dependencies."""
    width, height = 640, 430
    left, top, plot_width, plot_height = 66, 24, 530, 330
    bottom = top + plot_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Actual versus predicted grades scatter plot">',
        '<rect width="100%" height="100%" fill="#111318" rx="12"/>',
    ]
    for tick in range(0, 121, 30):
        x = left + plot_width * tick / 120
        y = top + plot_height * (120 - tick) / 120
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#343943"/>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#343943"/>',
                f'<text x="{x:.1f}" y="{bottom + 22}" fill="#d6d8dc" text-anchor="middle" font-size="13">{tick}</text>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" fill="#d6d8dc" text-anchor="end" font-size="13">{tick}</text>',
            ]
        )
    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot_width}" y2="{top}" '
        'stroke="#e8edf5" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    for row in rows.itertuples(index=False):
        x = left + plot_width * float(row.actual_grade) / 120
        y = top + plot_height * (120 - float(row.predicted_grade)) / 120
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#5eead4" fill-opacity=".82">'
            f'<title>Actual {float(row.actual_grade):.1f}, predicted {float(row.predicted_grade):.1f}</title></circle>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 24}" fill="#f1f3f5" text-anchor="middle" font-size="15">Actual Final mark</text>',
            f'<text x="20" y="{top + plot_height / 2}" fill="#f1f3f5" text-anchor="middle" font-size="15" transform="rotate(-90 20 {top + plot_height / 2})">Predicted Final mark</text>',
            '</svg>',
        ]
    )
    return "".join(parts)

st.set_page_config(page_title="Engineering Final Mark Predictor", page_icon="📘")
st.markdown(
    """
    <style>
    :root { --ink: #e9eef7; --muted: #b0bdcf; --accent: #65dfc0; --panel: #131d2d; --line: #2a3a50; }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 50% -24%, #1c2e47 0%, #101827 40%, #0a101c 82%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1180px; padding-top: 1.7rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: -0.03em; color: var(--ink); }
    h1 { font-size: clamp(2rem, 3.8vw, 2.8rem) !important; line-height: 1.12 !important; margin-bottom: .4rem !important; }
    h2 { font-size: clamp(1.45rem, 2.5vw, 1.9rem) !important; }
    .eyebrow { color: var(--accent); font-size: .73rem; font-weight: 750; letter-spacing: .15em; text-transform: uppercase; margin-bottom: .45rem; }
    .hero-copy { color: #b8c4d5; font-size: 1rem; max-width: 700px; line-height: 1.6; margin-top: 0; }
    .source-pill { display: inline-block; margin-top: .85rem; padding: .62rem .85rem; border-radius: 13px; border: 1px solid #33445a; color: #d3deeb; background: #111c2b; font-size: .8rem; line-height: 1.4; }
    .data-note { margin: .9rem 0 .7rem; padding: .75rem 1rem; border: 1px solid #2e554f; border-left: 4px solid var(--accent); border-radius: 12px; background: #102321; color: #d4e8e3; line-height: 1.5; }
    .data-note strong { display: inline; color: #83e8ce; margin-right: .35rem; font-size: .92rem; }
    [data-testid="stMetric"] { background: linear-gradient(145deg, #172439, #111a2a); border: 1px solid #30415a; border-radius: 15px; padding: .95rem 1.05rem; }
    [data-testid="stMetricLabel"] { color: #b8c5d7; }
    [data-testid="stMetricValue"] { color: #f4f7fb; }
    [data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
    [data-testid="stTabs"] button[role="tab"] { padding: .7rem .9rem; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }
    [data-testid="stForm"] { background: rgba(19, 29, 45, .78); border: 1px solid #304159; border-radius: 16px; padding: 1rem 1.2rem; }
    [data-baseweb="input"] > div, [data-baseweb="select"] > div { border-radius: 10px; }
    [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {
        border-radius: 10px; border: 1px solid #65dfc0; background: #65dfc0; color: #0b201d;
        font-weight: 700; min-height: 2.65rem; transition: transform .15s ease, filter .15s ease;
    }
    .stButton button { border-radius: 10px; border: 1px solid #34465d; background: #172438; color: #e3eaf3; font-weight: 600; min-height: 2.55rem; transition: background .15s ease, border-color .15s ease; }
    .stButton button:hover { border-color: #65dfc0; background: #1b3040; color: #f2fffb; }
    [data-testid="stFormSubmitButton"] button:hover, [data-testid="stDownloadButton"] button:hover { filter: brightness(1.06); transform: translateY(-1px); }
    [data-testid="stProgressBar"] > div > div { background: var(--accent); }
    [data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: var(--line); margin: 1.3rem 0; }
    .result-note { color: var(--muted); font-size: .9rem; }
    @media (max-width: 700px) {
        .block-container { padding: 1.1rem .8rem 2rem; }
        h1 { font-size: 2rem !important; }
        [data-testid="stForm"] { padding: .7rem; }
        .source-pill { white-space: normal; line-height: 1.4; border-radius: 12px; margin-top: .2rem; }
        .data-note { padding: .8rem .9rem; }
        [data-testid="stMetric"] { padding: .8rem; }
        [data-testid="stTabs"] button[role="tab"] { padding: .55rem .45rem; font-size: .86rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
metrics = (
    json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if METRICS_PATH.exists()
    else {}
)
active_model_source_path = MODEL_PATH.parent / "active_model.json"
active_model_source = (
    json.loads(active_model_source_path.read_text(encoding="utf-8-sig"))
    if active_model_source_path.exists()
    else None
)
st.markdown('<div class="eyebrow">Engineering student performance</div>', unsafe_allow_html=True)
hero_left, hero_right = st.columns([3.3, 1])
with hero_left:
    st.title("Engineering Grade Predictor")
    st.markdown(
        '<p class="hero-copy">Explore a Final-mark estimate using a public Chemical Engineering course dataset.</p>',
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown('<div class="source-pill">⚙️ Chemical Engineering dataset<br>Mark scale 0–120</div>', unsafe_allow_html=True)
if active_model_source:
    st.caption(f"Active model imported from Azure ML job: `{active_model_source['job_name']}`")
st.divider()
st.markdown(
    '<div class="data-note"><strong>Course demo</strong>'
    'Uses public marks from one Chemical Engineering course. Its single Midterm is labelled Mid-Sem here; '
    'the app is not validated for other courses.</div>',
    unsafe_allow_html=True,
)
with st.expander("About this demo: data and how to use it"):
    st.markdown(
        "1. Choose whether you have coursework marks only or have completed the Mid-Sem.\n"
        "2. Load an illustrative example or enter marks on the same scale.\n"
        "3. Select **Estimate final mark** to see the result."
    )
    st.caption("The example dataset contains 99 records, including 96 with a Final mark. Scores use a 0–120 scale. Estimates are for demonstration and are not official grades.")

if not MODEL_PATH.exists():
    st.info("The prediction models are not trained yet. Train them once to unlock the app.")
    if st.button("Download data and train model"):
        with st.spinner("Getting the public dataset and training the model..."):
            metrics = train_model()
        st.success("Both prediction models are trained.")

if MODEL_PATH.exists():
    st.subheader("1. What marks do you have?")

    def clear_batch_prediction_result():
        st.session_state.pop("batch_prediction_result", None)

    mode_label = st.radio(
        "Prediction stage",
        ["After the Mid-Sem", "Coursework marks only"],
        horizontal=True,
        label_visibility="collapsed",
        on_change=clear_batch_prediction_result,
    )
    mode = "updated" if mode_label.startswith("After") else "early"
    selected_metrics = metrics.get("models", {}).get(mode, {})
    if mode == "early" and selected_metrics.get("selected_algorithm") == "Mean baseline":
        st.info(
            "Coursework-only estimates use the course average for every input because coursework marks "
            "did not improve validation results. Choose **After the Mid-Sem** for an estimate that responds to marks."
        )
    single_tab, batch_tab, evaluation_tab, history_tab = st.tabs(
        ["✏️  Try one student", "📄  Try a class CSV", "📊  Model check", "🕘  History"]
    )
    with single_tab:
        def reset_student_inputs():
            for feature in MODEL_FEATURES["updated"]:
                st.session_state[f"input_{feature}"] = 70.0
            st.session_state.pop("last_student_prediction", None)

        st.subheader("2. Enter the marks")
        st.caption("Use the sample or enter marks from the 0–120 scale.")
        typical_values = DEMO_PRESETS[mode]["Typical marks"]
        if st.button("Load typical sample", key=f"demo_{mode}_typical", type="primary"):
            for feature, score in typical_values.items():
                st.session_state[f"input_{feature}"] = float(score)
            if mode == "updated":
                st.session_state["takehome_available"] = True
            st.session_state.pop("last_student_prediction", None)
        with st.expander("Try another sample"):
            st.caption("Illustrative examples only; these are not real student records.")
            low_col, high_col, reset_col = st.columns(3)
            for col, preset_name in [(low_col, "Lower marks"), (high_col, "Higher marks")]:
                if col.button(preset_name, key=f"demo_{mode}_{preset_name}", width="stretch"):
                    for feature, score in DEMO_PRESETS[mode][preset_name].items():
                        st.session_state[f"input_{feature}"] = float(score)
                    if mode == "updated":
                        st.session_state["takehome_available"] = True
                    st.session_state.pop("last_student_prediction", None)
            reset_col.button(
                "Reset marks", key=f"reset_{mode}", width="stretch",
                on_click=reset_student_inputs,
            )

        with st.form("student_inputs"):
            st.caption("Only enter marks you already have. The available fields depend on the stage you selected above.")
            values = {}
            for feature in MODEL_FEATURES[mode]:
                if feature == "takehome":
                    continue
                widget_key = f"input_{feature}"
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = 70.0
                values[feature] = st.number_input(
                    f"{FEATURE_LABELS[feature]} (0–120)", min_value=0.0,
                    max_value=float(SCORE_MAX), step=1.0, key=widget_key,
                    help=FEATURE_HELP[feature],
                )
            if mode == "updated":
                takehome_available = st.checkbox(
                    "Take-home mark is available", value=True, key="takehome_available"
                )
                if takehome_available:
                    values["takehome"] = st.number_input(
                        f"{FEATURE_LABELS['takehome']} (0–120)", min_value=0.0,
                        max_value=float(SCORE_MAX), step=1.0, key="input_takehome",
                        help="Uncheck the box if this assessment has not happened yet; the model will use its training-set median for this input.",
                    )
                else:
                    values["takehome"] = None
            submitted = st.form_submit_button("3. Estimate final mark", width="stretch")

        if submitted:
            estimate = predict_grade(values, mode=mode)
            st.session_state["last_student_prediction"] = {
                "mode": mode,
                "inputs": {feature: (None if value is None else float(value)) for feature, value in values.items()},
                "estimate": float(estimate),
                "algorithm": selected_metrics.get("selected_algorithm", "unknown"),
                "model_version": metrics.get("model_version", "local-model"),
            }
            if LOCAL_HISTORY_ENABLED:
                record_prediction(
                    mode=mode,
                    inputs=values,
                    estimated_grade=estimate,
                    algorithm=selected_metrics.get("selected_algorithm", "unknown"),
                    model_version=metrics.get("model_version", "local-model"),
                )

        last_prediction = st.session_state.get("last_student_prediction")
        current_values = {feature: (None if value is None else float(value)) for feature, value in values.items()}
        prediction_matches = (
            last_prediction is not None
            and last_prediction["mode"] == mode
            and last_prediction["inputs"] == current_values
        )
        if prediction_matches:
            with st.container(border=True):
                st.markdown("### Your estimated final mark")
                result_col, context_col = st.columns([1, 2])
                with result_col:
                    baseline_selected = selected_metrics.get("selected_algorithm") == "Mean baseline"
                    result_label = "Course average (same for all inputs)" if baseline_selected else "Estimated Final mark"
                    st.metric(result_label, f"{last_prediction['estimate']:.1f} / 120")
                    st.progress(last_prediction["estimate"] / 120)
                with context_col:
                    st.markdown("**Marks used for this estimate**")
                    st.write(" · ".join(
                        f"{FEATURE_LABELS[feature]}: {'Not available' if value is None else f'{value:g}'}"
                        for feature, value in last_prediction["inputs"].items()
                    ))
                    st.markdown(
                        '<p class="result-note">An estimate based on past course records; it is not an official grade or a guarantee.</p>',
                        unsafe_allow_html=True,
                    )
                st.caption(
                    ("Mid-Sem and take-home marks are included. " if mode == "updated" else "")
                    + "The model shows past patterns; it does not prove that changing one mark causes a score change."
                )
                if baseline_selected:
                    st.info(
                        "This coursework-only result is the training-set course average. "
                        "Coursework marks did not improve validation results, so they do not change this estimate."
                    )
                elif selected_metrics:
                    improvement = selected_metrics["baseline_mae"] - selected_metrics["mae"]
                    st.caption(
                        f"On a single held-out check of {selected_metrics['test_rows']} records, the MAE difference "
                        f"versus the course-average baseline was {improvement:+.1f} marks. With only "
                        f"{selected_metrics['test_rows']} records, this check is noisy; repeated cross-validation "
                        "is the primary model-comparison evidence, and neither result guarantees accuracy."
                    )
                result_csv = pd.DataFrame([{
                    "prediction_stage": "updated" if mode == "updated" else "early",
                    **{CSV_FEATURE_ALIASES[feature]: value for feature, value in last_prediction["inputs"].items()},
                    "estimated_final_score": round(last_prediction["estimate"], 1),
                    "model_algorithm": last_prediction["algorithm"],
                    "model_version": last_prediction["model_version"],
                }])
                result_csv = spreadsheet_safe_csv(result_csv)
                st.download_button(
                    "Download this result as CSV", data=result_csv,
                    file_name=f"final_mark_{mode}_result.csv", mime="text/csv",
                )
                with st.expander("How should I read this estimate?"):
                    mae_percent = selected_metrics["mae"] / SCORE_MAX * 100
                    baseline_percent = selected_metrics["baseline_mae"] / SCORE_MAX * 100
                    st.write(
                        f"When checked on {selected_metrics['test_rows']} past records not used to fit the model, "
                        f"its average miss was {selected_metrics['mae']:.1f} marks ({mae_percent:.1f}% of the scale). "
                        f"A simple average-based guess missed by {selected_metrics['baseline_mae']:.1f} marks "
                        f"({baseline_percent:.1f}% of the scale)."
                    )
                    st.caption(
                        "Average error describes a group of students, not a guaranteed range for one student. "
                        "The dataset covers one small public course, so accuracy may differ elsewhere."
                    )

    with batch_tab:
        required_features = MODEL_FEATURES[mode]
        csv_features = [CSV_FEATURE_ALIASES.get(feature, feature) for feature in required_features]
        st.subheader("Predict for a group")
        st.write(f"Upload one row per student. Required columns: `{', '.join(csv_features)}`.")
        st.caption(
            f"Optional: `student_id`. Only assessment columns and this ID are accepted. "
            f"Limit: {MAX_BATCH_ROWS:,} rows and 5 MB. IDs appear in your result file but are not saved to history."
        )
        example = {"student_id": ["S001", "S002"]}
        for feature in MODEL_FEATURES[mode]:
            example[CSV_FEATURE_ALIASES[feature]] = [72, 86]
        sample_col, upload_col = st.columns([1, 2])
        with sample_col:
            st.download_button(
                "Download sample CSV",
                data=pd.DataFrame(example).to_csv(index=False),
                file_name=f"student_{mode}_template.csv",
                mime="text/csv",
                width="stretch",
            )
        with upload_col:
            upload = st.file_uploader("Choose a student CSV file", type=["csv"], key="batch_csv")
            submitted_batch = st.button("Estimate marks for this class", width="stretch")
        upload_digest = hashlib.sha256(upload.getvalue()).hexdigest() if upload is not None else None
        previous_batch_result = st.session_state.get("batch_prediction_result")
        if previous_batch_result and (
            previous_batch_result.get("mode") != mode
            or previous_batch_result.get("upload_digest") != upload_digest
        ):
            st.session_state.pop("batch_prediction_result", None)
        if submitted_batch:
            st.session_state.pop("batch_prediction_result", None)
            try:
                if upload is None:
                    st.error("Choose a CSV file first.")
                elif upload.size > MAX_BATCH_FILE_BYTES:
                    st.error("This file is larger than 5 MB. Choose a smaller CSV.")
                else:
                    upload.seek(0)
                    uploaded_rows = pd.read_csv(upload, encoding="utf-8-sig", dtype="string", keep_default_na=False)
                    predictions = predict_batch(uploaded_rows, mode=mode)
                    if LOCAL_HISTORY_ENABLED:
                        record_batch_predictions(
                            rows=uploaded_rows,
                            predictions=predictions,
                            mode=mode,
                            algorithm=selected_metrics.get("selected_algorithm", "unknown"),
                            model_version=metrics.get("model_version", "local-model"),
                        )
                    st.session_state["batch_prediction_result"] = {
                        "mode": mode,
                        "upload_digest": upload_digest,
                        "predictions": predictions,
                    }
                    st.success(
                        "Estimates are ready. Prediction history is disabled."
                        if not LOCAL_HISTORY_ENABLED
                        else "Estimates are ready. Marks were added to this computer’s local history; IDs were not saved."
                    )
            except (
                ValueError,
                UnicodeDecodeError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ) as error:
                st.error(str(error))

        batch_result = st.session_state.get("batch_prediction_result")
        if batch_result and batch_result.get("mode") == mode:
            predictions = batch_result["predictions"]
            summary_col1, summary_col2 = st.columns(2)
            summary_col1.metric("Students predicted", len(predictions))
            summary_col2.metric(
                "Mean estimated Final mark",
                f"{predictions['estimated_final_score'].mean():.1f} / 120",
            )
            st.dataframe(predictions, width="stretch")
            st.download_button(
                "Download predictions",
                data=spreadsheet_safe_csv(predictions),
                file_name=f"final_mark_{mode}_predictions.csv",
                mime="text/csv",
            )

    with evaluation_tab:
        st.subheader("How well did the Final mark estimate perform?")
        st.write(
            "Each dot is a held-out test record. Dots close to the dashed diagonal "
            "are predictions close to the actual course Final mark."
        )
        if EVALUATION_PATH.exists():
            evaluation_mode = st.radio(
                "Evaluation results for",
                ["Coursework only", "After Mid-Sem"],
                horizontal=True,
                key="evaluation_mode",
            )
            evaluation_key = "early" if evaluation_mode == "Coursework only" else "updated"
            evaluation_rows = pd.read_csv(EVALUATION_PATH)
            evaluation_rows = evaluation_rows.loc[
                evaluation_rows["mode"] == evaluation_key
            ]
            evaluation_metrics = metrics["models"][evaluation_key]
            eval_model_col, eval_baseline_col, eval_count_col = st.columns(3)
            eval_model_col.metric("Model MAE", f"{evaluation_metrics['mae']:.2f} marks")
            eval_baseline_col.metric("Mean baseline MAE", f"{evaluation_metrics['baseline_mae']:.2f} marks")
            eval_count_col.metric("Held-out records", evaluation_metrics["test_rows"])
            st.caption(
                f"The model’s average miss is {evaluation_metrics['mae'] / SCORE_MAX * 100:.1f}% of the 0–120 score scale."
            )
            change = evaluation_metrics["mae_change_vs_baseline"]
            if change < 0:
                st.info(
                    f"On this single held-out check of {evaluation_metrics['test_rows']} records, the model’s MAE "
                    f"was {abs(change):.2f} marks lower than the mean baseline. Treat this as a noisy check; "
                    "repeated cross-validation is the primary comparison evidence."
                )
            elif change > 0:
                st.warning(f"The selected model's MAE was {change:.2f} marks higher than the mean baseline on this test split.")
            else:
                st.info("The selected model and mean baseline had the same MAE on this test split.")
            st.markdown(prediction_scatter_svg(evaluation_rows), unsafe_allow_html=True)
            st.caption("The dashed diagonal marks perfect Final mark predictions. Points farther away have larger errors.")
            largest_errors = evaluation_rows.nlargest(10, "absolute_error").rename(
                columns={
                    "dataset_row": "Dataset row",
                    "actual_grade": "Actual Final mark",
                    "predicted_grade": "Predicted Final mark",
                    "absolute_error": "Absolute error",
                }
            )
            st.write("**Largest errors on the held-out test set**")
            st.dataframe(
                largest_errors[["Dataset row", "Actual Final mark", "Predicted Final mark", "Absolute error"]].round(2),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Train the models to generate held-out evaluation results.")

    with history_tab:
        st.subheader("Recent predictions")
        if not LOCAL_HISTORY_ENABLED:
            st.info(
                "Prediction history is off by default. For a private local demo only, set "
                "STUDENT_PREDICTOR_LOCAL_HISTORY=1 before starting Streamlit."
            )
        else:
            st.caption(
                "Stored in a local file on this computer. Marks and timestamps can still identify someone "
                "in a small class, even without names or IDs. The newest 250 records are shown."
            )
            history = get_history(limit=250)
        if LOCAL_HISTORY_ENABLED and history.empty:
            st.info("No predictions saved yet. Make a single or batch prediction to start the history.")
        elif LOCAL_HISTORY_ENABLED:
            history_col1, history_col2 = st.columns(2)
            history_col1.metric("Saved predictions", len(history))
            history_col2.metric("Mean estimated Final mark", f"{history['estimated_grade'].mean():.1f} / 120")
            display_history = history.rename(
                columns={
                    "prediction_id": "Record",
                    "created_at_utc": "Created (UTC)",
                    "mode": "Prediction stage",
                    "model_version": "Model version",
                    "algorithm": "Algorithm",
                    "inputs_json": "Inputs",
                    "estimated_grade": "Estimated Final mark",
                }
            )
            display_history["Prediction stage"] = display_history["Prediction stage"].map(
                {"early": "Coursework only", "updated": "After Mid-Sem"}
            )
            display_history["Inputs"] = display_history["Inputs"].map(
                lambda raw: json.dumps(
                    {
                        FEATURE_LABELS.get(key, key): value
                        for key, value in json.loads(raw).items()
                    },
                    ensure_ascii=False,
                )
            )
            st.dataframe(
                display_history,
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Export history as CSV",
                data=spreadsheet_safe_csv(display_history),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
            if st.button("Clear all prediction history"):
                st.session_state["confirm_clear_history"] = True
            if st.session_state.get("confirm_clear_history", False):
                st.warning("This removes all locally saved prediction records.")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Confirm clear history"):
                    clear_history()
                    st.session_state["confirm_clear_history"] = False
                    st.rerun()
                if cancel_col.button("Keep history"):
                    st.session_state["confirm_clear_history"] = False
                    st.rerun()

if METRICS_PATH.exists():
    with st.expander("For reviewers: model checks and technical details"):
        st.write("**Average error on the held-out test records**")
        early_metrics = metrics["models"]["early"]
        updated_metrics = metrics["models"]["updated"]
        col_early, col_updated = st.columns(2)
        col_early.metric(
            "Coursework-only model",
            early_metrics["selected_algorithm"],
            f"Test MAE {early_metrics['mae']:.2f} points",
            delta_color="off",
        )
        col_updated.metric(
            "After-Mid-Sem model",
            updated_metrics["selected_algorithm"],
            f"Test MAE {updated_metrics['mae']:.2f} points",
            delta_color="off",
        )
        st.caption("Repeated cross-validation is the primary comparison on the training records. The single 20-record holdout is a noisy, secondary check; both stages use the same held-out records.")
        st.write("**Algorithm comparison · repeated cross-validation**")
        st.caption(
            "Mean validation MAE ± standard deviation across 15 folds; lower is better. "
            "Paired Δ ± heuristic SE compares each candidate with the mean baseline on the same folds."
        )
        comparison_rows = []
        for stage_name, model_result in [
            ("Coursework only", early_metrics),
            ("After Mid-Sem", updated_metrics),
        ]:
            for candidate in model_result["candidate_comparison"]:
                comparison_rows.append({
                    "Stage": stage_name,
                    "Algorithm": candidate["name"],
                    "CV MAE": f"{candidate['cv_mae']:.2f} ± {candidate['cv_mae_std']:.2f}",
                    "Paired Δ ± SE vs baseline": (
                        "—" if candidate["name"] == "Mean baseline"
                        else f"{candidate['cv_delta_vs_baseline']:+.2f} ± {candidate['cv_delta_se_vs_baseline']:.2f}"
                    ),
                    "Selected": "Yes" if candidate["selected"] else "",
                })
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
        st.caption(
            "The one-SE rule uses the fold-score SD divided by sqrt(5), rather than treating all 15 repeated "
            "fold scores as independent. This is a heuristic; paired fold differences are descriptive, not a significance test."
        )
        st.write("**Stage-specific model details**")
        for stage_label, stage_metrics in [
            ("Coursework only", early_metrics),
            ("After Mid-Sem", updated_metrics),
        ]:
            with st.expander(stage_label):
                st.write(
                    f"Selected model: **{stage_metrics['selected_algorithm']}** · "
                    f"Held-out R²: **{stage_metrics['r2']:.3f}** · "
                    f"CV MAE: **{next(item['cv_mae'] for item in stage_metrics['candidate_comparison'] if item['selected']):.2f}**"
                )
                if stage_metrics["selected_algorithm"] == "Mean baseline":
                    st.caption("This model returns the training-set average and does not use individual marks.")
                elif "linear_coefficients" in stage_metrics:
                    st.caption("Linear-model weights (descriptive, not causal):")
                    for feature, coefficient in stage_metrics["linear_coefficients"].items():
                        st.write(f"{FEATURE_LABELS.get(feature, feature)}: {coefficient:+.3f} estimated Final marks per input mark")

st.divider()
st.caption("Engineering Student Performance Predictor · Learning project · Estimates are not official academic results.")

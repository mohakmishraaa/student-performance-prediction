# Engineering Student Performance Prediction

A Streamlit learning project that estimates a course Final exam mark from assessment marks already available. It uses a public Chemical Engineering course dataset from McMaster University distributed by OpenMV.net. The dataset contains 99 records, with 96 Final marks available for training and evaluation. It contains one Midterm field, labelled **Mid-Sem** in this app; it does not contain two Mid-Sem exams.

This is a demonstration, not an official grading tool. The data covers one small course and does not establish accuracy for another class, college, or assessment scheme. Check the dataset source terms before redistributing the data.

## Prediction stages

- **After the Mid-Sem** (the default): Assignment, Tutorial, Mid-Sem, and Take-home assessment marks.
- **Coursework marks only:** Assignment and Tutorial marks.
- Input marks and the estimated Final mark use the dataset’s 0–120 scale. Some source marks exceed 100.

The coursework-only model may select the mean baseline. In that case, it returns the same course-average estimate for every input because coursework marks did not improve validation results. The interface labels this as a course average and explains the limitation. A responsive prediction at the later stage is not proof of useful accuracy.

## App features

- One-student prediction form with stage-specific fields and illustrative sample marks. If the Take-home assessment is not available, the model uses the training-set median for that input.
- Batch CSV prediction. Uploads are limited to 5 MB and 1,000 rows. Required score columns are `assignment_score,tutorial_score`; the later stage also uses `midterm_score,takehome_score`. Headers are case/space normalized; bare feature names and `mid-sem` are also accepted. An optional `student_id` is kept as text (including leading zeroes), returned with all marks used and the estimate, and never written to history. Unknown columns are rejected. Downloaded text fields are escaped against spreadsheet formula execution. Uploaded marks are processed in server memory.
- Model check view with a held-out actual-versus-predicted plot and errors. The reviewer section below the tabs contains the cross-validation algorithm comparison and stage-specific model details.
- Local prediction history is off by default. It can be enabled for a private local demo with `STUDENT_PREDICTOR_LOCAL_HISTORY=1`. Names and IDs are not stored, but marks and timestamps may identify someone in a small class. Do not enable shared history or put real student data into this demo.
- CSV download for a single prediction and batch results.

## Run locally

1. Install Python 3.12 to match the Azure training environment.
2. In PowerShell, from this folder, create and activate an environment and install requirements:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Train both stages and generate model metrics:

   ```powershell
   python -m src.train
   ```

   The bundled CSV is used. If it is missing, the loader downloads the public OpenMV file.

4. Start the app:

   ```powershell
   streamlit run app.py
   ```

   Prediction history is disabled by default. To enable it for a private local demo, set `$env:STUDENT_PREDICTOR_LOCAL_HISTORY = "1"` in PowerShell before starting Streamlit. Do not enable this setting on a shared host.

## Demo walkthrough

1. Open **Try one student**. The app defaults to **After the Mid-Sem**.
2. Click **Load typical sample**, or open **Try another sample** for lower/higher examples.
3. Click **3. Estimate final mark**.
4. Open **How should I read this estimate?** for held-out average-error context.
5. Open **Model check** to view the held-out plot; scroll to **For reviewers** for algorithm comparison and model details.

Sample values are illustrative and are not training records. They do not measure model accuracy.

## Model training and evaluation

`src/train.py` removes records without a Final mark, makes one fixed 80/20 train/test split, then uses repeated 5-fold cross-validation (3 repeats) on the training portion to compare a mean baseline, Linear Regression, Decision Tree, and Random Forest. Ridge was removed because with these unscaled marks and alpha it duplicated Linear Regression. Preprocessing is contained inside the cross-validation pipeline. The app uses a one-standard-error simplicity rule to prefer a simpler model when candidate scores are close. Since the 15 scores from repeated CV are correlated, the heuristic uses the fold-score standard deviation divided by `sqrt(5)`, not `sqrt(15)`. This is not an exact uncertainty estimate or formal significance test. Paired cross-validation deltas compare each candidate with the mean baseline on matching folds; their displayed standard errors use the same `sqrt(5)` heuristic and are descriptive, not significance tests.

`requirements.txt` and `azure/conda.yml` pin the same NumPy, pandas, scikit-learn, and joblib versions so an Azure-trained artifact is less likely to encounter library-version differences when loaded locally.

With the current dataset, there are 76 records in the training split and 20 in the held-out split. Repeated cross-validation on the 76 training records is the primary model comparison; the single 20-record holdout is a noisy secondary check that can vary with the split. After recording that independent evaluation, the saved app model is refit on all 96 labeled records. The evaluation predictions still come only from models fitted on the 76 training records. Neither the group-level MAE nor cross-validation results are confidence intervals for an individual student, and the model is not validated outside this one dataset.

In the generated metrics for the bundled data, coursework-only selection chooses the mean baseline (CV MAE 16.81; held-out MAE 12.885, equal to the held-out baseline). After the Mid-Sem, Linear Regression is selected (CV MAE 10.33; paired change from baseline −6.47 ± 1.02 marks using the stated heuristic). On the fixed held-out split, its MAE is 12.154 versus 12.885 for the mean baseline, with R² 0.027. This one small holdout is noisy and is secondary to the repeated-CV comparison; results still need validation on other courses before real use. Both stages use the same held-out target rows. The selected model and generated metrics can change if the selection rule or data changes.

## Azure Machine Learning training job

`azure/job.yml` submits the existing training code as an Azure Machine Learning command job. It uploads the public CSV as job input and writes `model.joblib`, `metrics.json`, and `evaluation_predictions.csv` to the `model_output` artifact. It expects an existing Azure ML workspace and a compute target named `cpu-cluster`. The compute should scale to zero while idle; Azure compute can incur charges while running.

This job trains a model in Azure and downloads it back to the local app. It does **not** host this Streamlit app, create a real-time endpoint, or make the app publicly accessible. Azure ML CLI job submission requires a subscription, resource group, workspace, CLI/ML extension, and available compute. See the [Azure ML CLI job documentation](https://learn.microsoft.com/cli/azure/ml/job) and [command-job YAML schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2).

The `scripts/import_azure_model.ps1` script downloads the selected job’s `model_output`, checks that the expected artifact files exist, and copies them into `outputs/`. Only import artifacts from a job you own or trust: `joblib` model files use Python pickle serialization.

## Hosting the Streamlit app

There is no Dockerfile, Azure App Service configuration, or container deployment configuration in this project yet. A hosted app is separate from the Azure ML training job. Streamlit does not provide authentication by default.

History is disabled unless explicitly enabled with `STUDENT_PREDICTOR_LOCAL_HISTORY=1`, and `STUDENT_PREDICTOR_PUBLIC=1` always disables it. Do not enable local history on a shared host. Uploaded marks are processed in server memory and are not written to history when history is off, but the app has no sign-in or access control; configure Azure authentication or restrict access separately before using any real student records. The current app is intended for illustrative data only. Review the selected hosting service’s current pricing before creating resources.

## Data source

OpenMV.net, [Class grades](https://openmv.net/info/class-grades), “Grades from a Chemical Engineering course at McMaster University,” 99 records. The source describes Assignment, Tutorial, Midterm, Take-home, and Final marks. Original CSV: <https://openmv.net/file/class-grades.csv>.

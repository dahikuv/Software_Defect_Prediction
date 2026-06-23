# Explainable Software Defect Prediction

Prototype codebase for the project **Explainable Software Defect Prediction Using Machine Learning and Commit Message Analysis**.

## Project Context
This project builds a reproducible baseline pipeline for software defect prediction using machine learning. The system handles the entire workflow from data preprocessing, class imbalance handling (e.g., SMOTE over-sampling), to training and evaluating core algorithms (Decision Trees, Random Forest, XGBoost, LightGBM). The goal is to produce highly modular source code that is easy to reuse and extend.

## Goals
- Build a reproducible data pipeline for software defect datasets.
- Engineer features from software metrics and commit messages.
- Train and compare Random Forest, XGBoost, and LightGBM.
- Explain predictions with SHAP.
- Provide a Streamlit demo for inspection and presentation.

## Architecture
```
prototype/
├── data/            # Raw input data and processed data. Do not commit to git.
├── notebooks/       # Exploratory notebooks.
├── src/             # Core source code (main logic).
│   ├── config/      # Configuration files (config.yaml).
│   ├── data/        # Ingestion, cleaning, schema unification, and validation.
│   ├── features/    # Feature engineering (metrics, commit text, hybrid).
│   ├── models/      # Model definitions, training, loading, and prediction.
│   ├── evaluation/  # Metrics, comparison, significance testing, error analysis.
│   ├── explainability/ # SHAP global and local explanation helpers.
│   ├── app/         # Streamlit demo application and service layer.
│   └── utils/       # Shared utilities (paths, I/O, logging, provenance, seed).
├── scripts/         # Standalone entry-point scripts for each pipeline stage.
├── tests/           # Unit and integration tests.
├── models/          # Trained model artifacts (joblib). Do not commit.
└── results/         # Evaluation outputs, figures, and tables. Do not commit.
```

## Setup
### 1. Create environment
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r prototype/requirements.txt
```

On Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run the pipeline
```bash
python prototype/scripts/run_data_pipeline.py
python prototype/scripts/run_feature_pipeline.py
python prototype/scripts/run_train_metrics_only.py
python prototype/scripts/run_evaluation.py
python prototype/scripts/run_shap.py
python prototype/scripts/run_error_analysis.py
```

### 3. Run the demo app
```bash
streamlit run prototype/src/app/streamlit_app.py
```

## Current Scope
The repository is focused on the **metrics-based baseline** first, using the `jm1`, `kc1`, `cm1`, and `pc1` datasets from the PROMISE repository. The stable baseline uses `loc`, `v(g)`, `ev(g)`, `iv(g)`, and `branchCount` as primary features.

The foundation covers:
- Data ingestion, cleaning, schema unification, and validation.
- Feature engineering from software metrics and commit messages (TF-IDF / SBERT).
- Model training with hyperparameter tuning (Random Forest, XGBoost, LightGBM).
- Model evaluation with significance testing and error analysis.
- SHAP-based global and local explainability.
- Streamlit demo with saved artifacts.

## Engineering Rules
- Prefer existing project helpers in `prototype/src/utils/paths.py` and `prototype/src/utils/io.py`.
- Preserve artifact provenance fields (dataset name, feature family, model path, split manifest).
- Use type hints and docstrings on all public functions.
- Separate data processing logic from training algorithms.
- Validate inputs before entering resource-intensive loops.
- Use `try-except` for file I/O and data transformation steps.

## Notes
- Keep experiment settings in `prototype/src/config/config.yaml`.
- Do not commit generated caches, logs, model binaries, raw datasets, or other local artifacts.
- Run `python -m compileall -q prototype\src prototype\scripts` to verify syntax before pushing.
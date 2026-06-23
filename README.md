# Explainable Software Defect Prediction

Prototype codebase for the project **Explainable Software Defect Prediction Using Machine Learning and Commit Message Analysis**.

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

## Prerequisites
- Python 3.10+
- Git

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/dahikuv/Software_Defect_Prediction.git
cd Software_Defect_Prediction
```

### 2. Create and activate environment
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r prototype/requirements.txt
```

### 3. Prepare datasets
The pipeline expects software defect datasets in CSV format. Place your files in `prototype/data/raw/` directory. The primary datasets used in this project are from the PROMISE repository:
- `jm1.csv`
- `kc1.csv`
- `cm1.csv`
- `pc1.csv`

These should be placed in a subdirectory like `prototype/data/raw/Promise + BPD/` (or update `prototype/src/config/config.yaml` to match your file locations).

### 4. Run the pipeline
```bash
# Data preprocessing
python prototype/scripts/run_data_pipeline.py

# Feature engineering
python prototype/scripts/run_feature_pipeline.py

# Model training
python prototype/scripts/run_train_metrics_only.py

# Evaluation
python prototype/scripts/run_evaluation.py

# Explainability
python prototype/scripts/run_shap.py

# Error analysis
python prototype/scripts/run_error_analysis.py
```

### 5. Run the demo app
```bash
streamlit run prototype/src/app/streamlit_app.py
```
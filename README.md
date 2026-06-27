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
├── data/
│   └── raw/Promise + BPD/   # NASA-PROMISE datasets (CM1, JM1, KC1, PC1). Committed.
├── src/                      # Core source code (main logic).
│   ├── config/               # Configuration files (config.yaml).
│   ├── data/                 # Ingestion, cleaning, schema unification, and validation.
│   ├── features/             # Feature engineering (metrics, commit text, hybrid).
│   ├── models/               # Model definitions, training, loading, and prediction.
│   ├── evaluation/           # Metrics, comparison, significance testing, error analysis.
│   ├── explainability/       # SHAP global and local explanation helpers.
│   ├── app/                  # Streamlit demo application and service layer.
│   └── utils/                # Shared utilities (paths, I/O, logging, provenance, seed).
├── scripts/                  # Standalone entry-point scripts for each pipeline stage.
├── tests/                    # Unit and integration tests.
├── models/                   # Trained model artifacts (joblib). Do not commit.
└── results/
    └── tables/               # Evaluation result tables (CSV, JSON). Committed.
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

### 3. Included Datasets

NASA-PROMISE module-level datasets are included in `prototype/data/raw/Promise + BPD/`:

| Dataset | Instances | Defects | Defect % | Imbalance Ratio |
|---------|-----------|---------|----------|-----------------|
| CM1     | 498       | 49      | 9.8%     | 1:9.2           |
| JM1     | 13,204    | 2,103   | 15.9%    | 1:5.3           |
| KC1     | 2,109     | 326     | 15.5%    | 1:5.5           |
| PC1     | 1,109     | 77      | 6.9%     | 1:13.4          |

Commit-level JIT datasets (JITFine, OpenStack, Qt) are not included due to licensing. See the original dataset releases for access.

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

## Scripts Reference

### Core Pipeline

| Script | Purpose |
|--------|---------|
| `run_data_pipeline.py` | Full data ingestion and preprocessing |
| `run_feature_pipeline.py` | Feature engineering (metrics + TF-IDF) |
| `run_train_metrics_only.py` | Train models on metrics-only features |
| `run_train_hybrid_tfidf.py` | Train models on hybrid (metrics + TF-IDF) features |
| `run_train_tuned_metrics.py` | Train with hyperparameter tuning |
| `run_evaluation.py` | Evaluate models and select best configuration |
| `run_shap.py` | Generate SHAP explanations |
| `run_error_analysis.py` | Analyze false positives and negatives |
| `run_split_datasets.py` | Create train/val/test splits |
| `run_app.py` | Launch the Streamlit demo |

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `run_ablation_full.py` | Ablation study: Metrics-only vs TF-IDF-only vs Hybrid |
| `run_ablation_tfidf_only.py` | TF-IDF-only ablation on JIT datasets |
| `run_hyperparam_comparison.py` | Compare default vs tuned hyperparameters |
| `run_repeated_cv.py` | Repeated 10-fold stratified cross-validation |
| `run_shap_extra_plots.py` | Additional SHAP plots (beeswarm, waterfall, dependence) |
| `run_stat_table.py` | Statistical significance tests (permutation, DeLong) |

### Ingestion

| Script | Purpose |
|--------|---------|
| `run_ingest_jitfine.py` | Ingest JITFine dataset |
| `run_ingest_jitline.py` | Ingest JITLine dataset |
| `run_experiment_datasets.py` | Prepare experiment datasets |
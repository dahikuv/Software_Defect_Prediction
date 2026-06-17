# Software Defect Prediction Prototype

Python prototype for explainable software defect prediction using software metrics,
commit-message TF-IDF features, model comparison, SHAP explainability, and a
Streamlit demo.

Main code lives in `prototype/`.

## Current Final Selection

The current evaluation policy is configured in
`prototype/src/config/config.yaml`:

- `evaluation.selection_policy`: `hybrid_validation_then_tuned`
- Hybrid datasets can be trained for `ghpr`, `openstack`, `qt`, and `jitfine`
- `ghpr` is excluded from final model selection through
  `features.hybrid.final_selection_excluded`

Current final selected models:

| Dataset | Model | Mode |
| --- | --- | --- |
| cm1 | xgb | tuned metrics |
| jm1 | xgb | tuned metrics |
| kc1 | xgb | tuned metrics |
| pc1 | rf | tuned metrics |
| jitfine | rf | hybrid TF-IDF |
| openstack | lgbm | hybrid TF-IDF |
| qt | xgb | hybrid TF-IDF |

Selection uses validation/CV signals for ranking. Test metrics are kept for
reporting only.

## Useful Commands

Run from the repository root:

```powershell
python -m compileall -q prototype\src prototype\scripts
python -m pytest prototype\tests -q
python prototype\scripts\run_evaluation.py
python prototype\scripts\run_shap.py
python prototype\scripts\run_error_analysis.py
streamlit run prototype\src\app\streamlit_app.py
```

Latest verification in this workspace:

```text
python -m compileall -q prototype\src prototype\scripts
python -m pytest prototype\tests -q
70 passed
git diff --check
```

`git diff --check` currently reports only Git LF/CRLF warnings.

## Artifact Policy

Generated artifacts are local research outputs and should be staged only
deliberately.

Keep code/config/tests/docs as normal source changes. Keep small CSV/JSON/PNG
result artifacts only when they are needed for the paper or demo.

Do not commit raw datasets, downloaded replication packages, model binaries,
cache folders, logs, PID files, or local AI tooling directories.

Important local artifact locations:

- `prototype/data/raw/`
- `prototype/data/interim/`
- `prototype/data/processed/`
- `prototype/data/splits/`
- `prototype/models/`
- `prototype/results/`

## Project Layout

- `prototype/src/data/`: ingestion, cleaning, validation, and splitting
- `prototype/src/features/`: metrics, TF-IDF/SBERT, and feature merging
- `prototype/src/models/`: training, model bundles, loading, and prediction
- `prototype/src/evaluation/`: ranking, final selection, impact, and error analysis
- `prototype/src/explainability/`: SHAP global and local helpers
- `prototype/src/app/`: Streamlit app and service/controller layer
- `prototype/scripts/`: runnable pipeline entry points
- `prototype/tests/`: artifact, integration, data, feature, and training tests

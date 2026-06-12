"""Model registry for baseline tabular classifiers."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


def get_model(name: str, random_state: int = 42):
    """Return a model instance by short name."""
    name = name.lower()
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed")
        return XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
    if name == "lgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed")
        return LGBMClassifier(
            random_state=random_state,
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            verbosity=-1,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model name: {name}")

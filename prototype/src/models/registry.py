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
            n_estimators=300,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed")
        return XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            n_jobs=-1,
        )
    if name == "lgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed")
        return LGBMClassifier(
            random_state=random_state,
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            verbosity=-1,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model name: {name}")

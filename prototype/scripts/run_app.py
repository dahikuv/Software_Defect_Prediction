"""Helper entry point for launching the Streamlit MVC app."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Print the command used to launch the Streamlit MVC app."""
    logger.info("Run the MVC app with: streamlit run src/app/streamlit_app.py")
    logger.info("The app reads saved artifacts from results/tables, results/figures/shap, and models/metrics.")


if __name__ == "__main__":
    main()

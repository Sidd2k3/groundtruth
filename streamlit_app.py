"""Entry point for Streamlit Community Cloud: runs the whole app in one process with the sample corpus preloaded."""
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("SEED_DIR", str(ROOT / "data" / "corpus"))
runpy.run_path(str(ROOT / "ui" / "app.py"), run_name="__main__")

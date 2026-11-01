"""Launch the Streamlit app."""

import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "src" / "wds_optimizer" / "app" / "streamlit_app.py"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP)],
        check=True,
    )
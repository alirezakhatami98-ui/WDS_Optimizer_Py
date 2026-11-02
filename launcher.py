"""WDS Optimizer desktop launcher.

Frozen (PyInstaller) -> runs bundled Streamlit app + auto-opens browser.
Script mode          -> same behavior, useful for `python launcher.py`.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path


def _resource_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _bundled_app_path() -> Path:
    base = _resource_dir()
    candidates = [
        base / "wds_optimizer" / "app" / "streamlit_app.py",
        base / "src" / "wds_optimizer" / "app" / "streamlit_app.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot locate streamlit_app.py. Searched: {candidates}"
    )


def _write_streamlit_config(st_home: Path) -> None:
    st_home.mkdir(parents=True, exist_ok=True)
    (st_home / "config.toml").write_text(
        "[global]\n"
        "developmentMode = false\n"
        "\n"
        "[server]\n"
        "headless = true\n"
        "port = 8501\n"
        "address = 'localhost'\n"
        "runOnSave = false\n"
        "fileWatcherType = 'none'\n"
        "\n"
        "[browser]\n"
        "gatherUsageStats = false\n"
        "serverAddress = 'localhost'\n"
        "serverPort = 8501\n",
        encoding="utf-8",
    )


def _open_browser_when_ready(url: str, timeout_s: float = 30.0) -> None:
    """Poll TCP until Streamlit port is open, then open browser."""
    import socket

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 8501), timeout=1.0):
                webbrowser.open(url, new=2)
                return
        except OSError:
            time.sleep(0.5)
    # Fallback: open anyway after timeout (may show connection error
    # but user can refresh once Streamlit finishes booting).
    webbrowser.open(url, new=2)


def main() -> None:
    # ---- Silence the benign WinError 10054 traceback ----
    import logging

    class _SuppressWinError10054(logging.Filter):
        def filter(self, record):
            if record.exc_info and record.exc_info[0] is ConnectionResetError:
                return False
            if "WinError 10054" in record.getMessage():
                return False
            return True

    logging.getLogger("asyncio").addFilter(_SuppressWinError10054())

    # ---- Streamlit config ----
    st_home = Path(tempfile.gettempdir()) / "wds_optimizer_streamlit"
    _write_streamlit_config(st_home)

    os.environ["STREAMLIT_HOME"] = str(st_home)
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    # ---- Auto-open browser in background ----
    threading.Thread(
        target=_open_browser_when_ready,
        args=("http://localhost:8501",),
        daemon=True,
    ).start()

    # ---- Run Streamlit (no argv port flags — see config.toml) ----
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(_bundled_app_path())]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
"""WDS Optimizer desktop launcher.

When frozen (PyInstaller), runs the bundled Streamlit app; when run as
a script, behaves like `streamlit run src/wds_optimizer/app/streamlit_app.py`.
"""

from __future__ import annotations

import os
import sys
import tempfile
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
    """Write a config.toml that disables developmentMode.

    Streamlit is bundled with PyInstaller in a way that makes it *think*
    it is running in development mode, which then rejects any argv-based
    ``--server.port``.  Putting all options in a config.toml (and using
    no port-related argv flags) avoids the conflict.
    """
    st_home.mkdir(parents=True, exist_ok=True)
    cfg = st_home / "config.toml"
    cfg.write_text(
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

    # ---- Streamlit environment (no port-related argv flags!) ----
    st_home = Path(tempfile.gettempdir()) / "wds_optimizer_streamlit"
    _write_streamlit_config(st_home)

    os.environ["STREAMLIT_HOME"] = str(st_home)
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    # ---- Run Streamlit ----
    from streamlit.web import cli as stcli

    app_path = str(_bundled_app_path())

    # IMPORTANT: do NOT pass --server.port / --server.address here.
    # They are already set via config.toml + environment variables.
    # Passing them as argv triggers the dev-mode conflict again.
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
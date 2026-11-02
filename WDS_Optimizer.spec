# WDS_Optimizer.spec
# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for WDS Optimizer (onedir mode)."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# ---- Streamlit ----
st_datas, st_binaries, st_hidden = collect_all("streamlit")

# ---- Plotly (JS assets) ----
pl_datas, pl_binaries, pl_hidden = collect_all("plotly")

# ---- WNTR (bundled libepanet + .inp examples) ----
wntr_datas, wntr_binaries, wntr_hidden = collect_all("wntr")

# ---- matplotlib (fonts, mpl-data, style) ----
mpl_datas, mpl_binaries, mpl_hidden = collect_all("matplotlib")

# ---- pandas / numpy metadata ----
pd_datas = collect_data_files("pandas")
np_datas = collect_data_files("numpy")

# ---- Our package's runtime data ----
here = Path(SPECPATH).resolve()
our_datas = [
    # Streamlit entry script and all source modules (Streamlit imports
    # submodules at runtime, so we need the whole src tree).
    (str(here / "src" / "wds_optimizer"), "wds_optimizer"),
    # Bundled benchmark networks
    (str(here / "networks"), "networks"),
]

hiddenimports = (
    st_hidden
    + pl_hidden
    + wntr_hidden
    + mpl_hidden
    + [
        # Streamlit loads these dynamically
        "streamlit.web.cli",
        "streamlit.web.bootstrap",
        "streamlit.runtime.scriptrunner.magic_funcs",
        # Our package
        "wds_optimizer",
        "wds_optimizer.app.streamlit_app",
        "wds_optimizer.app.state",
        "wds_optimizer.app.network_map",
        "wds_optimizer.app.compare",
        "wds_optimizer.algorithms.ga",
        "wds_optimizer.algorithms.pso",
        "wds_optimizer.algorithms.config",
        "wds_optimizer.algorithms.selection",
        "wds_optimizer.algorithms.compare",
        "wds_optimizer.experiments.runner",
        "wds_optimizer.hydraulics.simulation",
        "wds_optimizer.hydraulics.units",
        "wds_optimizer.optimization.problem",
        "wds_optimizer.optimization.evaluate",
        "wds_optimizer.optimization.cost",
        "wds_optimizer.optimization.constraints",
        "wds_optimizer.optimization.fitness",
        "wds_optimizer.data.loaders",
        "wds_optimizer.sweep.config",
        "wds_optimizer.sweep.runner",
        "wds_optimizer.persistence.snapshot",
        "wds_optimizer.reporting.report",
        "wds_optimizer.reporting.figures",
    ]
)

a = Analysis(
    ["launcher.py"],
    pathex=[str(here / "src")],
    binaries=st_binaries + pl_binaries + wntr_binaries + mpl_binaries,
    datas=st_datas + pl_datas + wntr_datas + mpl_datas + pd_datas + np_datas + our_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Keep the bundle smaller
        "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "IPython", "jupyter", "notebook",
        "pytest", "sphinx",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WDS_Optimizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # keep console so user sees logs if crash
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WDS_Optimizer",   # output folder
)
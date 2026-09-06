# PyInstaller spec for local Linux CPU and CUDA candidates.
# Build on Linux only: pyinstaller packaging/planetrecon-linux.spec
# Qt platform plugins ship with both bundles. PLANETRECON_BUNDLE_GPU=1
# includes Torch and CUDA libraries; the default CPU bundle excludes Torch.

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

project_root = str(Path(SPECPATH).parent)
gpu_bundle = os.environ.get("PLANETRECON_BUNDLE_GPU") == "1"
bundle_name = "planetrecon-gpu" if gpu_bundle else "planetrecon"
a = Analysis(
    [str(Path(project_root) / "planetrecon" / "__main__.py")],
    pathex=[project_root],
    binaries=[],
    datas=[],
    hiddenimports=["PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtCore", "scipy._cyutility", "tifffile"]
    + collect_submodules("numpy._core", filter=lambda name: ".tests" not in name)
    + (["torch"] if gpu_bundle else []),
    hookspath=[],
    runtime_hooks=[],
    excludes=[] if gpu_bundle else ["torch"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=bundle_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, name=bundle_name)

# PyInstaller spec for an early Linux packaging spike (W08).
# Build on Linux only: pyinstaller packaging/planetrecon-linux.spec
# Qt platform plugins must ship with the bundle. Do not include CUDA torch
# in the CPU spike; optional GPU builds are a later W12 artifact.

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

project_root = str(Path(SPECPATH).parent)
block_cipher = None
a = Analysis(
    [str(Path(project_root) / "planetrecon" / "__main__.py")],
    pathex=[project_root],
    binaries=[],
    datas=[],
    hiddenimports=["PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtCore", "scipy._cyutility"]
    + collect_submodules("numpy._core", filter=lambda name: ".tests" not in name),
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="planetrecon",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="planetrecon")

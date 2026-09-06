# Native PyInstaller CPU bundles for Linux, Windows and macOS; CUDA on Linux.
# Qt platform plugins ship with both bundles. PLANETRECON_BUNDLE_GPU=1
# includes Torch and CUDA libraries; the default CPU bundle excludes Torch.

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

project_root = str(Path(SPECPATH).parent)
gpu_bundle = os.environ.get("PLANETRECON_BUNDLE_GPU") == "1"
bundle_name = "planetrecon-gpu" if gpu_bundle else "planetrecon"
if gpu_bundle and sys.platform != 'linux':
    raise ValueError('The locked CUDA bundle is currently Linux-only')
a = Analysis(
    [str(Path(project_root) / "planetrecon" / "__main__.py")],
    pathex=[project_root],
    binaries=[],
    datas=[],
    hiddenimports=["PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtCore", "scipy._cyutility", "tifffile"]
    + collect_submodules("numpy._core", filter=lambda name: ".tests" not in name)
    + (["torch"] if gpu_bundle else [])
    + (["planetrecon._build_version"] if (Path(project_root)/'planetrecon/_build_version.py').exists() else []),
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
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='PlanetRecon.app', bundle_identifier='org.planetrecon.desktop',
                 version=os.environ.get('PLANETRECON_BUILD_VERSION', '0.1.0').split('-')[0],
                 info_plist={'NSHighResolutionCapable': True,
                             'PlanetReconVersion': os.environ.get('PLANETRECON_BUILD_VERSION', '0.1.0')})

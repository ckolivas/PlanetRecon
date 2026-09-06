The current deliverables are unsigned Linux CPU and CUDA candidates. Windows/macOS builds,
clean offline acceptance, other accelerators and signing/notarization have not
been qualified. The local RTX 5070 baseline has passed parity and recovery checks. The project release license has not been selected. Local smoke
results do not change those statuses.

The recorded local checks are in `results/releases/linux-local.json`, including
executable and inventory hashes, actual CUDA parity, and CPU/GPU GUI cancellation,
restart and save results. Inventories and bundled notices are under `out/release`
and each bundle's `licenses` directory. The CPU bundle is approximately 321 MiB;
the GPU bundle is approximately 4.4 GiB.

Build on the recorded Linux toolchain:

```sh
python3 tools/setup_venv.py --gpu
.venv/bin/python packaging/release.py check-toolchain
.venv/bin/python -m PyInstaller --noconfirm --distpath out/dist/cpu --workpath out/build/cpu packaging/planetrecon-linux.spec
PLANETRECON_BUNDLE_GPU=1 .venv/bin/python -m PyInstaller --noconfirm --distpath out/dist/gpu --workpath out/build/gpu packaging/planetrecon-linux.spec
QT_QPA_PLATFORM=offscreen out/dist/cpu/planetrecon/planetrecon gui-smoke --out out/validation/cpu-gui
QT_QPA_PLATFORM=offscreen out/dist/gpu/planetrecon-gpu/planetrecon-gpu gui-smoke --device gpu --out out/validation/gpu-gui
.venv/bin/python packaging/smoke_export.py out/dist/cpu/planetrecon/planetrecon
.venv/bin/python packaging/smoke_capture.py out/dist/cpu/planetrecon/planetrecon
.venv/bin/python packaging/release.py inventory --bundle out/dist/cpu/planetrecon --out out/release/cpu --toc out/build/cpu/planetrecon-linux/Analysis-00.toc
.venv/bin/python packaging/release.py verify --bundle out/dist/cpu/planetrecon --manifest out/release/cpu/manifest.json
```

Repeat the export/capture/inventory/verify commands for the GPU bundle using
`out/dist/gpu/planetrecon-gpu` and the GPU build TOC. The entire bundle directory
must travel together. The CPU bundle excludes Torch; the GPU bundle includes
Torch and the CUDA user-space libraries. A compatible NVIDIA host driver is
still required for CUDA. Both include the Python interpreter, Qt and numerical /
image dependencies; no installed Python or virtual environment is needed at run
time. Running a bundled executable without arguments opens the GUI.


`toolchain-linux.json` pins the observed Python and installed Python package
versions; it is checked before building. It is not a cross-platform wheel lock or
a complete OS dependency lock. The inventory records exact source file hashes,
base Git revision, bundled file checksums, symlink targets, a CycloneDX file SBOM
and copyright notices found from the build TOC's Debian packages and Python
metadata. Verify after copying the bundle. Missing/changed/extra files and links
that escape the bundle are errors. A complete redistribution review, project
license and target-specific signing remain prerequisites for a public release.

Native AVI decoding and TIFF/PNG encoding are bundled. The native capture smoke
sets PATH to a nonexistent directory to catch accidental external decoder use.
The Linux development host still supplies its OS/graphics stack; this is not a
clean-machine or cross-platform acceptance claim.

For local use, unpack/copy the entire `planetrecon` directory and run
`./planetrecon --threads 2 gui`. Choose CPU, open a SER or supported AVI, inspect
its metadata and select a Bayer override only when known. Configure calibration
and physical geometry only from justified values, then Run. The live image is a
bounded preview; zoom/levels do not modify the full scientific result. Magenta
marks invalid coverage. Save float TIFF to preserve scale, or set explicit shared
black/white levels for integer PNG/TIFF. Cancel processing retains the last
received result. Use the CLI's separate `--state-checkpoint`/`--resume` flags for
exact CPU translation continuation; GUI geometry resume is not implemented.

The distro Torch lacks sm_120, but the local venv and GPU bundle use
Torch 2.13.0+cu132 and support this RTX 5070. The advanced atmospheric solver
remains experimental and Q3 closed. Geometry remains on CPU.

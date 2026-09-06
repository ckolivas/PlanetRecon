The native release workflow builds Linux x64 CPU/CUDA, Windows x64 CPU and macOS
Intel/Apple Silicon CPU bundles. Windows/macOS runtime testing is excluded at the
user's request. Linux runs the CPU regression suite and frozen GUI, checkpoint,
encoder and capture checks. CUDA hardware checks run locally on the RTX 5070;
GitHub's ordinary hosted runners do not provide that GPU.

Push a version tag such as `v1.2.3` or `v1.2.3-rc.1` to trigger
[the release workflow](../.github/workflows/release.yml). The accepted prerelease
suffixes are `alpha.N`, `beta.N` and `rc.N`. All five targets must build and their
revision, version, inventories, checksums and complete archive hashes must agree
before publication. Upload failures leave a draft. Prerelease tags publish as
prereleases. A manual Actions run builds downloadable artifacts without
publishing. No tag or public release is created by a local build.

Each bundle contains Python, Qt, numerical/image components, third-party notices
and a quick-start guide. CPU bundles exclude Torch; the Linux CUDA bundle adds
Torch and CUDA user-space libraries with sm_120 support. Keep the entire extracted
bundle together. The host still supplies its OS graphics stack and, for CUDA, a
compatible NVIDIA driver. No installed Python or virtual environment is needed
at runtime. Run the executable without arguments to open the GUI.

The build matrix and native runner requirements are committed in
`targets.json` and the workflow. Windows uses the hosted runner's native tools;
macOS uses its Xcode command-line tools and `codesign`; Linux installs the listed
binutils and Qt graphics dependencies. These are native builds, not cross builds.
Python is pinned to 3.13.5. `requirements/build-native.lock` pins compatible binary
wheels, including Windows PE and macOS Mach-O tooling. CUDA's pure-Python and
native components are pinned separately in `build-gpu-python.lock` and
`gpu-cu132.lock`. `test.lock` pins the regression runner. Compatible Windows/macOS
wheel availability was checked; runtime behavior on those systems is untested.

For a native CPU build from a Python 3.13.5 environment:

```sh
python -m venv .venv
# Activate .venv using your platform's shell, then:
python -m pip install --only-binary=:all: -r requirements/build-native.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python packaging/build_native.py --tag v0.1.0 --flavor cpu --out out/native-cpu
```

For the Linux CUDA build, additionally install the pinned components before
running the same builder with `--flavor gpu` and a different output directory:

```sh
python -m pip install -r requirements/build-gpu-python.lock
python -m pip install -r requirements/gpu-cu132.lock
python packaging/build_native.py --tag v0.1.0 --flavor gpu --out out/native-gpu
```

Use a new output directory for each build. Builds temporarily stamp the tag's
version into the executable without editing the project version. The existing
`tools/setup_venv.py --gpu` remains the local Devuan development setup; its observed
Linux toolchain lock is distinct from the portable release requirements.

Assets appear in the output's `assets` directory: ZIP on Windows, tar.gz on
Linux/macOS, bundle inventory, CycloneDX SBOM, release descriptor and SHA-256 list.
Archives that exceed GitHub's per-file limit are split into numbered 1 GiB parts.
Download every part, check the part hashes and concatenate in numeric order:

```sh
sha256sum -c planetrecon-VERSION-linux-x86_64-gpu.sha256
cat planetrecon-VERSION-linux-x86_64-gpu.tar.gz.part* > planetrecon-gpu.tar.gz
```

The descriptor records the reassembled archive's SHA-256 and size. Extract it with
`tar -xzf planetrecon-gpu.tar.gz`. The publisher independently verifies both the
parts and their reassembled stream before uploading. macOS archives preserve Qt
framework symlinks. Inventories reject broken, cyclic or escaping links and
record exact source hashes, base revision and bundle contents. To verify an
extracted bundle, use `packaging/release.py verify --bundle PATH --manifest FILE`.

Current local candidates are in `out/native-release-cpu/dist/planetrecon` and
`out/native-release-gpu/dist/planetrecon-gpu`; their version is 0.1.0 for local
validation, without a corresponding release tag. Dated results in
`results/releases/native-local-2026-09-07.json` supersede the older
`linux-local.json` report. Both candidates passed Linux frozen GUI checkpoint
resume, export and native AVI/geometry checks. Real CUDA also passed GUI
cancellation/restart, CPU parity, completed-state resume and allocation-limit
recovery with Python/venv search paths disabled.

The GUI displays Bayer data with nearest-neighbour RGB. IR642 and L3 Mars are
both RGGB OSC inputs. Reconstruction consumes the raw CFA samples. Choose
calibration/geometry only from known capture information. Checkpoint and Resume
controls support translation on CPU/CUDA and geometry/Saturn on CPU with matching
input and configuration. CPU memory controls enforce a Linux process address-space
ceiling including mapped libraries; the GUI parent is excluded. CUDA's separate
allocator ceiling excludes driver and external-library allocations.

These are unsigned candidates; macOS uses ad-hoc signing, without publisher
signing or notarization. No project license has been selected or invented.
Bundled third-party notices do not constitute a completed redistribution review.
The advanced atmospheric solver remains experimental and Q3 is unqualified;
these baseline build artifacts do not assert full scientific acceptance.

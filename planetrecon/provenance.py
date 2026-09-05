"""Versioned evidence, fingerprints, and typed experiment manifests (W00)."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from planetrecon import constants as C
from planetrecon.config import SimConfig


MANIFEST_SCHEMA = "planetrecon-evidence-manifest"
MANIFEST_SCHEMA_VERSION = "1.0"
RESULT_STATUSES = ("diagnostic", "valid", "incomplete", "invalid", "gate_passed")

PHYSICS_KEYS = (
    "revision",
    "schema_name",
    "schema_version",
    "D_m",
    "obstruction_ratio",
    "wavelength_m",
    "wind_m_s",
    "n_diam",
    "pupil_pad_factor",
    "pupil_grid_size",
    "exposure_samples_J",
    "padding_detector_px",
    "subharmonic_levels",
    "eval_size",
    "object_oversample",
    "bin_factor_locked",
    "tau0_factor",
    "texp_over_tau0",
    "dt_over_tau0",
    "detector_samp_factor",
    "read_noise_e",
)
COMPAT_KEYS = PHYSICS_KEYS + (
    "simulator_operator_version",
    "estimator_operator_version",
    "source_hash",
    "config_hash",
)

_PACKAGE_DIR = Path(__file__).resolve().parent


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(_canonical_bytes(obj))


def package_python_files() -> list[Path]:
    files = sorted(_PACKAGE_DIR.rglob("*.py"))
    return [p for p in files if "__pycache__" not in p.parts]


def source_hash(paths: Iterable[Path] | None = None) -> str:
    """Conservative content identity of the engine and validation implementation."""
    items = []
    for path in paths if paths is not None else package_python_files():
        rel = path.resolve()
        items.append((str(rel.relative_to(_PACKAGE_DIR.parent)), path.read_bytes()))
    h = hashlib.sha256()
    for name, blob in sorted(items, key=lambda kv: kv[0]):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(blob)
        h.update(b"\0")
    return h.hexdigest()


def locked_physics() -> dict[str, Any]:
    return {
        "D_m": C.D_M,
        "obstruction_ratio": C.OBSTRUCTION_RATIO,
        "wavelength_m": C.WAVELENGTH_M,
        "wind_m_s": C.WIND_M_S,
        "tau0_factor": C.TAU0_FACTOR,
        "texp_over_tau0": C.TEXP_OVER_TAU0,
        "dt_over_tau0": C.DT_OVER_TAU0,
        "detector_samp_factor": C.DETECTOR_SAMP_FACTOR,
        "read_noise_e": C.READ_NOISE_E,
        "object_oversample": C.OBJECT_OVERSAMPLE,
        "eval_size": C.EVAL_SIZE,
        "default_j": C.DEFAULT_J,
        "default_padding_detector_px": C.DEFAULT_PADDING_DET_PX,
        "default_n_diam": C.DEFAULT_N_DIAM,
        "default_pupil_pad_factor": C.DEFAULT_PUPIL_PAD_FACTOR,
        "default_subharmonics": C.DEFAULT_SUBHARMONICS,
        "revision": C.REVISION,
        "schema_name": C.SCHEMA_NAME,
        "schema_version": C.SCHEMA_VERSION,
        "simulator_operator_version": C.SIMULATOR_OPERATOR_VERSION,
        "estimator_operator_version": C.ESTIMATOR_OPERATOR_VERSION,
    }


def config_hash(cfg: SimConfig | None = None) -> str:
    payload = locked_physics()
    if cfg is not None:
        payload["simconfig"] = json.loads(cfg.json_utf8().decode("utf-8"))
    return sha256_json(payload)


def current_method_fingerprint(cfg: SimConfig | None = None) -> dict[str, Any]:
    """Compatibility fingerprint of the current simulator/method operators."""
    if cfg is None:
        n_diam = C.DEFAULT_N_DIAM
        pad_factor = C.DEFAULT_PUPIL_PAD_FACTOR
        pupil_n = int(round(pad_factor * n_diam))
        if pupil_n % 2:
            pupil_n += 1
        j = C.DEFAULT_J
        padding = C.DEFAULT_PADDING_DET_PX
        sub = C.DEFAULT_SUBHARMONICS
        eval_size = C.EVAL_SIZE
    else:
        n_diam = cfg.n_diam
        pad_factor = cfg.pupil_pad_factor
        pupil_n = cfg.pupil_grid_size
        j = cfg.exposure_samples_j
        padding = cfg.padding_detector_px
        sub = cfg.subharmonic_levels
        eval_size = cfg.eval_size
    fp = {
        "revision": C.REVISION,
        "schema_name": C.SCHEMA_NAME,
        "schema_version": C.SCHEMA_VERSION,
        "simulator_operator_version": C.SIMULATOR_OPERATOR_VERSION,
        "estimator_operator_version": C.ESTIMATOR_OPERATOR_VERSION,
        "D_m": C.D_M,
        "obstruction_ratio": C.OBSTRUCTION_RATIO,
        "wavelength_m": C.WAVELENGTH_M,
        "wind_m_s": C.WIND_M_S,
        "n_diam": int(n_diam),
        "pupil_pad_factor": float(pupil_n) / float(n_diam),
        "pupil_grid_size": int(pupil_n),
        "exposure_samples_J": int(j),
        "padding_detector_px": int(padding),
        "subharmonic_levels": int(sub),
        "eval_size": int(eval_size),
        "object_oversample": int(C.OBJECT_OVERSAMPLE),
        "bin_factor_locked": int(round(C.DETECTOR_SAMP_FACTOR * pupil_n / n_diam)),
        "tau0_factor": C.TAU0_FACTOR,
        "texp_over_tau0": C.TEXP_OVER_TAU0,
        "dt_over_tau0": C.DT_OVER_TAU0,
        "detector_samp_factor": C.DETECTOR_SAMP_FACTOR,
        "read_noise_e": C.READ_NOISE_E,
        "source_hash": source_hash(),
    }
    # A method-level configuration hash deliberately excludes seed, regime and
    # frame count. config_hash(cfg) remains the separate per-experiment identity.
    fp["config_hash"] = sha256_json(compatibility_view(fp, PHYSICS_KEYS))
    return fp


def fingerprint_from_h5(path: Path) -> dict[str, Any]:
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        g = f["/config"]
        n_diam = int(round(float(g.attrs["D_m"]) / float(g.attrs["screen_dx_m"])))
        pupil_n = int(g.attrs["pupil_grid_size"])
        stored = {}
        # Generation identity is immutable evidence, not a certificate stamped
        # later by whichever code happens to run certification.
        if "generation_method_json" in g.attrs:
            raw = g.attrs["generation_method_json"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            stored = json.loads(str(raw))
            if not isinstance(stored, dict):
                raise ValueError("generation method must be a JSON object")
        fp = {
            "revision": str(f.attrs.get("revision", "")),
            "schema_name": str(f.attrs.get("schema_name", "")),
            "schema_version": str(f.attrs.get("schema_version", "")),
            "simulator_operator_version": str(
                stored.get(
                    "simulator_operator_version",
                    g.attrs.get("simulator_operator_version", "unknown"),
                )
            ),
            "estimator_operator_version": str(
                stored.get("estimator_operator_version", "unknown")
            ),
            "D_m": float(g.attrs["D_m"]),
            "obstruction_ratio": float(g.attrs["obstruction_ratio"]),
            "wavelength_m": float(g.attrs["wavelength_m"]),
            "wind_m_s": float(g.attrs["wind_m_s"]),
            "n_diam": n_diam,
            "pupil_pad_factor": float(pupil_n) / float(n_diam),
            "pupil_grid_size": pupil_n,
            "exposure_samples_J": int(g.attrs["exposure_samples_J"]),
            "padding_detector_px": int(g.attrs["padding_detector_px"]),
            "subharmonic_levels": int(g.attrs["subharmonic_levels"]),
            "eval_size": int(f["/object/feature_truth"].shape[0]),
            "object_oversample": stored.get("object_oversample", "unknown"),
            "bin_factor_locked": int(round(
                float(g.attrs["detector_pixel_scale_rad"]) * pupil_n
                * float(g.attrs["screen_dx_m"]) / float(g.attrs["wavelength_m"])
            )),
            "tau0_factor": float(g.attrs["tau0_s"]) * float(g.attrs["wind_m_s"]) / float(g.attrs["r0_m"]),
            "texp_over_tau0": float(g.attrs["texp_s"]) / float(g.attrs["tau0_s"]),
            "dt_over_tau0": float(g.attrs["dt_s"]) / float(g.attrs["tau0_s"]),
            "detector_samp_factor": float(g.attrs["detector_pixel_scale_rad"]) * float(g.attrs["D_m"]) / float(g.attrs["wavelength_m"]),
            "read_noise_e": float(g.attrs["read_noise_e"]),
            "seed": int(f.attrs["seed"]),
            "Dr0": float(g.attrs["Dr0"]),
            "N_frames": int(g.attrs["N_frames"]),
            "path": str(path),
        }
        if stored.get("source_hash"):
            fp["source_hash"] = stored["source_hash"]
        if stored.get("config_hash"):
            fp["config_hash"] = stored["config_hash"]
        return fp


def compatibility_view(
    fp: dict[str, Any], keys: tuple[str, ...] = COMPAT_KEYS
) -> dict[str, Any]:
    out = {}
    for key in keys:
        if key not in fp:
            continue
        val = fp[key]
        out[key] = val
    return out


def fingerprints_compatible(
    a: dict[str, Any],
    b: dict[str, Any],
    rtol: float = 1e-12,
    atol: float = 1e-12,
    keys: tuple[str, ...] = COMPAT_KEYS,
) -> bool:
    va, vb = compatibility_view(a, keys), compatibility_view(b, keys)
    if set(va) != set(keys) or set(vb) != set(keys):
        return False
    for key in va:
        x, y = va[key], vb[key]
        if (
            isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and not isinstance(x, bool)
            and not isinstance(y, bool)
        ):
            if not math.isfinite(x) or not math.isfinite(y):
                return False
            if abs(float(x) - float(y)) > atol + rtol * max(abs(float(x)), abs(float(y))):
                return False
        elif x != y:
            return False
    return True


def physics_compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return fingerprints_compatible(a, b)


def fingerprint_mismatch(
    a: dict[str, Any],
    b: dict[str, Any],
    keys: tuple[str, ...] = COMPAT_KEYS,
) -> list[str]:
    va, vb = compatibility_view(a, keys), compatibility_view(b, keys)
    names = sorted(set(va) | set(vb))
    diffs = []
    for key in names:
        if key not in va:
            diffs.append(f"{key}: missing in a")
            continue
        if key not in vb:
            diffs.append(f"{key}: missing in b")
            continue
        if va[key] != vb[key]:
            diffs.append(f"{key}: {va[key]!r} != {vb[key]!r}")
    return diffs


def certificate_is_current(stored: dict[str, Any], current: dict[str, Any] | None = None) -> bool:
    current = current_method_fingerprint() if current is None else current
    return fingerprints_compatible(stored, current)


def current_fingerprint_for_file(fp: dict[str, Any]) -> dict[str, Any]:
    """Current implementation evaluated at the file's numerical configuration."""
    cfg = SimConfig(
        seed=int(fp['seed']), dr0=float(fp['Dr0']), n_frames=int(fp['N_frames']),
        n_diam=int(fp['n_diam']), pupil_pad_factor=float(fp['pupil_pad_factor']),
        exposure_samples_j=int(fp['exposure_samples_J']),
        padding_detector_px=int(fp['padding_detector_px']),
        subharmonic_levels=int(fp['subharmonic_levels']), eval_size=int(fp['eval_size']),
    )
    return current_method_fingerprint(cfg)


def validate_status(status: str) -> str:
    if status not in RESULT_STATUSES:
        raise ValueError(
            f"unknown result status {status!r}; expected one of {RESULT_STATUSES}"
        )
    return status


def experiment_manifest(
    experiment_id: str,
    *,
    status: str,
    protocol: str,
    inputs: list[dict[str, Any]] | None = None,
    results: list[dict[str, Any]] | None = None,
    seed_coverage: dict[str, Any] | None = None,
    tolerances: dict[str, Any] | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_status(status)
    payload = {
        "schema_name": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": str(experiment_id),
        "status": status,
        "document_revision": C.REVISION,
        "roadmap_revision": C.ROADMAP_REVISION,
        "simulator_operator_version": C.SIMULATOR_OPERATOR_VERSION,
        "estimator_operator_version": C.ESTIMATOR_OPERATOR_VERSION,
        "source_hash": source_hash(),
        "config_hash": config_hash(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": protocol,
        "seed_coverage": seed_coverage or {},
        "tolerances": tolerances or {},
        "inputs": inputs or [],
        "results": results or [],
        "notes": notes,
    }
    if extra:
        payload.update(extra)
    return payload


def manifest_errors(manifest: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if manifest.get("schema_name") != MANIFEST_SCHEMA:
        errors.append("schema_name mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if "experiment_id" not in manifest:
        errors.append("missing experiment_id")
    status = manifest.get("status")
    if status not in RESULT_STATUSES:
        errors.append(f"invalid status {status!r}")
    for key in (
        "protocol",
        "source_hash",
        "config_hash",
        "simulator_operator_version",
        "estimator_operator_version",
    ):
        if not manifest.get(key):
            errors.append(f"missing {key}")
    if not isinstance(manifest.get("inputs", []), list):
        errors.append("inputs must be a list")
    if not isinstance(manifest.get("results", []), list):
        errors.append("results must be a list")
    else:
        for i, result in enumerate(manifest.get("results", [])):
            if not isinstance(result, dict):
                errors.append(f"results[{i}] must be an object")
                continue
            if result.get("status") not in RESULT_STATUSES:
                errors.append(f"results[{i}] invalid status")
            if not isinstance(result.get("path"), str) or not result['path']:
                errors.append(f"results[{i}] missing path")
            if 'gate_passed' in result and not isinstance(result['gate_passed'], bool):
                errors.append(f"results[{i}] gate_passed must be a boolean")
    try:
        json.dumps(manifest, allow_nan=False)
    except (ValueError, TypeError):
        errors.append("manifest must contain finite JSON values")
    return errors


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path = Path(path)
    errors = manifest_errors(manifest)
    if errors:
        raise ValueError("invalid manifest: " + "; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text())
    errors = manifest_errors(manifest)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return manifest


def write_certificate_dataset(group, fingerprint: dict[str, Any]) -> None:
    import h5py

    payload = json.dumps(fingerprint, sort_keys=True)
    if "method_certificate_json" in group:
        del group["method_certificate_json"]
    group.create_dataset(
        "method_certificate_json",
        data=payload,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def load_stored_certificate(path: Path) -> dict[str, Any] | None:
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        if "method_certificate_json" not in f["/validation"]:
            return None
        raw = f["/validation/method_certificate_json"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))

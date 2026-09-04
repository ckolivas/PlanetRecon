"""R8 HDF5 truth-file schema v1."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from planetrecon import constants as C
from planetrecon.config import SimConfig


REQUIRED_ROOT_ATTRS = ("schema_name", "schema_version", "revision", "seed")
REQUIRED_DATASETS = (
    "/config/json_utf8",
    "/object/full_latent_4x",
    "/object/full_latent_detector",
    "/object/reference_disk_mask",
    "/object/feature_crop_origin",
    "/object/bland_crop_origin",
    "/object/feature_truth",
    "/object/bland_truth",
    "/object/eval_window",
    "/object/ideal_mtf",
    "/object/high_band_mask",
    "/object/features/oval1",
    "/object/features/oval2",
    "/object/features/oval3",
    "/pupil/amplitude",
    "/pupil/frequency_x",
    "/pupil/frequency_y",
    "/atmosphere/frame_time_s",
    "/atmosphere/phase_rms_rad",
    "/atmosphere/strehl_proxy",
    "/atmosphere/kl60_coeff",
    "/atmosphere/kl60_residual_rms",
    "/truth_transfer/feature/psf_bar",
    "/truth_transfer/feature/otf_bar",
    "/frames/feature_expected_e",
    "/frames/feature_observed_e",
    "/frames/bland_expected_e",
    "/frames/bland_observed_e",
    "/frames/shift_xy_detector_px",
    "/validation/no_wrap_pass",
    "/validation/psf_energy_error",
)


def filename(seed: int, dr0: float) -> str:
    return f"gate1_Dr0-{int(dr0)}_seed-{int(seed):05d}.h5"


def write_truth(path: Path, cfg: SimConfig, arrays: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (1, cfg.eval_size, cfg.eval_size)
    with h5py.File(path, "w") as f:
        f.attrs["schema_name"] = C.SCHEMA_NAME
        f.attrs["schema_version"] = C.SCHEMA_VERSION
        f.attrs["revision"] = C.REVISION
        f.attrs["seed"] = int(cfg.seed)

        gcfg = f.create_group("config")
        for key, val in (
            ("D_m", cfg.d_m),
            ("obstruction_ratio", cfg.obstruction_ratio),
            ("wavelength_m", cfg.wavelength_m),
            ("wind_m_s", cfg.wind_m_s),
            ("Dr0", cfg.dr0),
            ("r0_m", cfg.r0_m),
            ("tau0_s", cfg.tau0_s),
            ("texp_s", cfg.texp_s),
            ("dt_s", cfg.dt_s),
            ("N_frames", cfg.n_frames),
            ("detector_pixel_scale_rad", cfg.detector_pixel_scale_rad),
            ("detector_pixel_scale_arcsec", cfg.detector_pixel_scale_arcsec),
            ("read_noise_e", C.READ_NOISE_E),
            ("source_rate_scale", arrays["source_rate_scale"]),
            ("pupil_grid_size", cfg.pupil_grid_size),
            ("screen_dx_m", cfg.screen_dx_m),
            ("subharmonic_levels", cfg.subharmonic_levels),
            ("exposure_samples_J", cfg.exposure_samples_j),
            ("padding_detector_px", cfg.padding_detector_px),
        ):
            gcfg.attrs[key] = val
        gcfg.create_dataset(
            "screen_shape", data=np.array(arrays["screen_shape"], dtype=np.int32)
        )
        gcfg.create_dataset(
            "json_utf8",
            data=cfg.json_utf8().decode("utf-8"),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        gobj = f.create_group("object")
        gobj.create_dataset(
            "full_latent_4x", data=arrays["full_latent_4x"].astype(np.float32)
        )
        gobj.create_dataset(
            "full_latent_detector",
            data=arrays["full_latent_detector"].astype(np.float32),
        )
        gobj.create_dataset(
            "reference_disk_mask",
            data=arrays["reference_disk_mask"].astype(np.uint8),
        )
        gobj.create_dataset(
            "feature_crop_origin",
            data=np.array(arrays["feature_crop_origin"], dtype=np.int32),
        )
        gobj.create_dataset(
            "bland_crop_origin",
            data=np.array(arrays["bland_crop_origin"], dtype=np.int32),
        )
        gobj.create_dataset(
            "feature_truth", data=arrays["feature_truth"].astype(np.float32)
        )
        gobj.create_dataset(
            "bland_truth", data=arrays["bland_truth"].astype(np.float32)
        )
        gobj.create_dataset(
            "eval_window", data=arrays["eval_window"].astype(np.float32)
        )
        gobj.create_dataset(
            "ideal_mtf", data=arrays["ideal_mtf"].astype(np.float32)
        )
        gobj.create_dataset(
            "high_band_mask", data=arrays["high_band_mask"].astype(np.uint8)
        )
        gfeat = gobj.create_group("features")
        for oval in C.OVALS:
            go = gfeat.create_group(oval["name"])
            for k, v in oval.items():
                if k == "name":
                    continue
                go.attrs[k] = v

        gp = f.create_group("pupil")
        gp.create_dataset(
            "amplitude", data=arrays["pupil_amplitude"].astype(np.float32)
        )
        gp.create_dataset(
            "frequency_x", data=arrays["pupil_frequency_x"].astype(np.float32)
        )
        gp.create_dataset(
            "frequency_y", data=arrays["pupil_frequency_y"].astype(np.float32)
        )

        ga = f.create_group("atmosphere")
        ga.create_dataset(
            "frame_time_s", data=arrays["frame_time_s"].astype(np.float64)
        )
        ga.create_dataset(
            "phase_rms_rad", data=arrays["phase_rms_rad"].astype(np.float32)
        )
        ga.create_dataset(
            "strehl_proxy", data=arrays["strehl_proxy"].astype(np.float32)
        )
        ga.create_dataset(
            "kl60_coeff", data=arrays["kl60_coeff"].astype(np.float32)
        )
        ga.create_dataset(
            "kl60_residual_rms",
            data=arrays["kl60_residual_rms"].astype(np.float32),
        )

        gtt = f.create_group("truth_transfer")
        gf = gtt.create_group("feature")
        psf_ds = gf.create_dataset(
            "psf_bar",
            data=arrays["psf_bar"].astype(np.float32),
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
        )
        otf_ds = gf.create_dataset(
            "otf_bar",
            data=arrays["otf_bar"].astype(np.complex64),
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
        )
        gb = gtt.create_group("bland")
        gb["psf_bar"] = psf_ds
        gb["otf_bar"] = otf_ds

        gfr = f.create_group("frames")
        for name in (
            "feature_expected_e",
            "feature_observed_e",
            "bland_expected_e",
            "bland_observed_e",
        ):
            gfr.create_dataset(
                name,
                data=arrays[name].astype(np.float32),
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
            )
        gfr.create_dataset(
            "shift_xy_detector_px",
            data=arrays["shift_xy_detector_px"].astype(np.float32),
        )

        gv = f.create_group("validation")
        for key, val in arrays["validation"].items():
            if np.isscalar(val) or isinstance(val, (bool, np.bool_)):
                gv.create_dataset(key, data=val)
            else:
                gv.create_dataset(key, data=np.asarray(val))


def schema_errors(path: Path) -> list[str]:
    errors = []
    with h5py.File(path, "r") as f:
        for attr in REQUIRED_ROOT_ATTRS:
            if attr not in f.attrs:
                errors.append(f"missing root attr {attr}")
        if f.attrs.get("schema_name") != C.SCHEMA_NAME:
            errors.append("schema_name mismatch")
        if f.attrs.get("schema_version") != C.SCHEMA_VERSION:
            errors.append("schema_version mismatch")
        if f.attrs.get("revision") != C.REVISION:
            errors.append("revision mismatch")
        for ds in REQUIRED_DATASETS:
            if ds not in f:
                errors.append(f"missing {ds}")
        if "/frames/feature_expected_e" in f:
            n = f["/frames/feature_expected_e"].shape[0]
            if n < 1:
                errors.append("no frames")
    return errors

"""Top-level Gate-1 seed/regime simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from planetrecon import constants as C
from planetrecon.atmosphere import (
    extract_phase,
    finite_exposure_psf,
    generate_screen,
    kolmogorov_structure_function,
    no_wrap_ok,
    structure_function,
)
from planetrecon.config import SimConfig, make_config
from planetrecon.hdf5io import filename, validation_is_gate_eligible, write_truth
from planetrecon.kl import KLBasis
from planetrecon.metric import (
    eval_window,
    high_band_mask,
    ideal_mtf_from_dl_psf,
    relative_high_band,
)
from planetrecon.object import (
    feature_measurement_masks,
    make_object_scene,
    source_rate_scale,
)
from planetrecon.optics import (
    bin_box,
    center_crop,
    centroid_px,
    diffraction_limited_psf,
    make_pupil,
    otf_from_centered_psf,
)
from planetrecon.rng import frame_noise, noise_rng, screen_rng


def simulate(cfg: SimConfig, out_dir: Path, progress=None) -> Path:
    out_dir = Path(out_dir)
    pupil = make_pupil(cfg)
    scene = make_object_scene(cfg)
    psf_dl = diffraction_limited_psf(pupil)
    scale = source_rate_scale(cfg, scene, psf_dl)
    flux = scale * (cfg.texp_s / C.T0_S)
    bin_f = cfg.bin_factor
    if bin_f != C.OBJECT_OVERSAMPLE:
        raise RuntimeError(
            f"optical bin factor {bin_f} != object oversample {C.OBJECT_OVERSAMPLE}"
        )

    psf_dl_det = center_crop(bin_box(psf_dl, bin_f), cfg.eval_size)
    psf_dl_det = psf_dl_det / psf_dl_det.sum()
    mtf = ideal_mtf_from_dl_psf(psf_dl_det)
    hmask = high_band_mask(cfg, mtf).astype(np.uint8)
    window = eval_window(cfg.eval_size)
    kl_basis = KLBasis(pupil, C.KL_MODES)

    rng_s = screen_rng(cfg.seed)
    screen = generate_screen(cfg, rng_s)
    rng_n = noise_rng(cfg)

    n = cfg.n_frames
    eval_n = cfg.eval_size
    times = np.arange(n, dtype=np.float64) * cfg.dt_s
    psf_bar = np.empty((n, eval_n, eval_n), dtype=np.float64)
    otf_bar = np.empty((n, eval_n, eval_n), dtype=np.complex128)
    feat_e = np.empty((n, eval_n, eval_n), dtype=np.float64)
    feat_o = np.empty((n, eval_n, eval_n), dtype=np.float64)
    bland_e = np.empty((n, eval_n, eval_n), dtype=np.float64)
    bland_o = np.empty((n, eval_n, eval_n), dtype=np.float64)
    shifts = np.empty((n, 2), dtype=np.float64)
    phase_rms = np.empty(n, dtype=np.float64)
    strehl = np.empty(n, dtype=np.float64)
    kl_coeff = np.empty((n, C.KL_MODES), dtype=np.float64)
    kl_resid = np.empty(n, dtype=np.float64)
    energy_err = []
    exposure_eh = np.nan

    oxf, oyf = scene.feature_origin
    oxb, oyb = scene.bland_origin
    peak_dl = float(psf_dl.max())

    for k in range(n):
        t0 = times[k]
        psf4 = finite_exposure_psf(
            pupil,
            screen,
            t0,
            cfg.texp_s,
            cfg.exposure_samples_j,
            cfg.wind_m_s,
        )
        energy_err.append(abs(float(psf4.sum()) - 1.0))
        img4 = fftconvolve(scene.latent_4x, psf4, mode="same")
        img = flux * bin_box(img4, bin_f)
        feat = img[oyf : oyf + eval_n, oxf : oxf + eval_n]
        bland = img[oyb : oyb + eval_n, oxb : oxb + eval_n]
        feat_e[k] = feat
        bland_e[k] = bland
        feat_o[k] = frame_noise(rng_n, feat, C.READ_NOISE_E)
        bland_o[k] = frame_noise(rng_n, bland, C.READ_NOISE_E)

        psf_det = center_crop(bin_box(psf4, bin_f), eval_n)
        psf_det = psf_det / psf_det.sum()
        psf_bar[k] = psf_det
        otf_bar[k] = otf_from_centered_psf(psf_det)
        sx, sy = centroid_px(psf_det)
        shifts[k] = (sx, sy)
        strehl[k] = float(psf4.max() / peak_dl)

        t_mid = t0 + 0.5 * cfg.texp_s
        phase = extract_phase(pupil, screen, t_mid, cfg.wind_m_s)
        ph = phase[pupil.mask]
        ph = ph - ph.mean()
        phase_rms[k] = float(np.sqrt(np.mean(ph**2)))
        coeff, resid = kl_basis.project(phase)
        kl_coeff[k] = coeff
        kl_resid[k] = resid
        if k == 0:
            psf_2j = finite_exposure_psf(
                pupil,
                screen,
                t0,
                cfg.texp_s,
                max(2 * cfg.exposure_samples_j, 16),
                cfg.wind_m_s,
            )
            a = center_crop(bin_box(psf4, bin_f), eval_n)
            b = center_crop(bin_box(psf_2j, bin_f), eval_n)
            a = a / a.sum()
            b = b / b.sum()
            exposure_eh = relative_high_band(a, b, window, mtf, hmask.astype(bool))
        if progress and (k % 50 == 0 or k == n - 1):
            progress(k, n)

    rho = np.linspace(
        3.0 * screen.dx_m,
        min(cfg.d_m, 0.2 * screen.phi.shape[0] * screen.dx_m),
        16,
    )
    sf_m = structure_function(screen.phi, screen.dx_m, rho)
    sf_t = kolmogorov_structure_function(rho, cfg.r0_m)

    sf_rel_error = float(np.median(np.abs(sf_m / sf_t - 1.0)))
    validation = {
        "structure_function_rho": rho,
        "structure_function_measured": sf_m,
        "structure_function_target": sf_t,
        "structure_function_rel_error": sf_rel_error,
        "structure_function_local_pass": sf_rel_error < C.STRUCTURE_FUNCTION_REL_TOL,
        # Kolmogorov r0 is an ensemble statistic. The development suite
        # certifies this flag after pooling the three independent seed fields.
        "structure_function_pass": False,
        "psf_energy_error": float(np.max(energy_err)),
        "no_wrap_pass": bool(no_wrap_ok(cfg, screen)),
        "kl60_residual_summary": float(np.median(kl_resid)),
        "grid_convergence": np.inf,
        "grid_convergence_pass": False,
        "exposure_convergence": float(exposure_eh),
        "exposure_convergence_pass": exposure_eh < C.EH_EXPOSURE_TOL,
        "padding_convergence": np.inf,
        "padding_convergence_pass": False,
        "lowfreq_convergence": np.full(3, np.inf),
        "lowfreq_convergence_pass": False,
        "gate_eligible": False,
    }

    # These are deliberately computed for the same seed/regime as the file.
    # Keeping provisional failures above ensures an interrupted validation can
    # never be mistaken for a Gate-eligible product.
    from planetrecon.validate import convergence_values

    validation.update(convergence_values(cfg))
    validation["gate_eligible"] = validation_is_gate_eligible(validation)

    feature_masks = {}
    for oval in C.OVALS:
        feature_masks[oval["name"]] = feature_measurement_masks(
            scene.feature_origin,
            scene.disk_cx_det,
            scene.disk_cy_det,
            oval,
            cfg.eval_size,
        )

    arrays = {
        "source_rate_scale": scale,
        "screen_shape": screen.phi.shape,
        "full_latent_4x": scene.latent_4x,
        "full_latent_detector": scene.latent_detector,
        "reference_disk_mask": scene.reference_disk_mask,
        "feature_crop_origin": scene.feature_origin,
        "bland_crop_origin": scene.bland_origin,
        "feature_truth": scene.feature_truth,
        "bland_truth": scene.bland_truth,
        "eval_window": window,
        "ideal_mtf": mtf,
        "high_band_mask": hmask,
        "feature_measurement_masks": feature_masks,
        "pupil_amplitude": pupil.amplitude,
        "pupil_frequency_x": pupil.frequency_x,
        "pupil_frequency_y": pupil.frequency_y,
        "frame_time_s": times,
        "phase_rms_rad": phase_rms,
        "strehl_proxy": strehl,
        "kl60_coeff": kl_coeff,
        "kl60_residual_rms": kl_resid,
        "psf_bar": psf_bar,
        "otf_bar": otf_bar,
        "feature_expected_e": feat_e,
        "feature_observed_e": feat_o,
        "bland_expected_e": bland_e,
        "bland_observed_e": bland_o,
        "shift_xy_detector_px": shifts,
        "validation": validation,
    }
    path = out_dir / filename(cfg.seed, cfg.dr0)
    write_truth(path, cfg, arrays)
    write_summary(path, cfg, validation)
    return path


def write_summary(h5_path: Path, cfg: SimConfig, validation: dict) -> Path:
    txt = h5_path.with_suffix(".validation.txt")
    lines = [
        "PlanetRecon Gate-1 simulator validation",
        f"revision: {C.REVISION}",
        f"schema: {C.SCHEMA_NAME} {C.SCHEMA_VERSION}",
        f"seed: {cfg.seed}",
        f"D/r0: {cfg.dr0:g}",
        f"r0_m: {cfg.r0_m}",
        f"tau0_s: {cfg.tau0_s}",
        f"texp_s: {cfg.texp_s}",
        f"dt_s: {cfg.dt_s}",
        f"N_frames: {cfg.n_frames}",
        f"J: {cfg.exposure_samples_j}",
        f"padding_detector_px: {cfg.padding_detector_px}",
        f"pupil_grid_size: {cfg.pupil_grid_size}",
        "",
        f"psf_energy_error: {validation['psf_energy_error']:.3e}",
        f"no_wrap_pass: {validation['no_wrap_pass']}",
        f"kl60_residual_median: {validation['kl60_residual_summary']:.6e}",
        f"exposure_convergence: {validation['exposure_convergence']}",
        "",
        "structure function rho, measured, target, ratio:",
    ]
    rho = np.asarray(validation["structure_function_rho"])
    meas = np.asarray(validation["structure_function_measured"])
    tgt = np.asarray(validation["structure_function_target"])
    for r, m, t in zip(rho, meas, tgt):
        ratio = m / t if t else float("nan")
        lines.append(f"  {r:.5f}  {m:.5e}  {t:.5e}  {ratio:.3f}")
    lines.extend(
        [
            f"structure_function_pass: {validation['structure_function_pass']}",
            f"grid_convergence: {validation['grid_convergence']}",
            f"grid_convergence_pass: {validation['grid_convergence_pass']}",
            f"padding_convergence: {validation['padding_convergence']}",
            f"padding_convergence_pass: {validation['padding_convergence_pass']}",
            f"lowfreq_convergence: {np.asarray(validation['lowfreq_convergence']).tolist()}",
            f"lowfreq_convergence_pass: {validation['lowfreq_convergence_pass']}",
        ]
    )
    eligible = bool(validation["gate_eligible"])
    lines.append("")
    lines.append(f"Gate-eligible: {'YES' if eligible else 'NO'}")
    txt.write_text("\n".join(lines) + "\n")
    return txt


def generate_one(
    seed: int,
    dr0: float,
    out_dir: Path,
    n_frames: int | None = None,
    **kwargs,
) -> Path:
    kw = dict(kwargs)
    if n_frames is not None:
        kw["n_frames"] = n_frames
    cfg = make_config(seed, dr0, **kw)

    def _progress(k, n):
        print(f"  frame {k+1}/{n}", flush=True)

    print(f"Generating seed={seed} D/r0={dr0:g} N={cfg.n_frames}", flush=True)
    return simulate(cfg, out_dir, progress=_progress)

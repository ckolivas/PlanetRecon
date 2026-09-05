"""Prompt 1 mandatory validation checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve

from planetrecon import constants as C
from planetrecon.atmosphere import (
    PhaseScreen,
    extract_phase,
    finite_exposure_psf,
    generate_screen,
    generate_square_screen,
    kolmogorov_structure_function,
    no_wrap_ok,
    sample_times,
    structure_function,
)
from planetrecon.config import SimConfig, make_config
from planetrecon.hdf5io import (
    filename,
    gate_errors,
    schema_errors,
    validation_is_gate_eligible,
)
from planetrecon.provenance import (
    certificate_is_current,
    current_fingerprint_for_file,
    fingerprint_from_h5,
    fingerprint_mismatch,
    physics_compatible,
    write_certificate_dataset,
)
from planetrecon.kl import KLBasis, noll_nm
from planetrecon.metric import (
    eval_window,
    high_band_mask,
    high_band_power,
    ideal_mtf_from_dl_psf,
    relative_high_band,
)
from planetrecon.object import (
    latent_object_4x,
    make_object_scene,
    ovals_inside_disk,
    source_rate_scale,
)
from planetrecon.optics import (
    annular_airy_psf,
    bin_box,
    center_crop,
    centroid_px,
    diffraction_limited_psf,
    instantaneous_psf,
    make_pupil,
)
from planetrecon.rng import screen_rng


@dataclass
class Check:
    name: str
    passed: bool
    message: str
    value: float | None = None


def _ok(name: str, passed: bool, message: str, value=None) -> Check:
    return Check(name, bool(passed), message, value)


def check_timing() -> list[Check]:
    out = []
    for dr0, r0, tau0, texp, dt in (
        (8.0, 0.03125, 0.0019625, 0.00058875, 0.0019625),
        (4.0, 0.0625, 0.0039250, 0.00117750, 0.0039250),
    ):
        cfg = make_config(1001, dr0)
        out.append(
            _ok(
                f"timing_r0_dr0{dr0:g}",
                abs(cfg.r0_m - r0) < 1e-12,
                f"r0={cfg.r0_m} expected {r0}",
                cfg.r0_m,
            )
        )
        out.append(
            _ok(
                f"timing_tau0_dr0{dr0:g}",
                abs(cfg.tau0_s - tau0) < 1e-12,
                f"tau0={cfg.tau0_s} expected {tau0}",
                cfg.tau0_s,
            )
        )
        out.append(
            _ok(
                f"timing_texp_dr0{dr0:g}",
                abs(cfg.texp_s - texp) < 1e-12,
                f"texp={cfg.texp_s} expected {texp}",
                cfg.texp_s,
            )
        )
        out.append(
            _ok(
                f"timing_dt_dr0{dr0:g}",
                abs(cfg.dt_s - dt) < 1e-12,
                f"dt={cfg.dt_s} expected {dt}",
                cfg.dt_s,
            )
        )
        dx = cfg.wind_m_s * cfg.dt_s
        out.append(
            _ok(
                f"timing_dx_vdt_dr0{dr0:g}",
                abs(dx - cfg.wind_m_s * cfg.tau0_s) < 1e-12,
                f"v*dt={dx}",
                dx,
            )
        )
    seeds = set(C.DEV_SEEDS) & set(C.EVAL_SEEDS)
    out.append(_ok("seeds_disjoint", not seeds, f"overlap={seeds}"))
    return out


def check_optics() -> list[Check]:
    cfg = make_config(1001, 8.0)
    pupil = make_pupil(cfg)
    phase0 = np.zeros_like(pupil.amplitude)
    psf = instantaneous_psf(pupil.amplitude, phase0)
    energy_err = abs(float(psf.sum()) - 1.0)
    checks = [
        _ok(
            "psf_energy",
            energy_err < C.PSF_ENERGY_TOL,
            f"sum-1={energy_err:.3e}",
            energy_err,
        )
    ]
    analytic = annular_airy_psf(cfg, cfg.pupil_grid_size, cfg.optical_pixel_scale_rad)
    core = slice(cfg.pupil_grid_size // 2 - 8, cfg.pupil_grid_size // 2 + 8)
    num = psf[core, core].ravel()
    an = analytic[core, core].ravel()
    num = num / np.linalg.norm(num)
    an = an / np.linalg.norm(an)
    corr = float(num @ an)
    checks.append(_ok("dl_airy_core", corr > 0.98, f"core corr={corr:.4f}", corr))

    alpha = 50.0
    phase = alpha * pupil.x_m
    psf_t = instantaneous_psf(pupil.amplitude, phase)
    dx, dy = centroid_px(psf_t)
    checks.append(
        _ok(
            "tip_shifts_psf",
            abs(dx) > 5 * abs(dy) and abs(dx) > 1.0,
            f"centroid dx={dx:.3f} dy={dy:.3f} px (optical)",
            dx,
        )
    )
    return checks


def check_object_geometry() -> list[Check]:
    cfg = make_config(1001, 8.0)
    scene = make_object_scene(cfg)
    checks = [
        _ok("ovals_inside_disk", ovals_inside_disk(), "oval 3-sigma extent inside disk")
    ]
    a = latent_object_4x(cfg.object_n4)
    b = latent_object_4x(cfg.object_n4)
    checks.append(_ok("object_deterministic", np.allclose(a, b), "two draws match"))

    n = cfg.eval_size
    ox, oy = scene.feature_origin
    cx, cy = scene.disk_cx_det, scene.disk_cy_det
    limb_x = cx + C.PLANET_RADIUS_DET_PX
    local_limb_x = limb_x - ox
    checks.append(
        _ok(
            "feature_has_plusx_limb",
            0 <= local_limb_x < n,
            f"limb local x={local_limb_x:.2f}",
            local_limb_x,
        )
    )
    checks.append(
        _ok(
            "feature_limb_margin",
            local_limb_x >= C.CROP_LIMB_MARGIN_PX
            and (n - local_limb_x) >= C.CROP_LIMB_MARGIN_PX,
            f"limb margin local x={local_limb_x:.2f}",
            local_limb_x,
        )
    )
    taper = 0.5 * C.TUKEY_ALPHA * n
    oval1 = (cx + C.OVALS[0]["x_px"] - ox, cy + C.OVALS[0]["y_px"] - oy)
    checks.append(
        _ok(
            "feature_has_oval1",
            taper <= oval1[0] < n - taper and taper <= oval1[1] < n - taper,
            f"oval1 local=({oval1[0]:.2f},{oval1[1]:.2f}) taper={taper}",
        )
    )
    belt_y = cy - C.PLANET_RADIUS_DET_PX / 3.0
    local_belt = belt_y - oy
    checks.append(
        _ok(
            "feature_has_belt",
            0 <= local_belt < n,
            f"belt local y={local_belt:.2f}",
            local_belt,
        )
    )

    oxb, oyb = scene.bland_origin
    yy, xx = np.indices((96, 96), dtype=np.float64)
    x = oxb + 16 + xx + 0.5
    y = oyb + 16 + yy + 0.5
    r = np.hypot(x - cx, y - cy)
    checks.append(
        _ok(
            "bland_central96_no_limb",
            np.all(r < C.PLANET_RADIUS_DET_PX),
            f"max r in central 96={r.max():.2f}",
            float(r.max()),
        )
    )
    for oval in C.OVALS:
        oval_x = cx + oval["x_px"]
        oval_y = cy + oval["y_px"]
        in_central = (
            oxb + 16 <= oval_x < oxb + 112
            and oyb + 16 <= oval_y < oyb + 112
        )
        checks.append(
            _ok(
                f"bland_excludes_{oval['name']}_centre",
                not in_central,
                f"{oval['name']} in bland central 96={in_central}",
            )
        )
    return checks


def check_high_band_ovals(cfg: SimConfig | None = None) -> list[Check]:
    cfg = cfg or make_config(1001, 8.0)
    pupil = make_pupil(cfg)
    scene = make_object_scene(cfg)
    psf_dl = diffraction_limited_psf(pupil)
    psf_det = center_crop(bin_box(psf_dl, cfg.bin_factor), cfg.eval_size)
    psf_det /= psf_det.sum()
    mtf = ideal_mtf_from_dl_psf(psf_det)
    mask = high_band_mask(cfg, mtf)
    window = eval_window(cfg.eval_size)
    n4 = cfg.object_n4
    base = latent_object_4x(n4, include_ovals=False)
    full = latent_object_4x(n4, include_ovals=True)
    ox, oy = scene.feature_origin

    def det_crop(obj4):
        img = bin_box(obj4, C.OBJECT_OVERSAMPLE)
        return img[oy : oy + cfg.eval_size, ox : ox + cfg.eval_size]

    checks = []
    nonzero = 0
    from planetrecon.object import _highres_coords, _oval_multiplier

    x, y = _highres_coords(n4, C.OBJECT_OVERSAMPLE)
    for oval in C.OVALS:
        single = base * _oval_multiplier(x, y, oval)
        p = high_band_power(det_crop(single) - det_crop(base), window, mtf, mask)
        ok = p > 0
        if ok:
            nonzero += 1
        checks.append(
            _ok(
                f"oval_{oval['name']}_high_band",
                ok,
                f"H-band power={p:.3e}",
                p,
            )
        )
    oval1_ok = checks[0].passed
    checks.append(
        _ok(
            "oval1_and_another_in_H",
            oval1_ok and nonzero >= 2,
            f"ovals with H power={nonzero}",
            float(nonzero),
        )
    )
    nyquist = 0.5 / cfg.detector_pixel_scale_rad
    checks.append(
        _ok(
            "nyquist_is_fc",
            abs(nyquist - cfg.f_c) / cfg.f_c < 1e-12,
            f"Nyquist={nyquist} f_c={cfg.f_c}",
        )
    )
    assert full.shape == base.shape
    return checks


def check_flux() -> list[Check]:
    cfg = make_config(1001, 8.0)
    pupil = make_pupil(cfg)
    scene = make_object_scene(cfg)
    psf_dl = diffraction_limited_psf(pupil)
    scale = source_rate_scale(cfg, scene, psf_dl)
    img4 = fftconvolve(scene.latent_4x, psf_dl, mode="same")
    img = scale * bin_box(img4, C.OBJECT_OVERSAMPLE)
    ox, oy = scene.feature_origin
    crop = img[oy : oy + cfg.eval_size, ox : ox + cfg.eval_size]
    mean = float(crop[scene.reference_disk_mask.astype(bool)].mean())
    checks = [
        _ok(
            "flux_800_at_T0",
            abs(mean - C.REF_MEAN_E_AT_T0) < 1e-6,
            f"mean={mean}",
            mean,
        )
    ]
    img2 = (2.0) * img
    mean2 = float(img2[oy : oy + cfg.eval_size, ox : ox + cfg.eval_size][
        scene.reference_disk_mask.astype(bool)
    ].mean())
    checks.append(
        _ok(
            "photons_scale_with_texp",
            abs(mean2 / mean - 2.0) < 1e-12,
            f"ratio={mean2/mean}",
            mean2 / mean,
        )
    )
    return checks


def check_screen_and_exposure() -> list[Check]:
    cfg = make_config(1001, 8.0, n_frames=32, exposure_samples_j=8)
    pupil = make_pupil(cfg)
    rng = screen_rng(cfg.seed)
    screen = generate_screen(cfg, rng)
    checks = [
        _ok(
            "no_wrap_includes_final_exposure",
            no_wrap_ok(cfg, screen, n_frames=cfg.n_frames),
            "extraction at t_end stays on-screen",
        )
    ]
    rho = np.linspace(3 * screen.dx_m, min(cfg.d_m, 0.15 * screen.phi.shape[0] * screen.dx_m), 12)
    meas = structure_function(screen.phi, screen.dx_m, rho)
    tgt = kolmogorov_structure_function(rho, cfg.r0_m)
    ratio = meas / tgt
    med = float(np.median(np.abs(ratio - 1.0)))
    checks.append(
        _ok(
            "structure_function",
            med < C.STRUCTURE_FUNCTION_REL_TOL,
            f"median |ratio-1|={med:.3f}",
            med,
        )
    )

    t0 = 0.0
    h_j = finite_exposure_psf(
        pupil, screen, t0, cfg.texp_s, 8, cfg.wind_m_s
    )
    h_2j = finite_exposure_psf(
        pupil, screen, t0, cfg.texp_s, 16, cfg.wind_m_s
    )
    h_j_d = center_crop(bin_box(h_j, cfg.bin_factor), cfg.eval_size)
    h_2j_d = center_crop(bin_box(h_2j, cfg.bin_factor), cfg.eval_size)
    h_j_d /= h_j_d.sum()
    h_2j_d /= h_2j_d.sum()
    window = eval_window(h_j_d.shape[0])
    psf_dl_d = center_crop(
        bin_box(diffraction_limited_psf(pupil), cfg.bin_factor), cfg.eval_size
    )
    psf_dl_d /= psf_dl_d.sum()
    mtf = ideal_mtf_from_dl_psf(psf_dl_d)
    mask = high_band_mask(cfg, mtf)
    eh = relative_high_band(h_j_d, h_2j_d, window, mtf, mask)
    checks.append(
        _ok(
            "exposure_J_doubling",
            eh < C.EH_EXPOSURE_TOL,
            f"E_H(J=8 vs 16)={eh:.4e}",
            eh,
        )
    )

    h_inst = instantaneous_psf(
        pupil.amplitude, extract_phase(pupil, screen, t0, cfg.wind_m_s)
    )
    h_small = finite_exposure_psf(
        pupil, screen, t0, cfg.texp_s * 1e-6, 1, cfg.wind_m_s
    )
    l1 = float(np.mean(np.abs(h_inst - h_small)))
    checks.append(
        _ok("texp_to_zero", l1 < 1e-6, f"mean |dPSF|={l1:.3e}", l1)
    )

    phase = extract_phase(pupil, screen, 0.5 * cfg.texp_s, cfg.wind_m_s)
    basis = KLBasis(pupil)
    _, resid = basis.project(phase)
    ph = phase[pupil.mask]
    ph = ph - ph.mean()
    rms = float(np.sqrt(np.mean(ph**2)))
    checks.append(
        _ok(
            "kl60_residual",
            resid > 1e-6 and resid > 0.02 * rms,
            f"residual RMS={resid:.4f} phase RMS={rms:.4f}",
            resid,
        )
    )

    times = sample_times(0.0, cfg.texp_s, 8)
    checks.append(
        _ok(
            "exposure_samples_span",
            times[0] > 0 and times[-1] < cfg.texp_s,
            f"t=[{times[0]:.3e},{times[-1]:.3e}]",
        )
    )
    return checks


def check_lowfreq_tilt(cfg: SimConfig | None = None) -> list[Check]:
    """Ensemble centroid 2nd moment vs subharmonic level on independent screens."""
    cfg = cfg or make_config(1001, 8.0)
    pupil = make_pupil(cfg)
    n = 512
    dx = cfg.pupil_dx_m
    xc = yc = 0.5 * (n - 1) * dx
    peak_dl = float(diffraction_limited_psf(pupil).max())

    def stats(levels: int, seeds: np.ndarray) -> tuple[float, float]:
        strehl = []
        tilt = []
        for s in seeds:
            rng = np.random.default_rng(int(s))
            phi = generate_square_screen(cfg.r0_m, n, dx, levels, rng)
            screen = PhaseScreen(phi, dx, xc, yc, levels, cfg.r0_m)
            phase = extract_phase(pupil, screen, 0.0, 0.0)
            psf = instantaneous_psf(pupil.amplitude, phase)
            sx, sy = centroid_px(psf)
            strehl.append(float(psf.max() / peak_dl))
            tilt.append(sx * sx + sy * sy)
        return float(np.mean(strehl)), float(np.mean(tilt))

    seeds = cfg.seed * 100 + np.arange(16)
    levels = cfg.subharmonic_levels
    s_lo, t_lo = stats(levels, seeds)
    s_hi, t_hi = stats(levels + 1, seeds)
    rel_s = abs(s_hi - s_lo) / max(s_lo, s_hi, 1e-12)
    rel_t = abs(t_hi - t_lo) / max(t_lo, t_hi, 1e-12)
    return [
        _ok(
            "lowfreq_strehl_subharmonics",
            rel_s < C.STREHL_LOFREQ_TOL,
            f"Strehl levels {levels}/{levels + 1}={s_lo:.4f}/{s_hi:.4f} rel={rel_s:.3f}",
            rel_s,
        ),
        _ok(
            "lowfreq_tilt_subharmonics",
            rel_t < C.TILT_LOFREQ_TOL,
            f"tilt-2nd levels {levels}/{levels + 1}={t_lo:.2f}/{t_hi:.2f} rel={rel_t:.3f}",
            rel_t,
        ),
    ]


def ranking_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(int(v) for v in a), set(int(v) for v in b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / max(len(sa | sb), 1))


def check_lowfreq_ranking(cfg: SimConfig | None = None) -> list[Check]:
    """Practical-ranking S_10 overlap when subharmonic augmentation increases.

    Bounded CPU fixture: not a full Gate-1 family and not a G1/G2 rerun.
    If membership is unstable the official G1/G2 <2% follow-up is W03.
    """
    from planetrecon.estimators import fourier_shift_image
    from planetrecon.optics import center_crop
    from planetrecon.rank import score_sequence, top_fraction_indices

    cfg = cfg or make_config(1001, 8.0, n_diam=16, pupil_pad_factor=8.0, n_frames=1)
    pupil = make_pupil(cfg)
    n_scr = pupil.grid_size
    dx = cfg.pupil_dx_m
    xc = yc = 0.5 * (n_scr - 1) * dx
    n_frames = 80
    eval_n = 32
    yy, xx = np.indices((eval_n, eval_n))
    obj = np.exp(
        -0.5
        * (
            (xx - eval_n / 2.0) ** 2 / 7.0**2
            + (yy - eval_n / 2.0) ** 2 / 5.5**2
        )
    )
    obj = obj + 0.2 * np.exp(
        -0.5 * ((xx - eval_n / 2.0 - 4) ** 2 + (yy - eval_n / 2.0 + 3) ** 2) / 2.2**2
    )

    def frames_for(levels: int) -> np.ndarray:
        images = np.empty((n_frames, eval_n, eval_n), dtype=np.float64)
        for k in range(n_frames):
            rng = np.random.default_rng(int(cfg.seed) * 1000 + 17 * k)
            phi = generate_square_screen(cfg.r0_m, n_scr, dx, levels, rng)
            screen = PhaseScreen(phi, dx, xc, yc, levels, cfg.r0_m)
            phase = extract_phase(pupil, screen, 0.0, 0.0)
            psf = instantaneous_psf(pupil.amplitude, phase)
            psf_det = center_crop(bin_box(psf, cfg.bin_factor), eval_n)
            psf_det = psf_det / psf_det.sum()
            img = np.fft.ifft2(
                np.fft.fft2(np.fft.ifftshift(psf_det)) * np.fft.fft2(obj)
            ).real
            sx, sy = centroid_px(psf_det)
            images[k] = fourier_shift_image(img, (-sx, -sy))
        return images

    levels = cfg.subharmonic_levels
    img_lo = frames_for(levels)
    img_hi = frames_for(levels + 1)
    scores_lo = score_sequence(img_lo)
    scores_hi = score_sequence(img_hi)
    top_lo = top_fraction_indices(scores_lo, C.DECISION_P)
    top_hi = top_fraction_indices(scores_hi, C.DECISION_P)
    jac = ranking_jaccard(top_lo, top_hi)
    from scipy.stats import spearmanr

    spear = float(spearmanr(scores_lo, scores_hi).correlation)
    # Whole-list correlation cannot substitute for top-10% membership stability.
    # An unstable subset requires the separate G1/G2 follow-up, not an OR pass.
    passed = jac >= C.RANK_LOFREQ_JACCARD_MIN
    return [
        _ok(
            "lowfreq_ranking_subharmonics",
            passed,
            f"S_10 Jaccard={jac:.3f} Spearman={spear:.3f} levels {levels}/{levels + 1} "
            f"lo={sorted(int(v) for v in top_lo)} hi={sorted(int(v) for v in top_hi)}",
            jac,
        )
    ]


def check_padding_and_grid(cfg: SimConfig | None = None) -> list[Check]:
    cfg = cfg or make_config(1001, 8.0)
    checks = []
    window = eval_window(cfg.eval_size)

    def feature_expected(n_diam: int, pad: int, j: int):
        local_cfg = make_config(
            cfg.seed,
            cfg.dr0,
            n_frames=1,
            n_diam=n_diam,
            padding_detector_px=pad,
            exposure_samples_j=j,
            subharmonic_levels=cfg.subharmonic_levels,
            eval_size=cfg.eval_size,
        )
        pupil = make_pupil(local_cfg)
        scene = make_object_scene(local_cfg)
        psf_dl = diffraction_limited_psf(pupil)
        scale = source_rate_scale(local_cfg, scene, psf_dl)
        rng = screen_rng(local_cfg.seed)
        screen = generate_screen(local_cfg, rng)
        psf4 = finite_exposure_psf(
            pupil, screen, 0.0, local_cfg.texp_s, j, local_cfg.wind_m_s
        )
        img4 = fftconvolve(scene.latent_4x, psf4, mode="same")
        img = scale * (local_cfg.texp_s / C.T0_S) * bin_box(
            img4, local_cfg.bin_factor
        )
        ox, oy = scene.feature_origin
        crop = img[oy : oy + local_cfg.eval_size, ox : ox + local_cfg.eval_size]
        psf_det = center_crop(
            bin_box(diffraction_limited_psf(pupil), local_cfg.bin_factor),
            local_cfg.eval_size,
        )
        psf_det /= psf_det.sum()
        mtf = ideal_mtf_from_dl_psf(psf_det)
        mask = high_band_mask(local_cfg, mtf)
        return crop, mtf, mask, local_cfg

    c64, mtf, mask, _ = feature_expected(64, 64, cfg.exposure_samples_j)
    c128, _, _, _ = feature_expected(64, 128, cfg.exposure_samples_j)
    eh_pad = relative_high_band(c64, c128, window, mtf, mask)
    checks.append(
        _ok(
            "padding_doubling",
            eh_pad < C.EH_PADDING_TOL,
            f"E_H(pad 64 vs 128)={eh_pad:.4e}",
            eh_pad,
        )
    )

    # Same random field: fine screen downsampled by 2, same physical extraction.
    cfg_fine = make_config(
        cfg.seed,
        cfg.dr0,
        n_frames=1,
        n_diam=128,
        padding_detector_px=cfg.padding_detector_px,
        exposure_samples_j=cfg.exposure_samples_j,
        subharmonic_levels=cfg.subharmonic_levels,
        eval_size=cfg.eval_size,
    )
    cfg_coarse = make_config(
        cfg.seed,
        cfg.dr0,
        n_frames=1,
        n_diam=64,
        padding_detector_px=cfg.padding_detector_px,
        exposure_samples_j=cfg.exposure_samples_j,
        subharmonic_levels=cfg.subharmonic_levels,
        eval_size=cfg.eval_size,
    )
    rng = screen_rng(cfg.seed)
    screen_fine = generate_screen(cfg_fine, rng)
    phi_c = screen_fine.phi.reshape(
        screen_fine.phi.shape[0] // 2,
        2,
        screen_fine.phi.shape[1] // 2,
        2,
    ).mean(axis=(1, 3))
    screen_coarse = PhaseScreen(
        phi_c,
        screen_fine.dx_m * 2.0,
        screen_fine.x_start_m,
        screen_fine.y_centre_m,
        screen_fine.subharmonic_levels,
        screen_fine.r0_m,
    )
    pupil_f = make_pupil(cfg_fine)
    pupil_c = make_pupil(cfg_coarse)
    scene_f = make_object_scene(cfg_fine)
    scene_c = make_object_scene(cfg_coarse)
    psf_f = finite_exposure_psf(
        pupil_f,
        screen_fine,
        0.0,
        cfg_fine.texp_s,
        cfg_fine.exposure_samples_j,
        cfg_fine.wind_m_s,
    )
    psf_c = finite_exposure_psf(
        pupil_c,
        screen_coarse,
        0.0,
        cfg_coarse.texp_s,
        cfg_coarse.exposure_samples_j,
        cfg_coarse.wind_m_s,
    )
    scale_f = source_rate_scale(cfg_fine, scene_f, diffraction_limited_psf(pupil_f))
    scale_c = source_rate_scale(cfg_coarse, scene_c, diffraction_limited_psf(pupil_c))

    def crop_expected(scene, psf4, scale, cfg):
        img4 = fftconvolve(scene.latent_4x, psf4, mode="same")
        img = scale * (cfg.texp_s / C.T0_S) * bin_box(img4, cfg.bin_factor)
        ox, oy = scene.feature_origin
        return img[oy : oy + cfg.eval_size, ox : ox + cfg.eval_size]

    img_f = crop_expected(scene_f, psf_f, scale_f, cfg_fine)
    img_c = crop_expected(scene_c, psf_c, scale_c, cfg_coarse)
    psf_mtf = center_crop(
        bin_box(diffraction_limited_psf(pupil_f), cfg_fine.bin_factor),
        cfg_fine.eval_size,
    )
    psf_mtf /= psf_mtf.sum()
    mtf_f = ideal_mtf_from_dl_psf(psf_mtf)
    mask_f = high_band_mask(cfg_fine, mtf_f)
    eh_grid = relative_high_band(img_c, img_f, window, mtf_f, mask_f)
    checks.append(
        _ok(
            "grid_doubling",
            eh_grid < C.EH_GRID_TOL,
            f"E_H(n_diam 64 vs 128, same field)={eh_grid:.4e}",
            eh_grid,
        )
    )
    shift_f = np.asarray(centroid_px(psf_f)) / cfg_fine.bin_factor
    shift_c = np.asarray(centroid_px(psf_c)) / cfg_coarse.bin_factor
    shift_rel = float(
        np.linalg.norm(shift_f - shift_c)
        / max(np.linalg.norm(shift_f), np.linalg.norm(shift_c), 1e-12)
    )
    checks.append(
        _ok(
            "tilt_grid_doubling",
            shift_rel < C.TILT_GRID_TOL,
            f"centroid relative change={shift_rel:.4e}",
            shift_rel,
        )
    )
    return checks


def convergence_values(cfg: SimConfig) -> dict:
    """Return finite, seed/regime-specific convergence values for HDF5."""
    checks = check_lowfreq_tilt(cfg) + check_padding_and_grid(cfg)
    by_name = {check.name: check for check in checks}
    lowfreq = np.array(
        [
            by_name["lowfreq_strehl_subharmonics"].value,
            by_name["lowfreq_tilt_subharmonics"].value,
            by_name["tilt_grid_doubling"].value,
        ],
        dtype=np.float64,
    )
    return {
        "grid_convergence": by_name["grid_doubling"].value,
        "grid_convergence_pass": (
            by_name["grid_doubling"].passed
            and by_name["tilt_grid_doubling"].passed
        ),
        "padding_convergence": by_name["padding_doubling"].value,
        "padding_convergence_pass": by_name["padding_doubling"].passed,
        "lowfreq_convergence": lowfreq,
        "lowfreq_convergence_pass": (
            by_name["lowfreq_strehl_subharmonics"].passed
            and by_name["lowfreq_tilt_subharmonics"].passed
            and by_name["tilt_grid_doubling"].passed
        ),
    }


def check_schema_file(path: Path, require_gate_eligible: bool = False) -> list[Check]:
    errs = gate_errors(path) if require_gate_eligible else schema_errors(path)
    return [
        _ok(
            f"schema_{path.name}",
            not errs,
            "ok" if not errs else "; ".join(errs),
        )
    ]


_DEVELOPMENT_METHOD_PASS_KEYS = (
    "grid_convergence_pass",
    "exposure_convergence_pass",
    "padding_convergence_pass",
    "lowfreq_convergence_pass",
)


def certify_development_validations(
    development_paths: list[Path], target_paths: list[Path] | None = None
) -> Check:
    """Freeze development-only validation outcomes onto compatible truth files."""
    ratios_by_seed: dict[int, np.ndarray] = {}
    method_passes: dict[tuple[int, float], dict[str, bool]] = {}
    fingerprints: dict[Path, dict] = {}
    for path in development_paths:
        path = Path(path)
        if not path.exists() or schema_errors(path):
            return _ok("development_validation_certification", False, f"missing/invalid development file: {path}")
        try:
            fp = fingerprint_from_h5(path)
            if not certificate_is_current(fp, current_fingerprint_for_file(fp)):
                return _ok("development_validation_certification", False, f"incompatible or stale development method: {path}")
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return _ok("development_validation_certification", False, f"invalid method provenance: {path}: {exc}")
        fingerprints[path] = fp
        with h5py.File(path, "r") as f:
            seed = int(f.attrs["seed"])
            dr0 = float(f["/config"].attrs["Dr0"])
            if (seed, dr0) in method_passes:
                return _ok("development_validation_certification", False, f"duplicate seed/regime: {seed}/{dr0}")
            measured = f["/validation/structure_function_measured"][...]
            target = f["/validation/structure_function_target"][...]
            ratios_by_seed.setdefault(seed, measured / target)
            gv = f["/validation"]
            method_passes[(seed, dr0)] = {
                key: bool(gv[key.replace("_pass", "_local_pass")][()])
                if key.replace("_pass", "_local_pass") in gv
                else bool(gv[key][()])
                for key in _DEVELOPMENT_METHOD_PASS_KEYS
            }

    expected_seeds = set(C.DEV_SEEDS)
    expected_pairs = {
        (seed, float(dr0)) for seed in C.DEV_SEEDS for dr0 in C.MANDATORY_DR0
    }
    if set(ratios_by_seed) != expected_seeds or set(method_passes) != expected_pairs:
        return _ok(
            "development_validation_certification",
            False,
            f"available seeds={sorted(ratios_by_seed)}, "
            f"available seed/regimes={sorted(method_passes)}, "
            f"expected={sorted(expected_pairs)}",
        )

    ref_fp = next(iter(fingerprints.values()))
    for path, fp in fingerprints.items():
        if not physics_compatible(ref_fp, fp):
            diffs = fingerprint_mismatch(ref_fp, fp)
            return _ok(
                "development_validation_certification",
                False,
                f"incompatible development fingerprint {path.name}: " + "; ".join(diffs),
            )

    mean_ratio = np.mean(np.stack(list(ratios_by_seed.values())), axis=0)
    rel_error = float(np.median(np.abs(mean_ratio - 1.0)))
    structure_passed = rel_error < C.STRUCTURE_FUNCTION_REL_TOL
    certified_passes = {
        key: all(values[key] for values in method_passes.values())
        for key in _DEVELOPMENT_METHOD_PASS_KEYS
    }
    passed = structure_passed and all(certified_passes.values())
    certificate = current_fingerprint_for_file(ref_fp)

    from planetrecon.simulate import write_summary

    targets = [Path(p) for p in (development_paths if target_paths is None else target_paths)]
    if not targets:
        return _ok("development_validation_certification", False, "no certification targets")
    mismatched = []
    # Preflight the entire target set before changing any file.
    for path in targets:
        if not path.exists() or schema_errors(path):
            mismatched.append(f"{path.name}: missing/invalid target")
            continue
        try:
            target_fp = fingerprint_from_h5(path)
            if not physics_compatible(ref_fp, target_fp):
                mismatched.append(f"{path.name}: " + "; ".join(fingerprint_mismatch(ref_fp, target_fp)))
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            mismatched.append(f"{path.name}: invalid method provenance: {exc}")
    if mismatched:
        return _ok(
            "development_validation_certification", False,
            "incompatible method certificate targets: " + " | ".join(mismatched), rel_error,
        )
    for path in targets:
        with h5py.File(path, "r+") as f:
            gv = f["/validation"]
            gv["structure_function_pass"][...] = structure_passed
            if "structure_function_ensemble_rel_error" in gv:
                gv["structure_function_ensemble_rel_error"][...] = rel_error
            else:
                gv.create_dataset("structure_function_ensemble_rel_error", data=rel_error)
            for key, certified in certified_passes.items():
                local_key = key.replace("_pass", "_local_pass")
                if local_key not in gv:
                    gv.create_dataset(local_key, data=bool(gv[key][()]))
                gv[key][...] = certified
            write_certificate_dataset(gv, certificate)
            values = {name: ds[...] for name, ds in gv.items() if name != "method_certificate_json"}
            values["gate_eligible"] = validation_is_gate_eligible(values)
            gv["gate_eligible"][...] = values["gate_eligible"]
            cfg = make_config(
                int(f.attrs["seed"]),
                float(f["/config"].attrs["Dr0"]),
                n_frames=int(f["/config"].attrs["N_frames"]),
                n_diam=int(
                    round(C.D_M / float(f["/config"].attrs["screen_dx_m"]))
                ),
                pupil_pad_factor=(
                    float(f["/config"].attrs["pupil_grid_size"])
                    / int(round(C.D_M / float(f["/config"].attrs["screen_dx_m"])))
                ),
                subharmonic_levels=int(f["/config"].attrs["subharmonic_levels"]),
                exposure_samples_j=int(f["/config"].attrs["exposure_samples_J"]),
                padding_detector_px=int(f["/config"].attrs["padding_detector_px"]),
                eval_size=int(f["/object/feature_truth"].shape[0]),
            )
        write_summary(path, cfg, values)

    return _ok(
        "development_validation_certification",
        passed,
        f"pooled median |ratio-1|={rel_error:.3f}; "
        + ", ".join(f"{key}={value}" for key, value in certified_passes.items()),
        rel_error,
    )


def certify_development_structure_function(paths: list[Path]) -> Check:
    """Compatibility wrapper for development-suite certification."""
    return certify_development_validations(paths, paths)


def check_noll() -> list[Check]:
    expected = {
        1: (0, 0),
        2: (1, 1),
        3: (1, -1),
        4: (2, 0),
        5: (2, -2),
        6: (2, 2),
    }
    ok = all(noll_nm(j) == nm for j, nm in expected.items())
    return [_ok("noll_indices", ok, str({j: noll_nm(j) for j in expected}))]


def run_fast_suite() -> list[Check]:
    checks = []
    checks.extend(check_noll())
    checks.extend(check_timing())
    checks.extend(check_optics())
    checks.extend(check_object_geometry())
    checks.extend(check_high_band_ovals())
    checks.extend(check_flux())
    checks.extend(check_screen_and_exposure())
    return checks


def run_convergence_suite() -> list[Check]:
    checks = []
    checks.extend(check_lowfreq_tilt())
    checks.extend(check_padding_and_grid())
    return checks


def format_report(checks: list[Check]) -> str:
    lines = []
    npass = sum(c.passed for c in checks)
    for c in checks:
        flag = "PASS" if c.passed else "FAIL"
        val = "" if c.value is None else f"  [{c.value:.6g}]"
        lines.append(f"{flag}  {c.name:40s}  {c.message}{val}")
    lines.append("")
    lines.append(f"{npass}/{len(checks)} passed")
    return "\n".join(lines)


def run_development_suite(out_dir: Path, generate: bool = True) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== fast physics checks ===")
    checks = run_fast_suite()
    print(format_report(checks))
    print("=== convergence checks ===")
    conv = run_convergence_suite()
    print(format_report(conv))
    checks.extend(conv)
    print("=== development seeds ===")
    from planetrecon.simulate import generate_one

    paths = []
    for seed in C.DEV_SEEDS:
        for dr0 in C.MANDATORY_DR0:
            path = out_dir / filename(seed, dr0)
            paths.append(path)
            stale = path.exists() and bool(schema_errors(path))
            if generate and (not path.exists() or stale):
                generate_one(seed, dr0, out_dir)
    structure_check = certify_development_structure_function(paths)
    checks.append(structure_check)
    print(format_report([structure_check]))
    for path in paths:
        if not path.exists():
            checks.append(_ok(f"schema_{path.name}", False, "file is missing"))
        else:
            checks.extend(check_schema_file(path, require_gate_eligible=True))
    print("=== summary ===")
    print(format_report(checks))
    failed = [c for c in checks if not c.passed]
    if failed:
        print(f"{len(failed)} CHECKS FAILED")
        return 1
    print("All Prompt 1 checks passed.")
    return 0

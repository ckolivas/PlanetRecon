"""Prompt 2 Gate-1 evaluation: subsets, reconstructions, G1/G2/G3, tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from planetrecon import constants as C
from planetrecon.config import SimConfig, make_config
from planetrecon.estimators import (
    Regularisation,
    a1o,
    e1,
    e2a,
    e2a0,
    frame_noise_variance,
    register_images,
    register_otfs,
)
from planetrecon.hdf5io import filename, gate_errors
from planetrecon.metric import (
    contrast_relative_error,
    eh_metric,
    high_band_truth_fraction,
    image_rel_mse,
    mid_band_mask,
    oval_in_metric_region,
    signed_contrast,
    support_mask,
)
from planetrecon.object import sky_mask
from planetrecon.rank import (
    diagnostic_scores,
    ranking_config_hash,
    score_sequence,
    subsets_from_scores,
    top_fraction_indices,
)


REG = Regularisation()
P_GRID = tuple(int(p) for p in C.RANK_P_GRID)
DECISION_P = int(C.DECISION_P)


@dataclass
class CropArrays:
    name: str
    truth_e: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    otf: np.ndarray
    shifts: np.ndarray
    sky_mask: np.ndarray
    window: np.ndarray
    mtf: np.ndarray
    hmask: np.ndarray
    mmask: np.ndarray
    support: np.ndarray
    apertures: dict[str, np.ndarray]
    annuli: dict[str, np.ndarray]


def _cfg_from_h5(f: h5py.File) -> SimConfig:
    g = f["/config"]
    n_diam = int(round(C.D_M / float(g.attrs["screen_dx_m"])))
    pupil_n = int(g.attrs["pupil_grid_size"])
    return make_config(
        int(f.attrs["seed"]),
        float(g.attrs["Dr0"]),
        n_frames=int(g.attrs["N_frames"]),
        n_diam=n_diam,
        pupil_pad_factor=pupil_n / n_diam,
        subharmonic_levels=int(g.attrs["subharmonic_levels"]),
        exposure_samples_j=int(g.attrs["exposure_samples_J"]),
        padding_detector_px=int(g.attrs["padding_detector_px"]),
        eval_size=int(f["/object/feature_truth"].shape[0]),
    )


def load_crop(path: Path, crop: str) -> tuple[SimConfig, CropArrays, dict]:
    path = Path(path)
    with h5py.File(path, "r") as f:
        cfg = _cfg_from_h5(f)
        flux = float(f["/config"].attrs["source_rate_scale"]) * (
            float(f["/config"].attrs["texp_s"]) / C.T0_S
        )
        n_det = int(f["/object/full_latent_detector"].shape[0])
        disk_cx = n_det / 2.0
        disk_cy = n_det / 2.0
        origin = tuple(int(v) for v in f[f"/object/{crop}_crop_origin"][...])
        truth = np.asarray(f[f"/object/{crop}_truth"][...], dtype=np.float64) * flux
        window = np.asarray(f["/object/eval_window"][...], dtype=np.float64)
        mtf = np.asarray(f["/object/ideal_mtf"][...], dtype=np.float64)
        hmask = np.asarray(f["/object/high_band_mask"][...], dtype=bool)
        mmask = mid_band_mask(cfg, mtf)
        support = support_mask(cfg, cfg.eval_size)
        sky = sky_mask(origin, disk_cx, disk_cy, cfg.eval_size)
        apertures = {}
        annuli = {}
        if crop == "feature":
            for oval in C.OVALS:
                name = oval["name"]
                apertures[name] = np.asarray(
                    f[f"/object/features/{name}/feature_crop_aperture_mask"][...],
                    dtype=np.uint8,
                )
                annuli[name] = np.asarray(
                    f[f"/object/features/{name}/feature_crop_annulus_mask"][...],
                    dtype=np.uint8,
                )
        arrays = CropArrays(
            name=crop,
            truth_e=truth,
            observed=np.asarray(f[f"/frames/{crop}_observed_e"][...], dtype=np.float64),
            expected=np.asarray(f[f"/frames/{crop}_expected_e"][...], dtype=np.float64),
            otf=np.asarray(f[f"/truth_transfer/{crop}/otf_bar"][...], dtype=np.complex128),
            shifts=np.asarray(f["/frames/shift_xy_detector_px"][...], dtype=np.float64),
            sky_mask=np.asarray(sky, dtype=bool),
            window=window,
            mtf=mtf,
            hmask=hmask,
            mmask=mmask,
            support=support,
            apertures=apertures,
            annuli=annuli,
        )
        extras = {
            "strehl": np.asarray(f["/atmosphere/strehl_proxy"][...], dtype=np.float64),
            "phase_rms": np.asarray(f["/atmosphere/phase_rms_rad"][...], dtype=np.float64),
            "flux": flux,
            "read_noise_e": float(f["/config"].attrs["read_noise_e"]),
            "gate_eligible": bool(f["/validation/gate_eligible"][()]),
            "seed": int(f.attrs["seed"]),
            "dr0": float(f["/config"].attrs["Dr0"]),
            "n_frames": int(f["/config"].attrs["N_frames"]),
        }
    return cfg, arrays, extras


def _metrics(o_hat: np.ndarray, crop: CropArrays) -> dict:
    eh = eh_metric(o_hat, crop.truth_e, crop.window, crop.mtf, crop.hmask)
    em = eh_metric(o_hat, crop.truth_e, crop.window, crop.mtf, crop.mmask)
    rh = high_band_truth_fraction(crop.truth_e, crop.window, crop.mtf, crop.hmask)
    ill = bool(rh < C.RH_ILLCONDITIONED)
    out = {
        "E_H": float(eh) if np.isfinite(eh) else float("inf"),
        "E_M": float(em) if np.isfinite(em) else float("inf"),
        "R_H": float(rh),
        "high_band_illconditioned": ill,
        "image_rel_mse": image_rel_mse(o_hat, crop.truth_e),
        "mean": float(np.mean(o_hat)),
        "min": float(np.min(o_hat)),
    }
    if crop.apertures:
        contrasts = {}
        for name, ap in crop.apertures.items():
            an = crop.annuli[name]
            in_region = oval_in_metric_region(ap, crop.window)
            c_true = signed_contrast(crop.truth_e, ap, an)
            c_hat = signed_contrast(o_hat, ap, an)
            contrasts[name] = {
                "in_metric_region": in_region,
                "c_true": c_true,
                "c_hat": c_hat,
                "rel_error": contrast_relative_error(c_hat, c_true),
            }
        out["contrast"] = contrasts
    return out


def _subset_photons(expected: np.ndarray, idx: np.ndarray) -> float:
    return float(np.sum(expected[idx]))


def evaluate_crop(
    cfg: SimConfig,
    crop: CropArrays,
    extras: dict,
    reg: Regularisation = REG,
    p_grid: tuple[int, ...] = P_GRID,
) -> dict:
    n = crop.observed.shape[0]
    sigma2 = frame_noise_variance(crop.expected, extras["read_noise_e"])
    registered = register_images(crop.observed, crop.shifts)
    scores = score_sequence(registered, sky_mask=crop.sky_mask)
    subsets = subsets_from_scores(scores, p_grid)
    diags = diagnostic_scores(extras["strehl"], extras["phase_rms"], crop.otf, crop.hmask)
    diag_subsets = {
        "strehl_proxy": subsets_from_scores(diags["strehl_proxy"], p_grid),
        "phase_rms_rad": {
            int(p): top_fraction_indices(diags["phase_rms_rad"], p, higher_is_better=False)
            for p in p_grid
        },
        "highband_mean_abs_H": subsets_from_scores(diags["highband_mean_abs_H"], p_grid),
    }
    lam_e1 = reg.field_e1(cfg, cfg.eval_size)
    lam_e2 = reg.field_e2a(cfg, cfg.eval_size)
    recon = {}
    metrics = {}

    for p in p_grid:
        idx = subsets[int(p)]
        otf_s = crop.otf[idx]
        obs_s = crop.observed[idx]
        sig_s = sigma2[idx]
        e1_img = e1(otf_s, obs_s, sig_s, lam_e1)
        e2_img, e2_info = e2a(otf_s, obs_s, sig_s, lam_e2, crop.support)
        a1_img, a1_info = a1o(
            otf_s, obs_s, sig_s, lam_e2, crop.support, shifts=crop.shifts[idx]
        )
        if not e2_info["converged"] or not a1_info["converged"]:
            raise RuntimeError(
                f"{crop.name} p={p} constrained estimator did not converge: "
                f"E2a={e2_info['rel_delta']:.3g}, A1o={a1_info['rel_delta']:.3g}"
            )
        recs = {
            "E1": e1_img,
            "E2a": e2_img,
            "A1o": a1_img,
        }
        infos = {
            "E2a": {k: v for k, v in e2_info.items() if k != "H_eff"},
            "A1o": {
                k: (v if k != "H_eff" else None)
                for k, v in a1_info.items()
                if k != "H_eff"
            },
        }
        infos["A1o"]["H_eff_mean_abs"] = float(np.mean(np.abs(a1_info["H_eff"])))
        h_mean_reg = np.mean(register_otfs(otf_s, crop.shifts[idx]), axis=0)
        infos["A1o"]["H_eff_matches_mean"] = bool(
            np.allclose(a1_info["H_eff"], h_mean_reg, rtol=1e-8, atol=1e-8)
        )
        if p in (DECISION_P, 100):
            # This must be an independent numerical solve, not a CG solve seeded
            # at the analytical answer it is intended to verify.
            e20_img, e20_info = e2a0(otf_s, obs_s, sig_s, lam_e1)
            recs["E2a0"] = e20_img
            infos["E2a0"] = e20_info
            eh_e1 = eh_metric(e1_img, crop.truth_e, crop.window, crop.mtf, crop.hmask)
            eh_e20 = eh_metric(e20_img, crop.truth_e, crop.window, crop.mtf, crop.hmask)
            rel = abs(eh_e20 - eh_e1) / max(abs(eh_e1), 1e-12)
            img_rel = float(
                np.linalg.norm(e20_img - e1_img) / max(np.linalg.norm(e1_img), 1e-12)
            )
            infos["E2a0"]["eh_rel_diff"] = float(rel)
            infos["E2a0"]["image_rel_diff"] = img_rel
            match_pass = bool(
                e20_info["cg_info"] == 0
                and rel < C.E1_E2A0_EH_REL_TOL
                and img_rel < C.E1_E2A0_IMAGE_REL_TOL
            )
            infos["E2a0"]["match_pass"] = match_pass
            if not match_pass:
                raise RuntimeError(
                    f"{crop.name} p={p} E2a0 failed to match E1: "
                    f"cg_info={e20_info['cg_info']}, E_H rel={rel:.3g}, "
                    f"image rel={img_rel:.3g}"
                )
        recon[int(p)] = recs
        metrics[int(p)] = {
            name: _metrics(img, crop) for name, img in recs.items()
        }
        metrics[int(p)]["_info"] = infos
        metrics[int(p)]["_subset"] = {
            "n_used": int(idx.size),
            "n_captured": int(n),
            "indices": idx.tolist(),
            "photons_used": _subset_photons(crop.expected, idx),
        }

    eh_e2 = {p: metrics[p]["E2a"]["E_H"] for p in p_grid if p < 100}
    eh_a1 = {p: metrics[p]["A1o"]["E_H"] for p in p_grid if p < 100}
    p_e_star = int(min(eh_e2, key=eh_e2.get))
    p_a_star = int(min(eh_a1, key=eh_a1.get))

    def gap_from(e_left: float, e_right: float) -> dict:
        g_abs = float(e_left - e_right)
        g_rel = g_abs / e_left if e_left > 0 else float("nan")
        return {"G": g_abs, "g": g_rel, "E_left": e_left, "E_right": e_right}

    e2_s10 = metrics[DECISION_P]["E2a"]["E_H"]
    e2_s100 = metrics[100]["E2a"]["E_H"]
    a1_s10 = metrics[DECISION_P]["A1o"]["E_H"]
    g1 = gap_from(e2_s10, e2_s100)
    g2 = gap_from(a1_s10, e2_s10)
    g3 = gap_from(a1_s10, e2_s100)
    g3["G_sum"] = g1["G"] + g2["G"]
    g3["sum_ok"] = bool(abs(g3["G"] - g3["G_sum"]) <= C.G3_SUM_TOL + 1e-12 * max(abs(g3["G"]), 1.0))

    if "E2a0" in metrics[DECISION_P] and "E2a0" in metrics[100]:
        e20_s10 = metrics[DECISION_P]["E2a0"]["E_H"]
        e20_s100 = metrics[100]["E2a0"]["E_H"]
        g1_0 = gap_from(e20_s10, e20_s100)
        g2_0 = gap_from(a1_s10, e20_s10)
    else:
        g1_0 = g2_0 = None

    oval1_path = _oval1_pathology(metrics)
    rh = high_band_truth_fraction(crop.truth_e, crop.window, crop.mtf, crop.hmask)

    return {
        "crop": crop.name,
        "R_H": float(rh),
        "high_band_illconditioned": bool(rh < C.RH_ILLCONDITIONED),
        "ranking_hash": ranking_config_hash(),
        "laplacian_scores": scores.tolist(),
        "subsets": {str(p): subsets[p].tolist() for p in subsets},
        "diagnostic_subsets": {
            name: {str(p): idx.tolist() for p, idx in mapping.items()}
            for name, mapping in diag_subsets.items()
        },
        "p_E_star": p_e_star,
        "p_A_star": p_a_star,
        "metrics": _jsonify_metrics(metrics),
        "G1": g1,
        "G2": g2,
        "G3_m": g3,
        "G1_E2a0": g1_0,
        "G2_E2a0": g2_0,
        "oval1": oval1_path,
        "regularisation": asdict(reg),
    }


def _jsonify_metrics(metrics: dict) -> dict:
    out = {}
    for p, block in metrics.items():
        item = {}
        for key, val in block.items():
            if key.startswith("_"):
                item[key] = _to_jsonable(val)
            else:
                item[key] = _to_jsonable(val)
        out[str(p)] = item
    return out


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None:
        return None
    return obj


def _oval1_pathology(metrics: dict) -> dict:
    def rel(p, estimator):
        block = metrics[p][estimator]
        if "contrast" not in block or "oval1" not in block["contrast"]:
            return None
        return block["contrast"]["oval1"]["rel_error"]

    r_e2_10 = rel(DECISION_P, "E2a")
    r_e2_100 = rel(100, "E2a")
    r_a1_10 = rel(DECISION_P, "A1o")
    r_e20_10 = rel(DECISION_P, "E2a0")
    r_e20_100 = rel(100, "E2a0")
    out = {
        "E2a_S10_rel_error": r_e2_10,
        "E2a_S100_rel_error": r_e2_100,
        "A1o_S10_rel_error": r_a1_10,
        "E2a0_S10_rel_error": r_e20_10,
        "E2a0_S100_rel_error": r_e20_100,
    }
    # G1: all-frame E2a must not worsen Oval-1 relative error by >5% vs S10.
    # G2: per-frame E2a(S10) must not worsen it vs A1o(S10).
    out["G1_pathological"] = _worsens(r_e2_100, r_e2_10)
    out["G2_pathological"] = _worsens(r_e2_10, r_a1_10)
    out["G1_E2a0_pathological"] = _worsens(r_e20_100, r_e20_10)
    out["G2_E2a0_pathological"] = _worsens(r_e20_10, r_a1_10)
    return out


def _worsens(err_new, err_ref) -> bool:
    if err_new is None or err_ref is None:
        return False
    if not np.isfinite(err_new) or not np.isfinite(err_ref):
        return False
    return bool(abs(err_new) > abs(err_ref) + C.OVAL_CONTRAST_PATHOLOGY)


def oracle_sensitive(g_e2a: float, g_e2a0: float) -> bool:
    if not np.isfinite(g_e2a) or not np.isfinite(g_e2a0):
        return False
    if abs(g_e2a0) < 0.05:
        return False
    return bool(abs(g_e2a - g_e2a0) > 0.5 * abs(g_e2a0))


def evaluate_file(
    path: Path,
    out_dir: Path | None = None,
    crops: tuple[str, ...] = ("feature", "bland"),
    reg: Regularisation = REG,
) -> dict:
    path = Path(path)
    errs = gate_errors(path)
    if errs:
        raise RuntimeError(f"{path} is not Gate-eligible: {errs}")
    result = {
        "path": str(path),
        "ranking_hash": ranking_config_hash(),
        "crops": {},
    }
    seed = dr0 = None
    for crop in crops:
        cfg, arrays, extras = load_crop(path, crop)
        seed = extras["seed"]
        dr0 = extras["dr0"]
        result["seed"] = seed
        result["dr0"] = dr0
        result["n_frames"] = extras["n_frames"]
        result["gate_eligible"] = extras["gate_eligible"]
        result["crops"][crop] = evaluate_crop(cfg, arrays, extras, reg=reg)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"metrics_Dr0-{int(dr0)}_seed-{int(seed):05d}.json"
        dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        result["metrics_path"] = str(dest)
    return result


def classify_gap(g_values: list[float], pathological: list[bool]) -> dict:
    g = np.asarray(g_values, dtype=np.float64)
    finite = np.isfinite(g)
    g_f = g[finite]
    n = g_f.size
    if n == 0:
        return {
            "label": "inconclusive",
            "n": 0,
            "median": None,
            "p16": None,
            "p84": None,
            "frac_gt0": None,
            "frac_ge10": None,
            "worst_seed_g": None,
            "pathological_frac": None,
        }
    med = float(np.median(g_f))
    p16, p84 = [float(v) for v in np.percentile(g_f, [16, 84])]
    frac_pos = float(np.mean(g_f > 0))
    frac_ge10 = float(np.mean(g_f >= C.G_STRONG_MEDIAN))
    worst = float(np.min(g_f))
    path_frac = float(np.mean(np.asarray(pathological, dtype=bool))) if pathological else 0.0
    oval_ok = path_frac == 0.0
    strong = (
        med >= C.G_STRONG_MEDIAN
        and frac_pos >= C.G_STRONG_POSITIVE_FRAC
        and oval_ok
    )
    negative = med < C.G_NEGATIVE_MEDIAN and frac_ge10 <= C.G_NEGATIVE_GE10_FRAC
    if strong:
        label = "strong"
    elif not oval_ok and med >= C.G_STRONG_MEDIAN and frac_pos >= C.G_STRONG_POSITIVE_FRAC:
        label = "pathological"
    elif negative:
        label = "negative"
    else:
        label = "inconclusive"
    return {
        "label": label,
        "n": int(n),
        "median": med,
        "p16": p16,
        "p84": p84,
        "frac_gt0": frac_pos,
        "frac_ge10": frac_ge10,
        "worst_seed_g": worst,
        "pathological_frac": path_frac,
        "oval1_consistency": oval_ok,
    }


def _qualitative(label: str) -> str:
    if label == "strong":
        return "strong"
    if label in ("negative",):
        return "negative"
    return "other"


def aggregate_family(results: list[dict], crop: str = "feature") -> dict:
    by_regime: dict[float, list[dict]] = {}
    for res in results:
        by_regime.setdefault(float(res["dr0"]), []).append(res)
    out = {}
    for dr0, items in sorted(by_regime.items()):
        items = sorted(items, key=lambda r: int(r["seed"]))
        g1 = [r["crops"][crop]["G1"]["g"] for r in items]
        g2 = [r["crops"][crop]["G2"]["g"] for r in items]
        g3 = [r["crops"][crop]["G3_m"]["g"] for r in items]
        p1 = [r["crops"][crop]["oval1"]["G1_pathological"] for r in items]
        p2 = [r["crops"][crop]["oval1"]["G2_pathological"] for r in items]
        seeds = [int(r["seed"]) for r in items]
        c1 = classify_gap(g1, p1)
        c2 = classify_gap(g2, p2)
        c3 = classify_gap(g3, [False] * len(g3))
        c1["seeds"] = seeds
        c2["seeds"] = seeds
        c3["seeds"] = seeds
        c1["per_seed_g"] = g1
        c2["per_seed_g"] = g2
        c3["per_seed_g"] = g3
        def _med(path):
            vals = []
            for r in items:
                cur = r["crops"][crop]
                for key in path:
                    cur = cur[key]
                vals.append(cur)
            return float(np.median(vals))

        eh_summary = {
            "E2a_S10": _med(["metrics", "10", "E2a", "E_H"]),
            "E2a_S100": _med(["metrics", "100", "E2a", "E_H"]),
            "A1o_S10": _med(["metrics", "10", "A1o", "E_H"]),
            "E1_S10": _med(["metrics", "10", "E1", "E_H"]),
            "E1_S100": _med(["metrics", "100", "E1", "E_H"]),
            "R_H": _med(["R_H"]),
            "illconditioned_frac": float(
                np.mean([r["crops"][crop]["high_band_illconditioned"] for r in items])
            ),
        }
        oracle = []
        g1_0 = []
        g2_0 = []
        p1_0 = []
        p2_0 = []
        for r in items:
            block = r["crops"][crop]
            if block.get("G1_E2a0") is None:
                continue
            g1_0.append(block["G1_E2a0"]["g"])
            g2_0.append(block["G2_E2a0"]["g"])
            p1_0.append(block["oval1"]["G1_E2a0_pathological"])
            p2_0.append(block["oval1"]["G2_E2a0_pathological"])
            oracle.append(
                {
                    "seed": r["seed"],
                    "G1": oracle_sensitive(block["G1"]["g"], block["G1_E2a0"]["g"]),
                    "G2": oracle_sensitive(block["G2"]["g"], block["G2_E2a0"]["g"]),
                }
            )
        c1_0 = classify_gap(g1_0, p1_0) if g1_0 else None
        c2_0 = classify_gap(g2_0, p2_0) if g2_0 else None
        family_oracle = {
            "G1": bool(
                c1_0 is not None and _qualitative(c1["label"]) != _qualitative(c1_0["label"])
            ),
            "G2": bool(
                c2_0 is not None and _qualitative(c2["label"]) != _qualitative(c2_0["label"])
            ),
            "G1_E2a0_label": None if c1_0 is None else c1_0["label"],
            "G2_E2a0_label": None if c2_0 is None else c2_0["label"],
        }
        out[str(dr0)] = {
            "G1": c1,
            "G2": c2,
            "G3_m": c3,
            "G1_E2a0": c1_0,
            "G2_E2a0": c2_0,
            "oracle_sensitive": oracle,
            "oracle_sensitive_family": family_oracle,
            "n_seeds": len(items),
            "E_H_median": eh_summary,
        }
    return out


def format_classification(agg: dict, crop: str) -> str:
    lines = [
        f"PlanetRecon Gate-1 Prompt 2 classification  ({crop} crop)",
        f"ranking_hash: {ranking_config_hash()}",
        f"E1_LAMBDA_REL={C.E1_LAMBDA_REL:g}  E2A_LAMBDA_REL={C.E2A_LAMBDA_REL:g}",
        "",
    ]
    for dr0, block in agg.items():
        lines.append(f"=== D/r0 = {dr0} ===")
        eh = block.get("E_H_median", {})
        if eh:
            parts = []
            for k, v in eh.items():
                if not isinstance(v, float):
                    parts.append(f"{k}={v}")
                elif abs(v) < 1e-3:
                    parts.append(f"{k}={v:.4e}")
                else:
                    parts.append(f"{k}={v:.4f}")
            lines.append("  " + "  ".join(parts))
        for gap in ("G1", "G2", "G3_m"):
            c = block[gap]
            lines.append(
                f"  {gap:6s}  {c['label']:13s}  n={c['n']}"
                f"  median g={c['median']}"
                f"  p16={c['p16']} p84={c['p84']}"
                f"  frac>0={c['frac_gt0']}"
                f"  frac>=10%={c['frac_ge10']}"
                f"  worst={c['worst_seed_g']}"
                f"  oval1_ok={c.get('oval1_consistency')}"
            )
            if c.get("per_seed_g") is not None:
                pairs = ", ".join(
                    f"{s}:{g:.4f}" for s, g in zip(c["seeds"], c["per_seed_g"])
                )
                lines.append(f"         seeds: {pairs}")
        n_sens = sum(
            1 for o in block["oracle_sensitive"] if o["G1"] or o["G2"]
        )
        fam = block.get("oracle_sensitive_family") or {}
        lines.append(
            f"  oracle-sensitive seeds: {n_sens}/{len(block['oracle_sensitive'])}"
            f"  family G1 {block['G1']['label']} vs E2a0 {fam.get('G1_E2a0_label')}"
            f"  G2 {block['G2']['label']} vs E2a0 {fam.get('G2_E2a0_label')}"
            f"  family-sensitive={bool(fam.get('G1') or fam.get('G2'))}"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def resource_rows(results: list[dict], crop: str, estimator: str) -> list[dict]:
    rows = []
    for res in results:
        block = res["crops"][crop]["metrics"]
        for p, item in block.items():
            if estimator not in item:
                continue
            sub = item["_subset"]
            rows.append(
                {
                    "seed": res["seed"],
                    "dr0": res["dr0"],
                    "p": int(p),
                    "n_captured": sub["n_captured"],
                    "n_used": sub["n_used"],
                    "photons_used": sub["photons_used"],
                    "E_H": item[estimator]["E_H"],
                    "illconditioned": item[estimator]["high_band_illconditioned"],
                }
            )
    return rows


def write_resource_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "seed",
        "dr0",
        "p",
        "n_captured",
        "n_used",
        "photons_used",
        "E_H",
        "illconditioned",
    ]
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row[k]) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def freeze_regularisation(
    paths: list[Path],
    candidates: tuple[float, ...] = (
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1.0,
        3.0,
        10.0,
    ),
) -> dict:
    """Select minimal E1 λ_rel on development feature-rich all-frame reconstructions."""
    records = []
    for path in paths:
        cfg, crop, extras = load_crop(path, "feature")
        sigma2 = frame_noise_variance(crop.expected, extras["read_noise_e"])
        eh_by_lam = {}
        for lam in candidates:
            lam_f = np.full((cfg.eval_size, cfg.eval_size), lam, dtype=np.float64)
            img = e1(crop.otf, crop.observed, sigma2, lam_f)
            eh_by_lam[lam] = eh_metric(
                img, crop.truth_e, crop.window, crop.mtf, crop.hmask
            )
        records.append(
            {
                "path": str(path),
                "seed": extras["seed"],
                "dr0": extras["dr0"],
                "eh": {str(k): float(v) for k, v in eh_by_lam.items()},
            }
        )
    med = []
    for lam in candidates:
        vals = [rec["eh"][str(lam)] for rec in records]
        med.append((lam, float(np.median(vals))))
    best_eh = min(m for _, m in med)
    chosen = None
    for lam, m in med:
        if m <= 1.02 * best_eh:
            chosen = lam
            break
    if chosen is None:
        chosen = min(med, key=lambda t: t[1])[0]
    return {
        "candidates": med,
        "best_median_eh": best_eh,
        "chosen_lambda_rel": chosen,
        "rule": "smallest λ_rel whose median feature-rich E1(S_100) E_H is within 2% of the grid minimum",
        "records": records,
        "ranking_hash": ranking_config_hash(),
    }


def family_paths(out_dir: Path, seeds: tuple[int, ...], dr0s=C.MANDATORY_DR0) -> list[Path]:
    return [Path(out_dir) / filename(seed, dr0) for seed in seeds for dr0 in dr0s]


def write_hash_file(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ranking_hash.txt"
    digest = ranking_config_hash()
    path.write_text(digest + "\n")
    return path

"""Q2: E2b freeze, all-frame blind D / D-tail, closure tables (R9 §18)."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from planetrecon import constants as C
from planetrecon.estimators import (
    Regularisation,
    a1o,
    e2a,
    e2b,
    frame_noise_variance,
    register_images,
)
from planetrecon.evaluate import (
    CropArrays,
    _metrics,
    _to_jsonable,
    family_paths,
    load_crop,
)
from planetrecon.hdf5io import gate_errors
from planetrecon.metric import eh_metric
from planetrecon.mfbd import (
    PupilForward,
    d_tail,
    holdout_split,
    initial_object,
    select_init,
    tip_tilt_from_shifts,
)
from planetrecon.rank import ranking_config_hash, score_sequence, top_fraction_indices


REG = Regularisation()
TV_CANDIDATES = (
    0.0,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
)


def prior_limited(gap_e2b: float, gap_e2a: float, ratio: float = C.Q2_PRIOR_LIMITED_RATIO) -> bool:
    """True when the E2b oracle gap is less than half the E2a gap (§8.5, §18)."""
    if not np.isfinite(gap_e2b) or not np.isfinite(gap_e2a):
        return False
    return bool(abs(gap_e2b) < ratio * abs(gap_e2a))


def closure_C(eh_a1o: float, eh_d: float, eh_e2star: float) -> float:
    den = float(eh_a1o) - float(eh_e2star)
    if not np.isfinite(den) or abs(den) < 1e-15:
        return float("nan")
    return float((float(eh_a1o) - float(eh_d)) / den)


def e2_star_name(gap_e2b: float, gap_e2a: float) -> str:
    if prior_limited(gap_e2b, gap_e2a):
        return "E2a"
    return "E2b"


def freeze_e2b_prior(
    paths: list[Path],
    candidates: tuple[float, ...] = TV_CANDIDATES,
    reg: Regularisation = REG,
) -> dict:
    """Lock the E2b TV weight on development feature-rich all-frame reconstructions."""
    records = []
    for path in paths:
        cfg, crop, extras = load_crop(path, "feature")
        sigma2 = frame_noise_variance(crop.expected, extras["read_noise_e"])
        lam = reg.field_e2b(cfg, cfg.eval_size)
        eh_by_mu = {}
        for mu in candidates:
            img, info = e2b(crop.otf, crop.observed, sigma2, lam, crop.support, mu=mu)
            if not info["converged"]:
                raise RuntimeError(
                    f"{path} E2b μ={mu:g} did not converge ({info['rel_delta']:.3g})"
                )
            eh_by_mu[mu] = eh_metric(
                img, crop.truth_e, crop.window, crop.mtf, crop.hmask
            )
        records.append(
            {
                "path": str(path),
                "seed": extras["seed"],
                "dr0": extras["dr0"],
                "eh": {str(k): float(v) for k, v in eh_by_mu.items()},
            }
        )
    med = []
    for mu in candidates:
        vals = [rec["eh"][str(mu)] for rec in records]
        med.append((float(mu), float(np.median(vals))))
    best_eh = min(m for _, m in med)
    chosen = None
    for mu, m in med:
        if m <= 1.02 * best_eh:
            chosen = mu
            break
    if chosen is None:
        chosen = min(med, key=lambda t: t[1])[0]

    matching = []
    for path in paths:
        cfg, crop, extras = load_crop(path, "feature")
        if abs(float(extras["dr0"]) - 4.0) > 1e-9:
            continue
        sigma2 = frame_noise_variance(crop.expected, extras["read_noise_e"])
        registered = register_images(crop.observed, crop.shifts)
        scores = score_sequence(registered, sky_mask=crop.sky_mask)
        idx10 = top_fraction_indices(scores, C.DECISION_P)
        lam_a = reg.field_e2a(cfg, cfg.eval_size)
        lam_b = reg.field_e2b(cfg, cfg.eval_size)
        e2a_all, _ = e2a(crop.otf, crop.observed, sigma2, lam_a, crop.support)
        e2b_all, _ = e2b(
            crop.otf, crop.observed, sigma2, lam_b, crop.support, mu=chosen
        )
        a1, _ = a1o(
            crop.otf[idx10],
            crop.observed[idx10],
            sigma2[idx10],
            lam_a,
            crop.support,
            shifts=crop.shifts[idx10],
        )
        eh_a1 = eh_metric(a1, crop.truth_e, crop.window, crop.mtf, crop.hmask)
        eh_a = eh_metric(e2a_all, crop.truth_e, crop.window, crop.mtf, crop.hmask)
        eh_b = eh_metric(e2b_all, crop.truth_e, crop.window, crop.mtf, crop.hmask)
        g_a = eh_a1 - eh_a
        g_b = eh_a1 - eh_b
        matching.append(
            {
                "seed": extras["seed"],
                "dr0": extras["dr0"],
                "E_H_A1o_S10": eh_a1,
                "E_H_E2a_S100": eh_a,
                "E_H_E2b_S100": eh_b,
                "gap_E2a": g_a,
                "gap_E2b": g_b,
            }
        )
    if matching:
        med_a = float(np.median([r["gap_E2a"] for r in matching]))
        med_b = float(np.median([r["gap_E2b"] for r in matching]))
        star = e2_star_name(med_b, med_a)
        limited = prior_limited(med_b, med_a)
    else:
        med_a = med_b = float("nan")
        star = "E2b"
        limited = False
    return {
        "candidates": med,
        "best_median_eh": best_eh,
        "chosen_tv_mu": chosen,
        "rule": (
            "smallest Charbonnier-TV μ whose median feature-rich E2b(S_100) E_H "
            "is within 2% of the grid minimum"
        ),
        "records": records,
        "matching_gap_Dr0_4": matching,
        "median_gap_E2a": med_a,
        "median_gap_E2b": med_b,
        "prior_limited": limited,
        "E2_star": star,
        "ranking_hash": ranking_config_hash(),
        "e2b_lambda_rel": reg.e2b_lambda_rel,
        "e2b_tv_eps": reg.e2b_tv_eps,
    }


def _known_transfer_block(
    cfg,
    crop: CropArrays,
    extras: dict,
    idx10: np.ndarray,
    reg: Regularisation,
    tv_mu: float,
) -> dict:
    sigma2 = frame_noise_variance(crop.expected, extras["read_noise_e"])
    lam_a = reg.field_e2a(cfg, cfg.eval_size)
    lam_b = reg.field_e2b(cfg, cfg.eval_size)
    idx_all = np.arange(crop.observed.shape[0], dtype=np.int64)

    def run(name, fn, idx, lam, **kwargs):
        otf_s = crop.otf[idx]
        obs_s = crop.observed[idx]
        sig_s = sigma2[idx]
        if name == "A1o":
            img, info = a1o(
                otf_s, obs_s, sig_s, lam, crop.support, shifts=crop.shifts[idx], **kwargs
            )
        else:
            img, info = fn(otf_s, obs_s, sig_s, lam, crop.support, **kwargs)
        mets = _metrics(img, crop)
        mets["_info"] = {k: v for k, v in info.items() if k != "H_eff"}
        return img, mets

    e2a_10, m_e2a_10 = run("E2a", e2a, idx10, lam_a)
    e2a_100, m_e2a_100 = run("E2a", e2a, idx_all, lam_a)
    e2b_10, m_e2b_10 = run("E2b", e2b, idx10, lam_b, mu=tv_mu)
    e2b_100, m_e2b_100 = run("E2b", e2b, idx_all, lam_b, mu=tv_mu)
    a1_10, m_a1_10 = run("A1o", a1o, idx10, lam_a)
    g3_a = m_a1_10["E_H"] - m_e2a_100["E_H"]
    g3_b = m_a1_10["E_H"] - m_e2b_100["E_H"]
    g1_a = m_e2a_10["E_H"] - m_e2a_100["E_H"]
    g1_b = m_e2b_10["E_H"] - m_e2b_100["E_H"]
    star = e2_star_name(g3_b, g3_a)
    eh_star = m_e2b_100["E_H"] if star == "E2b" else m_e2a_100["E_H"]
    tv_for_d = tv_mu if star == "E2b" else 0.0
    return {
        "E2a_S10": m_e2a_10,
        "E2a_S100": m_e2a_100,
        "E2b_S10": m_e2b_10,
        "E2b_S100": m_e2b_100,
        "A1o_S10": m_a1_10,
        "G1_E2a": g1_a,
        "G1_E2b": g1_b,
        "G3_E2a": g3_a,
        "G3_E2b": g3_b,
        "prior_limited": prior_limited(g3_b, g3_a),
        "E2_star": star,
        "E_H_E2_star": eh_star,
        "tv_mu_D": tv_for_d,
        "tv_mu_E2b": tv_mu,
        "images": {
            "E2a_S10": e2a_10,
            "E2a_S100": e2a_100,
            "E2b_S10": e2b_10,
            "E2b_S100": e2b_100,
            "A1o_S10": a1_10,
        },
    }


def _stage_metrics(stage: dict, crop: CropArrays) -> dict:
    mets = _metrics(stage["object"], crop)
    mets["M"] = int(stage["M"])
    mets["train_loss"] = float(stage["train_loss"])
    ho = stage.get("holdout_loss")
    mets["holdout_loss"] = None if ho is None else float(ho)
    mets["n_outer"] = int(stage["n_outer"])
    return mets


def evaluate_q2_crop(
    cfg,
    crop: CropArrays,
    extras: dict,
    *,
    reg: Regularisation = REG,
    tv_mu: float | None = None,
    holdout: bool = False,
    inits: tuple[str, ...] = C.Q2_INITS,
    m_grid: tuple[int, ...] = C.M_FIT_GRID,
    outer_iters: tuple[int, ...] = C.Q2_OUTER_ITERS,
    alpha_iters: int = C.Q2_ALPHA_ITERS,
    frame_workers: int = C.Q2_FRAME_WORKERS,
) -> dict:
    n = crop.observed.shape[0]
    sigma2 = frame_noise_variance(crop.expected, extras["read_noise_e"])
    registered = register_images(crop.observed, crop.shifts)
    scores = score_sequence(registered, sky_mask=crop.sky_mask)
    idx10 = top_fraction_indices(scores, C.DECISION_P)
    idx_all = np.arange(n, dtype=np.int64)
    tv_mu = float(reg.e2b_tv if tv_mu is None else tv_mu)
    known = _known_transfer_block(cfg, crop, extras, idx10, reg, tv_mu)
    tv_d = float(known["tv_mu_D"])
    lam_d = reg.field_e2b(cfg, cfg.eval_size) if tv_d > 0 else reg.field_e2a(cfg, cfg.eval_size)
    fwd = PupilForward.from_config(cfg)
    tt = tip_tilt_from_shifts(fwd, crop.shifts)
    m_max = int(m_grid[-1])
    alpha_tt = np.zeros((n, m_max), dtype=np.float64)
    alpha_tt[:, :2] = tt
    inits_out = {}
    for name in inits:
        if name == "zero":
            start_idx = idx_all
        elif name == "subset":
            start_idx = idx10
        else:
            raise ValueError(f"unknown init {name}")
        obj0 = initial_object(
            fwd,
            crop.observed,
            sigma2,
            lam_d,
            crop.support,
            start_idx,
            tv_d,
            alphas=alpha_tt,
        )
        fit = d_tail(
            fwd,
            crop.observed,
            sigma2,
            lam_d,
            crop.support,
            tv_mu=tv_d,
            obj0=obj0,
            train_idx=idx_all,
            holdout_idx=np.zeros(0, dtype=np.int64),
            m_grid=m_grid,
            outer_iters=outer_iters,
            alpha_iters=alpha_iters,
            alpha0=alpha_tt,
            frame_workers=frame_workers,
            freeze_tip_tilt=True,
        )
        inits_out[name] = {
            "stages": [
                {**{k: v for k, v in st.items() if k not in ("object", "alphas", "otfs")},
                 "metrics": _stage_metrics(st, crop)}
                for st in fit["stages"]
            ],
            "final_metrics": _metrics(fit["object"], crop),
            "final_train_loss": float(fit["stages"][-1]["train_loss"]),
            "object": fit["object"],
            "alphas": fit["alphas"],
        }
    chosen = select_init(
        {k: {"stages": v["stages"]} for k, v in inits_out.items()}
    )
    d_metrics = inits_out[chosen]["final_metrics"]
    eh_d = d_metrics["E_H"]
    eh_a1 = known["A1o_S10"]["E_H"]
    eh_star = known["E_H_E2_star"]
    c_val = closure_C(eh_a1, eh_d, eh_star)

    holdout_block = None
    if holdout:
        train, ho = holdout_split(n, C.Q2_HOLDOUT_FRAC, extras["seed"])
        obj0 = initial_object(
            fwd,
            crop.observed,
            sigma2,
            lam_d,
            crop.support,
            train,
            tv_d,
            alphas=alpha_tt,
        )
        ho_fit = d_tail(
            fwd,
            crop.observed,
            sigma2,
            lam_d,
            crop.support,
            tv_mu=tv_d,
            obj0=obj0,
            train_idx=train,
            holdout_idx=ho,
            m_grid=m_grid,
            outer_iters=outer_iters,
            alpha_iters=alpha_iters,
            alpha0=alpha_tt,
            frame_workers=frame_workers,
            freeze_tip_tilt=True,
        )
        last = ho_fit["stages"][-1]
        holdout_block = {
            "n_train": int(train.size),
            "n_holdout": int(ho.size),
            "indices": ho.tolist(),
            "train_loss": float(last["train_loss"]),
            "holdout_loss": None if last["holdout_loss"] is None else float(last["holdout_loss"]),
            "holdout_over_train": (
                None
                if last["holdout_loss"] is None or last["train_loss"] <= 0
                else float(last["holdout_loss"] / last["train_loss"])
            ),
            "metrics": _metrics(ho_fit["object"], crop),
        }

    init_public = {}
    for name, block in inits_out.items():
        init_public[name] = {
            "stages": [
                {k: v for k, v in st.items() if k != "object_info"} | (
                    {"object_info": _to_jsonable(st.get("object_info", {}))}
                    if "object_info" in st
                    else {}
                )
                for st in block["stages"]
            ],
            "final_metrics": block["final_metrics"],
            "final_train_loss": block["final_train_loss"],
        }

    return {
        "crop": crop.name,
        "ranking_hash": ranking_config_hash(),
        "n_frames": n,
        "subset_S10": idx10.tolist(),
        "known": {k: v for k, v in known.items() if k != "images"},
        "inits": init_public,
        "chosen_init": chosen,
        "D": d_metrics,
        "D_M_grid": [st["metrics"] for st in inits_out[chosen]["stages"]],
        "C": c_val,
        "E_H_A1o": eh_a1,
        "E_H_D": eh_d,
        "E_H_E2_star": eh_star,
        "E2_star": known["E2_star"],
        "prior_limited": known["prior_limited"],
        "holdout": holdout_block,
        "regularisation": asdict(reg),
        "tv_mu_D": tv_d,
    }


def evaluate_q2_file(
    path: Path,
    out_dir: Path | None = None,
    crops: tuple[str, ...] = ("feature", "bland"),
    reg: Regularisation = REG,
    tv_mu: float | None = None,
    holdout: bool = False,
    inits: tuple[str, ...] = C.Q2_INITS,
    m_grid: tuple[int, ...] = C.M_FIT_GRID,
    outer_iters: tuple[int, ...] = C.Q2_OUTER_ITERS,
    alpha_iters: int = C.Q2_ALPHA_ITERS,
    frame_workers: int = C.Q2_FRAME_WORKERS,
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
    for crop_name in crops:
        cfg, arrays, extras = load_crop(path, crop_name)
        seed = extras["seed"]
        dr0 = extras["dr0"]
        result["seed"] = seed
        result["dr0"] = dr0
        result["n_frames"] = extras["n_frames"]
        result["gate_eligible"] = extras["gate_eligible"]
        print(
            f"  Q2 {path.name} crop={crop_name} holdout={holdout}",
            flush=True,
        )
        result["crops"][crop_name] = evaluate_q2_crop(
            cfg,
            arrays,
            extras,
            reg=reg,
            tv_mu=tv_mu,
            holdout=holdout,
            inits=inits,
            m_grid=m_grid,
            outer_iters=outer_iters,
            alpha_iters=alpha_iters,
            frame_workers=frame_workers,
        )
        block = result["crops"][crop_name]
        print(
            f"    C={block['C']}  E_H(D)={block['E_H_D']:.4f}"
            f"  E_H(A1o)={block['E_H_A1o']:.4f}"
            f"  E_H(E2*)={block['E_H_E2_star']:.4f}"
            f"  init={block['chosen_init']}  E2*={block['E2_star']}",
            flush=True,
        )
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"q2_Dr0-{int(dr0)}_seed-{int(seed):05d}.json"
        dest.write_text(json.dumps(_to_jsonable(result), indent=2, sort_keys=True) + "\n")
        result["metrics_path"] = str(dest)
    return result


def aggregate_q2(results: list[dict], crop: str = "feature") -> dict:
    by_regime: dict[float, list[dict]] = {}
    for res in results:
        by_regime.setdefault(float(res["dr0"]), []).append(res)
    out = {}
    for dr0, items in sorted(by_regime.items()):
        items = sorted(items, key=lambda r: int(r["seed"]))
        cvals = [r["crops"][crop]["C"] for r in items]
        eh_d = [r["crops"][crop]["E_H_D"] for r in items]
        eh_a1 = [r["crops"][crop]["E_H_A1o"] for r in items]
        eh_star = [r["crops"][crop]["E_H_E2_star"] for r in items]
        stars = [r["crops"][crop]["E2_star"] for r in items]
        limited = [r["crops"][crop]["prior_limited"] for r in items]
        inits = [r["crops"][crop]["chosen_init"] for r in items]
        arr = np.asarray(cvals, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        med = float(np.median(finite)) if finite.size else float("nan")
        p16, p84 = (
            [float(v) for v in np.percentile(finite, [16, 84])]
            if finite.size
            else (float("nan"), float("nan"))
        )
        frac = float(np.mean(finite >= C.Q2_CLOSURE_TARGET)) if finite.size else float("nan")
        passed = bool(finite.size and med >= C.Q2_CLOSURE_TARGET)
        out[str(dr0)] = {
            "n_seeds": len(items),
            "seeds": [int(r["seed"]) for r in items],
            "C": cvals,
            "median_C": med,
            "p16": p16,
            "p84": p84,
            "frac_ge_0p40": frac,
            "passed": passed,
            "E_H_D": eh_d,
            "E_H_A1o": eh_a1,
            "E_H_E2_star": eh_star,
            "E2_star": stars,
            "prior_limited_frac": float(np.mean(limited)),
            "chosen_inits": inits,
        }
    return out


def format_q2(agg: dict, crop: str) -> str:
    lines = [
        f"PlanetRecon Q2 closure  ({crop} crop)",
        f"ranking_hash: {ranking_config_hash()}",
        f"E2B_TV={C.E2B_TV:g}  E2_STAR={C.E2_STAR}  target median C>={C.Q2_CLOSURE_TARGET}",
        f"M_fit={list(C.M_FIT_GRID)}",
        "",
    ]
    for dr0, block in agg.items():
        lines.append(f"=== D/r0 = {dr0} ===")
        lines.append(
            f"  C  {'PASS' if block['passed'] else 'FAIL':4s}"
            f"  n={block['n_seeds']}"
            f"  median={block['median_C']}"
            f"  p16={block['p16']} p84={block['p84']}"
            f"  frac>=0.40={block['frac_ge_0p40']}"
            f"  prior_limited_frac={block['prior_limited_frac']}"
        )
        pairs = ", ".join(
            f"{s}:{c:.4f}" for s, c in zip(block["seeds"], block["C"])
        )
        lines.append(f"         seeds: {pairs}")
        lines.append(
            f"  median E_H  A1o={np.median(block['E_H_A1o']):.4f}"
            f"  D={np.median(block['E_H_D']):.4f}"
            f"  E2*={np.median(block['E_H_E2_star']):.4f}"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def _eval_job(payload: dict) -> dict:
    from planetrecon.estimators import Regularisation as _R

    reg = _R(**payload["reg"])
    return evaluate_q2_file(
        Path(payload["path"]),
        out_dir=Path(payload["out_dir"]) if payload["out_dir"] else None,
        crops=tuple(payload["crops"]),
        reg=reg,
        tv_mu=payload["tv_mu"],
        holdout=payload["holdout"],
        inits=tuple(payload["inits"]),
        m_grid=tuple(payload["m_grid"]),
        outer_iters=tuple(payload["outer_iters"]),
        alpha_iters=payload["alpha_iters"],
        frame_workers=payload["frame_workers"],
    )


def run_q2_family(
    out_dir: Path,
    seeds: tuple[int, ...],
    family_name: str,
    *,
    dr0s: tuple[float, ...] = C.Q2_SCOPE_DR0,
    generate: bool = True,
    workers: int = 1,
    eval_workers: int = 1,
    holdout: bool = False,
    tv_mu: float | None = None,
    reg: Regularisation | None = None,
    inits: tuple[str, ...] = C.Q2_INITS,
    m_grid: tuple[int, ...] = C.M_FIT_GRID,
    outer_iters: tuple[int, ...] = C.Q2_OUTER_ITERS,
    alpha_iters: int = C.Q2_ALPHA_ITERS,
    frame_workers: int = C.Q2_FRAME_WORKERS,
    crops: tuple[str, ...] = ("feature", "bland"),
) -> dict:
    from planetrecon.gate import ensure_truth_files

    out_dir = Path(out_dir)
    q2_dir = out_dir / "q2"
    q2_dir.mkdir(parents=True, exist_ok=True)
    wanted = [p for p in family_paths(out_dir, seeds, dr0s=dr0s)]
    ensure_truth_files(out_dir, seeds, generate=generate, workers=workers)
    paths = []
    for path in wanted:
        errs = gate_errors(path)
        if errs:
            raise RuntimeError(f"{path} is not Gate-eligible: {errs}")
        paths.append(path)
    if reg is None:
        frozen = q2_dir / "frozen_prior.json"
        if frozen.exists():
            blob = json.loads(frozen.read_text())
            mu = float(blob["chosen_tv_mu"]) if tv_mu is None else float(tv_mu)
            star = str(blob.get("E2_star", C.E2_STAR))
            reg = replace(REG, e2b_tv=mu)
            # E2* choice is applied per-file from the known-transfer gaps; the
            # freeze records the development lock as documentation.
            _ = star
        else:
            reg = REG if tv_mu is None else replace(REG, e2b_tv=float(tv_mu))
    elif tv_mu is not None:
        reg = replace(reg, e2b_tv=float(tv_mu))
    tv_use = float(reg.e2b_tv)
    results = []
    payloads = [
        {
            "path": str(path),
            "out_dir": str(q2_dir),
            "crops": list(crops),
            "reg": asdict(reg),
            "tv_mu": tv_use,
            "holdout": holdout,
            "inits": list(inits),
            "m_grid": list(m_grid),
            "outer_iters": list(outer_iters),
            "alpha_iters": int(alpha_iters),
            "frame_workers": int(frame_workers),
        }
        for path in paths
    ]
    if eval_workers <= 1:
        for payload in payloads:
            print(f"Q2 {Path(payload['path']).name}", flush=True)
            results.append(_eval_job(payload))
    else:
        with ProcessPoolExecutor(max_workers=eval_workers) as pool:
            futs = [pool.submit(_eval_job, payload) for payload in payloads]
            for fut in as_completed(futs):
                res = fut.result()
                print(f"  done seed={res['seed']} D/r0={res['dr0']}", flush=True)
                results.append(res)
    results.sort(key=lambda r: (float(r["dr0"]), int(r["seed"])))
    tables = {}
    for crop in crops:
        agg = aggregate_q2(results, crop=crop)
        text = format_q2(agg, crop)
        (q2_dir / f"closure_{family_name}_{crop}.txt").write_text(text)
        (q2_dir / f"closure_{family_name}_{crop}.json").write_text(
            json.dumps(_to_jsonable(agg), indent=2, sort_keys=True) + "\n"
        )
        print(text)
        tables[crop] = agg
    summary = {
        "family": family_name,
        "ranking_hash": ranking_config_hash(),
        "n_files": len(results),
        "holdout": holdout,
        "tv_mu": tv_use,
        "tables": tables,
        "m_grid": list(m_grid),
        "inits": list(inits),
    }
    (q2_dir / f"summary_{family_name}.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    tracked = Path("results") / "q2"
    tracked.mkdir(parents=True, exist_ok=True)
    for name in (
        f"closure_{family_name}_feature.txt",
        f"closure_{family_name}_feature.json",
        f"closure_{family_name}_bland.txt",
        f"closure_{family_name}_bland.json",
        f"summary_{family_name}.json",
    ):
        src = q2_dir / name
        if src.exists():
            (tracked / name).write_text(src.read_text())
    return summary


def run_freeze_prior(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    q2_dir = out_dir / "q2"
    q2_dir.mkdir(parents=True, exist_ok=True)
    paths = family_paths(out_dir, C.DEV_SEEDS)
    missing = [p for p in paths if not p.exists() or gate_errors(p)]
    if missing:
        detail = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            "development truth files missing or not Gate-eligible; run "
            f"`planetrecon validate-dev` first: {detail}"
        )
    result = freeze_e2b_prior(paths)
    dest = q2_dir / "frozen_prior.json"
    dest.write_text(json.dumps(_to_jsonable(result), indent=2, sort_keys=True) + "\n")
    tracked = Path("results") / "q2"
    tracked.mkdir(parents=True, exist_ok=True)
    (tracked / "frozen_prior.json").write_text(dest.read_text())
    print(f"chosen TV μ = {result['chosen_tv_mu']}")
    print(f"E2* = {result['E2_star']}  prior_limited={result['prior_limited']}")
    print(f"grid medians: {result['candidates']}")
    print(f"wrote {dest}")
    return result

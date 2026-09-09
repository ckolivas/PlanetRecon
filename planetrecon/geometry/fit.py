"""Disc/ellipse pose estimates and degeneracy flags. Metadata is not truth."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from planetrecon.rank import laplacian_score


def fit_disc_ellipse(image: np.ndarray) -> dict:
    """Background-subtracted centre and moment-equivalent ellipse semiaxes."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = 0.5 * img[..., 1] + 0.25 * img[..., 0] + 0.25 * img[..., 2]
    if img.size == 0 or not np.any(np.isfinite(img)):
        return {"ok": False, "degeneracy": ("no_disc",)}
    finite = img[np.isfinite(img)]
    sky = float(np.percentile(finite, 10))
    peak = float(np.percentile(finite, 99))
    thresh = sky + 0.25 * (peak - sky)
    mask = np.isfinite(img) & (img > thresh)
    if int(mask.sum()) < 9:
        return {"ok": False, "degeneracy": ("no_disc",)}
    yy, xx = np.nonzero(mask)
    w = img[mask] - sky
    cx = float(np.average(xx + 0.5, weights=w))
    cy = float(np.average(yy + 0.5, weights=w))
    dx = xx + 0.5 - cx
    dy = yy + 0.5 - cy
    mxx = float(np.average(dx * dx, weights=w))
    myy = float(np.average(dy * dy, weights=w))
    mxy = float(np.average(dx * dy, weights=w))
    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = max(tr * tr * 0.25 - det, 0.0)
    l1 = 0.5 * tr + np.sqrt(disc)
    l2 = 0.5 * tr - np.sqrt(disc)
    a = float(np.sqrt(max(4.0 * l1, 0.0)))
    b = float(np.sqrt(max(4.0 * l2, 0.0)))
    pa = 0.5 * float(np.arctan2(2.0 * mxy, mxx - myy))
    flattening = 0.0 if a <= 1e-9 else max(0.0, 1.0 - b / a)
    return {
        "ok": True,
        "cx": cx,
        "cy": cy,
        "radius": a,
        "semi_major": a,
        "semi_minor": b,
        "pa_rad": pa,
        "flattening": flattening,
        "n_pix": int(mask.sum()),
        "degeneracy": (),
    }


def estimate_field_angle(
    reference: np.ndarray,
    frame: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> dict:
    """Polar correlation around the disc. Circular featureless discs are unconstrained."""
    ref = np.asarray(reference, dtype=np.float64)
    img = np.asarray(frame, dtype=np.float64)
    if ref.ndim == 3:
        ref = 0.5 * ref[..., 1] + 0.25 * ref[..., 0] + 0.25 * ref[..., 2]
    if img.ndim == 3:
        img = 0.5 * img[..., 1] + 0.25 * img[..., 0] + 0.25 * img[..., 2]
    r_max = max(float(radius), 4.0)
    n_theta = 128
    n_r = 24
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rr = np.linspace(1.0, r_max, n_r)
    tt, rgrid = np.meshgrid(theta, rr)
    sx = rgrid * np.cos(tt)
    sy = rgrid * np.sin(tt)
    x = sx + float(cx)
    y = float(cy) - sy
    coords = np.stack([y - 0.5, x - 0.5], axis=0)
    polar_ref = map_coordinates(ref, coords, order=1, mode="constant", cval=0.0)
    polar_img = map_coordinates(img, coords, order=1, mode="constant", cval=0.0)
    # Drop the outermost limb ring so a circular edge is not treated as texture.
    polar_ref = polar_ref[:-2]
    polar_img = polar_img[:-2]
    ang_ref = polar_ref.mean(axis=0)
    ang_img = polar_img.mean(axis=0)
    mean_ref = float(np.mean(ang_ref))
    energy = float(np.var(ang_ref))
    rel_energy = energy / (mean_ref * mean_ref + 1e-15)
    disc_score = laplacian_score(ref)
    if rel_energy < 1e-4:
        return {
            "angle_rad": 0.0,
            "peak": 0.0,
            "degeneracy": ("roll_unconstrained",),
            "texture": disc_score,
            "polar_energy": energy,
            "polar_rel_energy": rel_energy,
        }
    a = ang_ref - ang_ref.mean()
    b = ang_img - ang_img.mean()
    corr = np.fft.ifft(np.fft.fft(b) * np.conj(np.fft.fft(a))).real
    peak_i = int(np.argmax(corr))
    peak = float(corr[peak_i])
    shift = peak_i if peak_i < n_theta / 2.0 else peak_i - n_theta
    # Resolve rotations smaller than one polar bin. Wrap neighbours at zero:
    # small negative rotations have their integer peak at zero too.
    left, right = corr[(peak_i - 1) % n_theta], corr[(peak_i + 1) % n_theta]
    curvature = left - 2.0 * peak + right
    if curvature < 0.0:
        shift += float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    angle = float(shift) * (2.0 * np.pi / n_theta)
    degeneracy = []
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0 or peak < 0.2 * denom:
        degeneracy.append("roll_unconstrained")
    return {
        "angle_rad": angle,
        "peak": peak,
        "degeneracy": tuple(degeneracy),
        "texture": disc_score,
        "polar_energy": energy,
        "polar_rel_energy": rel_energy,
    }


def sequence_degeneracy(
    frames: list[np.ndarray],
    *,
    circular_tol: float = 0.02,
) -> tuple[str, ...]:
    flags: list[str] = []
    if not frames:
        return ("no_disc",)
    disc = fit_disc_ellipse(frames[0])
    if not disc["ok"]:
        return tuple(disc["degeneracy"])
    if disc["flattening"] < circular_tol:
        flags.append("near_circular_disc")
    other = frames[1] if len(frames) > 1 else frames[0]
    est = estimate_field_angle(
        frames[0], other, float(disc["cx"]), float(disc["cy"]), float(disc["radius"])
    )
    flags.extend(est["degeneracy"])
    if est.get("polar_rel_energy", 1.0) < 1e-4:
        flags.append("spin_unconstrained")
    return tuple(dict.fromkeys(flags))


def _as_plane(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        return 0.5 * img[..., 1] + 0.25 * img[..., 0] + 0.25 * img[..., 2]
    return img

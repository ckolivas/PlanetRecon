"""Disc/ellipse pose estimates and degeneracy flags. Metadata is not truth."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import resample

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
    *,
    frame_center: tuple[float, float] | None = None,
) -> dict:
    """Polar correlation around the disc. Circular featureless discs are unconstrained."""
    estimate = _field_angle(reference, frame, cx, cy, radius, frame_center, dense=False)
    if not estimate['degeneracy']:
        return estimate
    # Sparse radii can miss observed narrow detail. Retry an unresolved fit
    # with detector-scale sampling and cubic interpolation of estimation pixels.
    # A blanket replacement biases already-resolved Bayer drift estimates.
    retry = _field_angle(reference, frame, cx, cy, radius, frame_center, dense=True)
    return retry if not retry['degeneracy'] else estimate


def _field_angle(reference, frame, cx, cy, radius, frame_center, *, dense):
    ref = np.asarray(reference, dtype=np.float64)
    img = np.asarray(frame, dtype=np.float64)
    if ref.ndim == 3:
        ref = 0.5 * ref[..., 1] + 0.25 * ref[..., 0] + 0.25 * ref[..., 2]
    if img.ndim == 3:
        img = 0.5 * img[..., 1] + 0.25 * img[..., 0] + 0.25 * img[..., 2]
    # Only complete circles have comparable angular support after rotation.
    # Off-detector padding is not observed dark sky: correlating it would lock
    # onto the fixed capture boundary instead of the planet's orientation.
    h, w = ref.shape
    frame_cx, frame_cy = (cx, cy) if frame_center is None else frame_center
    ih, iw = img.shape
    r_max = min(max(float(radius), 4.0), float(cx)-.5, w-.5-float(cx),
                float(cy)-.5, h-.5-float(cy), frame_cx-.5,
                iw-.5-frame_cx, frame_cy-.5, ih-.5-frame_cy)
    if r_max < 4.0:
        return {
            "angle_rad": 0.0, "peak": 0.0,
            "degeneracy": ("roll_unconstrained",),
            "texture": laplacian_score(ref),
            "polar_energy": 0.0, "polar_rel_energy": 0.0,
        }
    # Resolve detector-scale structure at the outer sampled radius. A fixed
    # 128-bin grid aliases fine angular texture on larger planets, potentially
    # reporting a different rotation direction or a distant correlation peak.
    n_theta = max(128, 1 << int(np.ceil(np.log2(2*np.pi*r_max))))
    # Keep the existing limb exclusion (the last two of 24 radii), but resolve
    # detector-scale radial structure inside it. Sparse rings can miss visible
    # narrow features entirely and incorrectly report an unconstrained angle.
    sample_radius = 1.0 + (r_max-1.0)*21.0/23.0 if dense else r_max
    n_r = max(22, int(np.ceil(sample_radius))) if dense else 24
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rr = np.linspace(1.0, sample_radius, n_r)
    tt, rgrid = np.meshgrid(theta, rr)
    sx = rgrid * np.cos(tt)
    sy = rgrid * np.sin(tt)
    x = sx + float(cx)
    y = float(cy) - sy
    coords = np.stack([y - 0.5, x - 0.5], axis=0)
    order = 3 if dense else 1
    polar_ref = map_coordinates(ref, coords, order=order, mode="constant", cval=0.0)
    # Sample the original frame about its own centre. Recentering an image
    # first both interpolates twice and fills detector regions never observed.
    frame_coords = coords + np.array([frame_cy-cy, frame_cx-cx])[:, None, None]
    polar_img = map_coordinates(img, frame_coords, order=order, mode="constant", cval=0.0)
    if not dense:
        polar_ref = polar_ref[:-2]
        polar_img = polar_img[:-2]
    # Remove the radial brightness profile, then correlate matching radii.
    # Averaging rings first cancels real angular structure when features at
    # different radii have opposite contrast or phase.
    a = polar_ref - polar_ref.mean(axis=1, keepdims=True)
    b = polar_img - polar_img.mean(axis=1, keepdims=True)
    mean_ref = float(np.mean(polar_ref))
    energy = float(np.mean(a*a))
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
    corr = np.fft.ifft(np.sum(np.fft.fft(b, axis=1)
        * np.conj(np.fft.fft(a, axis=1)), axis=0)).real
    # Compare peaks between angular bins before choosing a winner. With fine
    # repeated texture, the correct peak can lie between bins while a weaker
    # competing peak happens to land on one. Refining only that winner is too late.
    corr = resample(corr, 8*n_theta)
    n_theta = len(corr)
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

"""Deterministic R9 synthetic Jupiter-like object, crops, and masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from planetrecon import constants as C
from planetrecon.config import SimConfig
from planetrecon.optics import bin_box


@dataclass
class ObjectScene:
    latent_4x: np.ndarray
    latent_detector: np.ndarray
    disk_cx_det: float
    disk_cy_det: float
    feature_origin: tuple[int, int]
    bland_origin: tuple[int, int]
    reference_disk_mask: np.ndarray
    feature_truth: np.ndarray
    bland_truth: np.ndarray
    sky_mask_feature: np.ndarray


def _highres_coords(n4: int, oversample: int):
    c = n4 / 2.0
    yy, xx = np.indices((n4, n4), dtype=np.float64)
    x = (xx + 0.5 - c) / oversample
    y = (yy + 0.5 - c) / oversample
    return x, y


def _limb_darkening(r: np.ndarray, radius: float, u: float) -> np.ndarray:
    mu2 = 1.0 - (r / radius) ** 2
    inside = mu2 >= 0.0
    mu = np.zeros_like(r)
    mu[inside] = np.sqrt(mu2[inside])
    L = np.zeros_like(r)
    L[inside] = 1.0 - u * (1.0 - mu[inside])
    return L


def _belts(y: np.ndarray, radius: float) -> np.ndarray:
    yn = y / radius
    return (
        1.0
        + 0.07 * np.cos(3.0 * np.pi * yn)
        - 0.04 * np.cos(7.0 * np.pi * yn)
        + 0.025 * np.sin(11.0 * np.pi * yn)
    )


def _oval_multiplier(x: np.ndarray, y: np.ndarray, oval: dict) -> np.ndarray:
    th = np.deg2rad(oval["angle_deg"])
    dx = x - oval["x_px"]
    dy = y - oval["y_px"]
    xp = dx * np.cos(th) + dy * np.sin(th)
    yp = -dx * np.sin(th) + dy * np.cos(th)
    q = 0.5 * (
        (xp / oval["sigma_x_px"]) ** 2 + (yp / oval["sigma_y_px"]) ** 2
    )
    return 1.0 + oval["contrast"] * np.exp(-q)


def ovals_inside_disk() -> bool:
    r = C.PLANET_RADIUS_DET_PX
    for oval in C.OVALS:
        extent = 3.0 * max(oval["sigma_x_px"], oval["sigma_y_px"])
        rc = float(np.hypot(oval["x_px"], oval["y_px"]))
        if rc + extent >= r:
            return False
    return True


def latent_object_4x(
    n4: int, oversample: int = C.OBJECT_OVERSAMPLE, include_ovals: bool = True
) -> np.ndarray:
    x, y = _highres_coords(n4, oversample)
    r = np.hypot(x, y)
    obj = _limb_darkening(r, C.PLANET_RADIUS_DET_PX, C.LIMB_DARKENING_U)
    obj *= np.clip(_belts(y, C.PLANET_RADIUS_DET_PX), 1e-6, None)
    if include_ovals:
        for oval in C.OVALS:
            obj *= _oval_multiplier(x, y, oval)
    obj = np.clip(obj, 0.0, None)
    obj[r > C.PLANET_RADIUS_DET_PX] = 0.0
    return obj


def _crop(img: np.ndarray, origin: tuple[int, int], size: int) -> np.ndarray:
    ox, oy = origin
    return img[oy : oy + size, ox : ox + size]


def feature_crop_origin(disk_cx: float, disk_cy: float) -> tuple[int, int]:
    # Place +x limb near local x=100 and centre the crop in y.
    ox = int(round(disk_cx - 20.0))
    oy = int(round(disk_cy - 64.0))
    return ox, oy


def bland_crop_origin(disk_cx: float, disk_cy: float) -> tuple[int, int]:
    # R9 positions the central 96×96 wholly inside the disk and between
    # all three planted oval centres.
    ox = int(round(disk_cx - 72.0))
    oy = int(round(disk_cy - 54.0))
    return ox, oy


def feature_measurement_masks(
    origin: tuple[int, int],
    disk_cx: float,
    disk_cy: float,
    oval: dict,
    size: int = C.EVAL_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the locked 2-sigma aperture and 3--5-sigma annulus masks."""
    ox, oy = origin
    yy, xx = np.indices((size, size), dtype=np.float64)
    x = ox + xx + 0.5 - disk_cx
    y = oy + yy + 0.5 - disk_cy
    th = np.deg2rad(oval["angle_deg"])
    dx = x - oval["x_px"]
    dy = y - oval["y_px"]
    xp = dx * np.cos(th) + dy * np.sin(th)
    yp = -dx * np.sin(th) + dy * np.cos(th)
    radius = np.sqrt(
        (xp / oval["sigma_x_px"]) ** 2
        + (yp / oval["sigma_y_px"]) ** 2
    )
    aperture = radius <= C.MEASUREMENT_APERTURE_SIGMA
    annulus = (
        (radius >= C.MEASUREMENT_ANNULUS_INNER_SIGMA)
        & (radius <= C.MEASUREMENT_ANNULUS_OUTER_SIGMA)
    )
    return aperture.astype(np.uint8), annulus.astype(np.uint8)


def source_rate_scale(
    cfg: SimConfig, scene: ObjectScene, psf_dl_4x: np.ndarray
) -> float:
    """Scale the fixed source rate to 800 e-/px at the reference exposure."""
    img4 = fftconvolve(scene.latent_4x, psf_dl_4x, mode="same")
    img = bin_box(img4, C.OBJECT_OVERSAMPLE)
    crop = _crop(img, scene.feature_origin, cfg.eval_size)
    mean = float(crop[scene.reference_disk_mask.astype(bool)].mean())
    if mean <= 0:
        raise RuntimeError("reference-mask mean is zero")
    return C.REF_MEAN_E_AT_T0 / mean


def reference_disk_mask(
    origin: tuple[int, int],
    disk_cx: float,
    disk_cy: float,
    size: int = C.EVAL_SIZE,
) -> np.ndarray:
    ox, oy = origin
    yy, xx = np.indices((size, size), dtype=np.float64)
    x = ox + xx + 0.5
    y = oy + yy + 0.5
    r = np.hypot(x - disk_cx, y - disk_cy)
    return (r <= (C.PLANET_RADIUS_DET_PX - C.LIMB_EXCLUSION_PX)).astype(np.uint8)


def sky_mask(
    origin: tuple[int, int],
    disk_cx: float,
    disk_cy: float,
    size: int = C.EVAL_SIZE,
) -> np.ndarray:
    ox, oy = origin
    yy, xx = np.indices((size, size), dtype=np.float64)
    x = ox + xx + 0.5
    y = oy + yy + 0.5
    r = np.hypot(x - disk_cx, y - disk_cy)
    return r > C.PLANET_RADIUS_DET_PX


def make_object_scene(cfg: SimConfig) -> ObjectScene:
    n4 = cfg.object_n4
    over = C.OBJECT_OVERSAMPLE
    if n4 % over:
        raise ValueError("object grid is not aligned to detector pixels")
    latent_4x = latent_object_4x(n4, over)
    latent_det = bin_box(latent_4x, over)
    n_det = latent_det.shape[0]
    disk_cx = n_det / 2.0
    disk_cy = n_det / 2.0
    feat_o = feature_crop_origin(disk_cx, disk_cy)
    bland_o = bland_crop_origin(disk_cx, disk_cy)
    size = cfg.eval_size
    for name, orig in (("feature", feat_o), ("bland", bland_o)):
        ox, oy = orig
        if ox < 0 or oy < 0 or ox + size > n_det or oy + size > n_det:
            raise RuntimeError(f"{name} crop {orig} does not fit in {n_det}x{n_det}")
    ref = reference_disk_mask(feat_o, disk_cx, disk_cy, size)
    return ObjectScene(
        latent_4x=latent_4x,
        latent_detector=latent_det,
        disk_cx_det=disk_cx,
        disk_cy_det=disk_cy,
        feature_origin=feat_o,
        bland_origin=bland_o,
        reference_disk_mask=ref,
        feature_truth=_crop(latent_det, feat_o, size),
        bland_truth=_crop(latent_det, bland_o, size),
        sky_mask_feature=sky_mask(feat_o, disk_cx, disk_cy, size),
    )


def tukey_2d(n: int, alpha: float = C.TUKEY_ALPHA) -> np.ndarray:
    from scipy.signal.windows import tukey

    w = tukey(n, alpha=alpha, sym=True)
    return np.outer(w, w)

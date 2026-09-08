"""Explicit forward and adjoint operators for the W01 consistency audit.

The frozen estimator still uses crop-sized circular Fourier products. This
module exposes that model, the simulator's padded linear convolution, and
crop/bin adjoints so mismatch can be measured without silently changing the
R9 statistical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from planetrecon import constants as C
from planetrecon.optics import bin_box, otf_from_centered_psf


def circular_convolve(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """Crop-sized circular convolution in the estimator FFT layout."""
    image = np.asarray(image, dtype=np.float64)
    otf = np.asarray(otf, dtype=np.complex128)
    return np.fft.ifft2(otf * np.fft.fft2(image)).real


def circular_convolve_adjoint(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    return circular_convolve(image, np.conj(otf))


def circular_psf_convolve(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    return circular_convolve(image, otf_from_centered_psf(psf))


def _same_slices(in_shape: tuple[int, int], h_shape: tuple[int, int]):
    full = (in_shape[0] + h_shape[0] - 1, in_shape[1] + h_shape[1] - 1)
    start = ((full[0] - in_shape[0]) // 2, (full[1] - in_shape[1]) // 2)
    sl = (
        slice(start[0], start[0] + in_shape[0]),
        slice(start[1], start[1] + in_shape[1]),
    )
    return sl, full


def linear_convolve_same(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Linear convolution, ``mode='same'``, matching the simulator."""
    image = np.asarray(image, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)
    return np.asarray(fftconvolve(image, psf, mode="same"), dtype=np.float64)


def linear_convolve_same_adjoint(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Adjoint of :func:`linear_convolve_same` with respect to the image."""
    y = np.asarray(image, dtype=np.float64)
    h = np.asarray(psf, dtype=np.float64)
    # A[j,i] = h[j + floor((hsize-1)/2) - i]. Its transpose is
    # convolution with the reversed kernel, cropped at ceil((hsize-1)/2).
    # scipy chooses efficient padded FFT sizes; the previous explicit FFT on
    # n+h-1 can have costly prime dimensions at full optical resolution.
    acc = fftconvolve(y, h[::-1, ::-1], mode="full")
    sy, sx = h.shape[0]//2, h.shape[1]//2
    return np.asarray(acc[sy:sy+y.shape[0], sx:sx+y.shape[1]], dtype=np.float64)


def spatial_convolve_same(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    circular: bool = False,
) -> np.ndarray:
    """Independent real-space convolution used as an audit oracle."""
    image = np.asarray(image, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)
    ny, nx = image.shape
    py, px = psf.shape
    # scipy's linear 'same' crop starts at floor((kernel_size - 1)/2).
    # Circular FFT centring instead uses floor(kernel_size/2).
    cy, cx = (py // 2, px // 2) if circular else ((py - 1) // 2, (px - 1) // 2)
    out = np.zeros((ny, nx), dtype=np.float64)
    for y in range(ny):
        for x in range(nx):
            acc = 0.0
            for i in range(py):
                yy = y + cy - i
                for j in range(px):
                    xx = x + cx - j
                    if circular:
                        acc += psf[i, j] * image[yy % ny, xx % nx]
                    elif 0 <= yy < ny and 0 <= xx < nx:
                        acc += psf[i, j] * image[yy, xx]
            out[y, x] = acc
    return out


def crop_xy(image: np.ndarray, origin_xy, size: int) -> np.ndarray:
    x0, y0 = int(origin_xy[0]), int(origin_xy[1])
    return np.asarray(image, dtype=np.float64)[y0 : y0 + size, x0 : x0 + size]


def crop_xy_adjoint(crop: np.ndarray, origin_xy, full_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(full_shape, dtype=np.float64)
    x0, y0 = int(origin_xy[0]), int(origin_xy[1])
    h, w = crop.shape
    out[y0 : y0 + h, x0 : x0 + w] = np.asarray(crop, dtype=np.float64)
    return out


def bin_box_adjoint(detector: np.ndarray, factor: int) -> np.ndarray:
    det = np.asarray(detector, dtype=np.float64)
    return np.repeat(np.repeat(det, factor, axis=0), factor, axis=1)


def fourier_shift_image(image: np.ndarray, shift_xy) -> np.ndarray:
    from scipy.ndimage import fourier_shift

    sx, sy = float(shift_xy[0]), float(shift_xy[1])
    shifted = fourier_shift(np.fft.fft2(np.asarray(image, dtype=np.float64)), shift=(sy, sx))
    return np.fft.ifft2(shifted).real


def fourier_shift_adjoint(image: np.ndarray, shift_xy) -> np.ndarray:
    return fourier_shift_image(image, (-float(shift_xy[0]), -float(shift_xy[1])))


def quadratic_gradient(
    image: np.ndarray,
    otfs: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
) -> np.ndarray:
    """Unconstrained gradient of the E1/E2a0 quadratic."""
    from planetrecon.estimators import wiener_num_den

    num, den = wiener_num_den(otfs, images, sigma2, lam_f)
    of = np.fft.fft2(np.asarray(image, dtype=np.float64))
    return np.fft.ifft2(den * of - num).real


def interior_mask(n: int, border: int) -> np.ndarray:
    mask = np.ones((n, n), dtype=bool)
    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False
    return mask


def relative_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if mask is not None:
        a = a[mask]
        b = b[mask]
    den = float(np.linalg.norm(b))
    if den <= 0:
        den = float(np.linalg.norm(a)) or 1.0
    return float(np.linalg.norm(a - b) / den)


def adjoint_relative_error(
    forward,
    adjoint,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """| <y, A x> - <A* y, x> | / (||x|| ||y||)."""
    ax = np.asarray(forward(x), dtype=np.float64)
    aty = np.asarray(adjoint(y), dtype=np.float64)
    lhs = float(np.vdot(y.ravel(), ax.ravel()).real)
    rhs = float(np.vdot(aty.ravel(), x.ravel()).real)
    scale = float(np.linalg.norm(x) * np.linalg.norm(y))
    if scale <= 0:
        return abs(lhs - rhs)
    return abs(lhs - rhs) / scale


def padded_detector_crop(
    scene_4x: np.ndarray,
    psf4: np.ndarray,
    origin_xy,
    eval_size: int,
    bin_factor: int,
    flux: float,
) -> np.ndarray:
    """Simulator observation mean: pad-convolve, bin, crop, source-rate scale."""
    img4 = linear_convolve_same(scene_4x, psf4)
    img = float(flux) * bin_box(img4, bin_factor)
    return crop_xy(img, origin_xy, eval_size)


def padded_detector_crop_adjoint(
    residual: np.ndarray,
    psf4: np.ndarray,
    origin_xy,
    scene_shape: tuple[int, int],
    bin_factor: int,
    flux: float,
) -> np.ndarray:
    det_ny = scene_shape[0] // bin_factor
    det_nx = scene_shape[1] // bin_factor
    det = crop_xy_adjoint(residual, origin_xy, (det_ny, det_nx))
    optical = bin_box_adjoint(float(flux) * det, bin_factor)
    return linear_convolve_same_adjoint(optical, psf4)


@dataclass(frozen=True)
class InteriorMismatch:
    interior_rel: float
    full_rel: float
    border_rel: float
    interior_ok: bool


def circular_vs_linear_mismatch(
    image: np.ndarray,
    psf: np.ndarray,
    border: int = 6,
    interior_tol: float = 0.02,
) -> InteriorMismatch:
    linear = linear_convolve_same(image, psf)
    circular = circular_psf_convolve(image, psf)
    mask = interior_mask(image.shape[0], border)
    interior = relative_l2(circular, linear, mask)
    full = relative_l2(circular, linear)
    border_rel = relative_l2(circular, linear, ~mask)
    return InteriorMismatch(
        interior_rel=interior,
        full_rel=full,
        border_rel=border_rel,
        interior_ok=interior < interior_tol,
    )


def scalar_noise_mismatch(
    expected: np.ndarray,
    read_rms: float = C.READ_NOISE_E,
) -> dict[str, float]:
    """Compare the frozen scalar variance to the heteroscedastic Poisson+read map.

    Does not change the statistical model.
    """
    true_var = np.clip(np.asarray(expected, dtype=np.float64), 0.0, None) + float(read_rms) ** 2
    scalar = float(np.mean(np.clip(expected, 0.0, None)) + float(read_rms) ** 2)
    var_rel_rms = float(np.sqrt(np.mean((true_var - scalar) ** 2)) / max(scalar, 1e-15))
    w_true = 1.0 / np.clip(true_var, C.DEN_FLOOR, None)
    w_scalar = 1.0 / max(scalar, C.DEN_FLOOR)
    weight_rel_rms = float(np.sqrt(np.mean((w_true - w_scalar) ** 2)) / w_scalar)
    dynamic = float(true_var.max() / max(true_var.min(), 1e-15))
    return {
        "scalar_variance": scalar,
        "variance_rel_rms": var_rel_rms,
        "weight_rel_rms": weight_rel_rms,
        "true_var_dynamic_range": dynamic,
        "n_pixels": float(true_var.size),
    }


def translate_cells(image: np.ndarray, shift_xy, *, adjoint=False) -> np.ndarray:
    """Zero-extended bilinear flux deposition and its exact transpose."""
    x = np.asarray(image, dtype=np.float64)
    sx, sy = map(float, shift_xy)
    if not np.isfinite([sx, sy]).all():
        raise ValueError('translation must be finite')
    ix, iy = int(np.floor(sx)), int(np.floor(sy))
    fx, fy = sx-ix, sy-iy
    out = np.zeros_like(x)
    h, w = x.shape[:2]
    for dx, wx in ((ix, 1-fx), (ix+1, fx)):
        for dy, wy in ((iy, 1-fy), (iy+1, fy)):
            if wx*wy == 0 or abs(dx) >= w or abs(dy) >= h:
                continue
            src = (slice(max(0, -dy), min(h, h-dy)), slice(max(0, -dx), min(w, w-dx)))
            dst = (slice(max(0, dy), min(h, h+dy)), slice(max(0, dx), min(w, w+dx)))
            if adjoint:
                out[src] += wx*wy*x[dst]
            else:
                out[dst] += wx*wy*x[src]
    return out


@dataclass(frozen=True)
class SceneDetectorOperator:
    """Extended optical scene → exposure → detector integration/crop/mask/CFA.

    See docs/scene-detector-contract.md. Arrays are copied and made read-only so
    an operator cannot silently change during a solve. RGB channels share a PSF;
    chromatic PSFs and non-translation geometry are not yet implemented here.
    """
    scene_shape: tuple
    psfs: tuple
    bin_factor: int
    origin_xy: tuple
    detector_shape: tuple
    flux: float = 1.
    shifts_xy: tuple | None = None
    exposure_weights: tuple | None = None
    valid_mask: np.ndarray | None = None
    cfa_pattern: str | None = None
    cfa_offset_xy: tuple = (0, 0)

    def __post_init__(self):
        shape = tuple(self.scene_shape)
        if len(shape) not in (2, 3) or (len(shape) == 3 and shape[2] != 3):
            raise ValueError('scene must be HxW or HxWx3')
        if any(int(v) != v or v < 1 for v in shape):
            raise ValueError('scene dimensions must be positive integers')
        b = self.bin_factor
        if int(b) != b or b < 1 or shape[0] % b or shape[1] % b:
            raise ValueError('scene must be divisible by positive integer bin factor')
        origin, det = tuple(self.origin_xy), tuple(self.detector_shape)
        if len(origin) != 2 or len(det) != 2 or any(int(v) != v for v in (*origin, *det)):
            raise ValueError('crop dimensions and origin must be integers')
        ox, oy = origin
        if min(origin) < 0 or min(det) < 1 or oy+det[0] > shape[0]//b or ox+det[1] > shape[1]//b:
            raise ValueError('detector crop outside scene')
        if not np.isfinite(self.flux) or self.flux <= 0:
            raise ValueError('flux must be finite and positive')
        kernels = tuple(np.array(p, dtype=np.float64, copy=True) for p in self.psfs)
        if not kernels or any(p.ndim != 2 or not p.size or not np.isfinite(p).all()
                              or p.min() < 0 or not np.isclose(p.sum(), 1., atol=1e-12, rtol=1e-10)
                              for p in kernels):
            raise ValueError('PSFs must be finite nonnegative unit-energy 2D kernels')
        shifts = tuple((0., 0.) for _ in kernels) if self.shifts_xy is None else tuple(tuple(s) for s in self.shifts_xy)
        weights = tuple(1./len(kernels) for _ in kernels) if self.exposure_weights is None else tuple(self.exposure_weights)
        if len(shifts) != len(kernels) or any(len(s) != 2 for s in shifts) or not np.isfinite(shifts).all():
            raise ValueError('one finite xy shift per exposure sample is required')
        if len(weights) != len(kernels) or not np.isfinite(weights).all() or min(weights) < 0 or not np.isclose(sum(weights), 1., atol=1e-12, rtol=0):
            raise ValueError('exposure weights must be nonnegative and sum to one')
        mask = np.ones(det, dtype=bool) if self.valid_mask is None else np.array(self.valid_mask, copy=True)
        if mask.shape != det or mask.dtype != np.bool_:
            raise ValueError('valid mask must be a boolean detector-shaped array')
        if self.cfa_pattern is not None and (len(shape) != 3 or self.cfa_pattern not in ('RGGB', 'BGGR', 'GRBG', 'GBRG')):
            raise ValueError('CFA requires RGB scene and valid Bayer pattern')
        if len(self.cfa_offset_xy) != 2 or any(int(v) != v for v in self.cfa_offset_xy):
            raise ValueError('CFA parity offset must be integer xy')
        for name, value in (('scene_shape', tuple(map(int, shape))), ('psfs', kernels),
                            ('bin_factor', int(b)), ('origin_xy', tuple(map(int, origin))),
                            ('detector_shape', tuple(map(int, det))), ('shifts_xy', shifts),
                            ('exposure_weights', weights), ('valid_mask', mask),
                            ('cfa_offset_xy', tuple(map(int, self.cfa_offset_xy)))):
            object.__setattr__(self, name, value)
        for arr in (*kernels, mask):
            arr.flags.writeable = False

    @property
    def output_shape(self):
        return self.detector_shape + ((3,) if len(self.scene_shape) == 3 and self.cfa_pattern is None else ())

    def _channels(self, image, operation):
        if image.ndim == 2:
            return operation(image)
        return np.stack([operation(image[..., c]) for c in range(3)], axis=-1)

    def _cfa_indices(self):
        y, x = np.indices(self.detector_shape)
        ox, oy = np.add(self.origin_xy, self.cfa_offset_xy)
        tile = np.array(['RGB'.index(c) for c in self.cfa_pattern]).reshape(2, 2)
        return tile[(y+oy) % 2, (x+ox) % 2]

    def forward(self, scene):
        scene = np.asarray(scene, dtype=np.float64)
        if scene.shape != self.scene_shape:
            raise ValueError('wrong scene shape')
        optical = np.zeros_like(scene)
        for psf, shift, weight in zip(self.psfs, self.shifts_xy, self.exposure_weights):
            optical += weight*self._channels(translate_cells(scene, shift), lambda x: linear_convolve_same(x, psf))
        detector = self.flux*self._channels(optical, lambda x: bin_box(x, self.bin_factor))
        ox, oy = self.origin_xy
        h, w = self.detector_shape
        result = detector[oy:oy+h, ox:ox+w]
        if self.cfa_pattern is not None:
            result = np.take_along_axis(result, self._cfa_indices()[..., None], axis=2)[..., 0]
        return result * (self.valid_mask[..., None] if result.ndim == 3 else self.valid_mask)

    def adjoint(self, residual):
        residual = np.asarray(residual, dtype=np.float64)
        if residual.shape != self.output_shape:
            raise ValueError('wrong detector shape')
        residual = residual * (self.valid_mask[..., None] if residual.ndim == 3 else self.valid_mask)
        if self.cfa_pattern is not None:
            rgb = np.zeros(self.detector_shape+(3,))
            np.put_along_axis(rgb, self._cfa_indices()[..., None], residual[..., None], axis=2)
            residual = rgb
        det_shape = (self.scene_shape[0]//self.bin_factor, self.scene_shape[1]//self.bin_factor)
        optical = self._channels(residual, lambda x: bin_box_adjoint(
            crop_xy_adjoint(x, self.origin_xy, det_shape), self.bin_factor))
        result = np.zeros(self.scene_shape)
        for psf, shift, weight in zip(self.psfs, self.shifts_xy, self.exposure_weights):
            acc = self._channels(optical, lambda x: linear_convolve_same_adjoint(x, psf))
            result += self.flux*weight*translate_cells(acc, shift, adjoint=True)
        return result

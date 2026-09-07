"""Compare an unsharpened result to a 16-bit RGB PNG reference, without fitting detail.

Run with the project GUI venv. Position and per-channel gain/offset are nuisance
fits; no filtering, sharpening or local warp is used to match the result. This
is a development comparison, not independent reconstruction qualification.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, shift
from planetrecon.pipeline.align import phase_correlation_shift
from planetrecon.result import load_snapshot


def read_reference(path):
    from PySide6.QtGui import QImage
    image = QImage(str(path))
    if image.isNull():
        raise ValueError('cannot read reference image')
    # Preserve all 16 bits; Pillow's RGB loader would reduce a 16-bit PNG to 8 bits.
    image = image.convertToFormat(QImage.Format.Format_RGBA64)
    pixels = np.frombuffer(image.constBits(), dtype=np.uint16).reshape(image.height(), image.bytesPerLine()//2)
    return pixels[:, :image.width()*4].reshape(image.height(), image.width(), 4)[..., :3].astype(float)


def compare(reference, result):
    if result.image.shape != reference.shape or result.incomplete:
        raise ValueError('comparison requires a complete RGB stack on the reference grid')
    if result.provenance.get('sharpening'):
        raise ValueError('use an unsharpened result')
    dx, dy = phase_correlation_shift(reference[..., 1], result.image[..., 1])
    aligned = shift(result.image, (-dy, -dx, 0), order=1, mode='constant', prefilter=False)
    valid = shift(result.validity.astype(float), (-dy, -dx, 0), order=1, mode='constant', prefilter=False) > 1-1e-9
    mask = binary_erosion(reference[..., 1] > reference[..., 1].max()*.15, iterations=10)
    mask &= valid.all(axis=2) & np.isfinite(aligned).all(axis=2)
    if mask.sum() < 100:
        raise ValueError('insufficient common interior support')
    fit = []
    for c in range(3):
        a, b = np.linalg.lstsq(np.column_stack([aligned[..., c][mask], np.ones(mask.sum())]),
                              reference[..., c][mask], rcond=None)[0]
        aligned[..., c] = aligned[..., c]*a + b
        fit.append([float(a), float(b)])
    hp_ref = reference-gaussian_filter(reference, (3, 3, 0))
    hp_image = aligned-gaussian_filter(aligned, (3, 3, 0))
    metrics = {
        'role': 'development comparison; reference is not ground truth',
        'reference_precision': '16-bit RGB', 'n_used': result.n_used, 'n_rejected': result.n_rejected,
        'interior_pixels': int(mask.sum()), 'registration_xy_px': [dx, dy],
        'per_channel_gain_offset': fit,
        'relative_rmse': float(np.linalg.norm((aligned-reference)[mask])/np.linalg.norm(reference[mask])),
        'highpass_correlation_sigma3': float(np.corrcoef(hp_ref[mask].ravel(), hp_image[mask].ravel())[0, 1]),
        'registration_method': result.provenance.get('registration'),
    }
    return metrics, aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ref = read_reference(args.reference)
    metrics, aligned = compare(ref, load_snapshot(args.result))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(args.out/'comparison.npz', reference=ref, aligned=aligned)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

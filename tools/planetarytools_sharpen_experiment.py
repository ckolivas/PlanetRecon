"""Replay the user's exact two-step PlanetaryTools sharpening workflow.

Run using the PlanetaryTools Python environment. All outputs go to a new
directory; neither repository's production code nor source images are edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planetary-tools', type=Path, required=True,
                        help='Path to PlanetaryTools/planetary-app')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify', nargs=2, action='append', default=[],
                        metavar=('INPUT', 'EXPECTED_SHARPENED'))
    parser.add_argument('images', type=Path, nargs='+')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.planetary_tools.resolve()))
    from planetary_tools.io.loader import load_image, save_image
    from planetary_tools.io.png_read import read_png_rgb16
    from planetary_tools.filters.registry import FILTERS, apply_filter
    wavelet = dict(FILTERS['wavelet_sharpen'].default_params,
                   fine=27., medium=0., coarse=0., chunky=0., auto=False)
    deconv = dict(FILTERS['adaptive_deconv'].default_params,
                  amount=15.6, adaptive=True, auto=False)
    report = dict(planetary_tools=str(args.planetary_tools.resolve()),
        planetary_tools_commit=subprocess.check_output(
            ['git', '-C', str(args.planetary_tools), 'rev-parse', 'HEAD'], text=True).strip(),
        wavelet=wavelet, deconvolution=deconv, images={})
    report['implementation_sha256'] = {
        name: hashlib.sha256((args.planetary_tools/'planetary_tools'/name).read_bytes()).hexdigest()
        for name in ('io/loader.py', 'io/png_read.py', 'io/png_write.py',
                     'filters/registry.py', 'filters/wavelet.py', 'filters/adaptive_deconv.py',
                     'core/colour.py', 'core/brightness.py')}
    verified = {str(Path(a).resolve()): Path(b) for a, b in args.verify}
    if len({p.stem for p in args.images}) != len(args.images):
        raise ValueError('Input stems must be unique')
    for path in args.images:
        doc = load_image(path, pin_noise=False)
        stage1 = apply_filter('wavelet_sharpen', doc.data, doc.is_grayscale, wavelet)
        stage2 = apply_filter('adaptive_deconv', stage1, doc.is_grayscale, deconv)
        assert np.isfinite(stage1).all() and np.isfinite(stage2).all()
        np.savez_compressed(args.out/f'{path.stem}_stages.npz', wavelet=stage1, sharpened=stage2)
        doc.set_data(stage2)
        output = args.out/f'{path.stem}_sharpened.png'
        save_image(doc, output, bit_depth=16)
        item = dict(source=str(path.resolve()),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output=str(output.resolve()), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        expected = verified.get(str(path.resolve()))
        if expected is not None:
            actual, reference = read_png_rgb16(output), read_png_rgb16(expected)
            difference = actual.astype(float)-reference
            item['verification'] = dict(expected=str(expected.resolve()),
                maximum_code_difference=float(abs(difference).max()),
                rms_code_difference=float(np.sqrt(np.mean(difference**2))),
                different_channel_samples=int(np.count_nonzero(difference)))
            np.testing.assert_array_equal(actual, reference)
        report['images'][path.stem] = item
        print(json.dumps(item), flush=True)
    (args.out/'sharpening_report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()

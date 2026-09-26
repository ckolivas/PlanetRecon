"""Controlled mono-capture experiment; instruments, but does not edit, the stacker.

Run with PYTHONPATH=. .venv/bin/python tools/alignment_noise_experiment.py
--capture CAPTURE --export ORIGINAL_PLANETRECON_PNG --out NEW_DIRECTORY.
Requires CUDA. All variants share the production reference and accepted frames.
Integer translation is a diagnostic control, not a proposed production method.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from PySide6.QtGui import QImage
from scipy.ndimage import center_of_mass, gaussian_filter, laplace

from planetrecon.export import ExportConfig, export_result
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.pipeline.preprocess_cache import load_cache
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import save_snapshot
from planetrecon.runtime import apply_thread_limits


def read_png(path):
    original = QImage(str(path))
    if original.isNull():
        raise ValueError(f'Cannot read {path}')
    image = original.convertToFormat(QImage.Format.Format_Grayscale16)
    array = np.frombuffer(image.constBits(), dtype=np.uint16).reshape(
        image.height(), image.bytesPerLine() // 2)[:, :image.width()].astype(float)
    return array, original.text('PlanetRecon')


def measurements(image, *, difference=None):
    """Use the same relative disc patches as the earlier export comparisons.

For split differences, divide (half0-half1) by two: independent equal halves
then estimate the random component of their full mean. Shared reference noise
and fixed structure cancel, so this does not measure every possible artifact.
"""
    cy, cx = np.round(center_of_mass(np.maximum(image-image.max()*.02, 0))).astype(int)
    output = {}
    for name, dy in [('upper', (-55, -30)), ('lower', (35, 60))]:
        section = np.s_[cy+dy[0]:cy+dy[1], cx-40:cx+40]
        roi = image[section] if difference is None else difference[section]
        level = float(np.median(image[section]))
        residual = (roi-gaussian_filter(roi, 2))[4:-4, 4:-4]
        curvature = laplace(roi)[2:-2, 2:-2]
        output[name] = dict(median=level,
            highpass_percent=float(100*1.4826*np.median(abs(residual-np.median(residual)))/level),
            laplacian_percent=float(100*1.4826*np.median(abs(curvature-np.median(curvature)))/np.sqrt(20)/level))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    apply_thread_limits(8)
    import torch
    from planetrecon.backends.torch_accel import TorchBackend
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this controlled experiment')
    torch.set_num_threads(8)
    pixels, text = read_png(args.export)
    metadata = json.loads(text)
    config = replace(ReconstructionConfig.from_dict(metadata['result']['provenance']['config']),
                     device='gpu', threads=8)
    if not config.local_alignment or config.alignment_method != 'circular_multiscale':
        raise ValueError('Supply an export made with circular multiscale alignment')
    mapping = metadata['mapping']
    assert mapping['gamma'] == 1
    names = ('local', 'global_subpixel', 'global_integer')
    started = time.perf_counter()
    with SERSource(args.capture) as source:
        assert source.color_mode() == 'mono'
        assert not any(getattr(config, k) is not None for k in
                       ('bias_path', 'dark_path', 'flat_path', 'gain_e_per_adu'))
        selection, status = load_cache(source, config)
        if selection is None:
            raise ValueError(status)
        indices = np.flatnonzero(best_frame_mask(selection, config.stack_percent, config.frame_selection_mode))
        shape = source.frame_shape()
        sums = {name: np.zeros((2, *shape)) for name in names}
        weights = {name: np.zeros((2, *shape)) for name in names}
        shifts = []
        field_rms = []
        state = dict(count=0, ready=False)
        original_project = TorchBackend.backproject

        class ProbeMatcher(CircularMultiscaleRegistration):
            def __init__(self, reference, **kwargs):
                super().__init__(reference, **kwargs)
                np.save(args.out/'reference.npy', reference)
                state['reference_sha256'] = hashlib.sha256(reference.tobytes()).hexdigest()

            def displacement_tensor(self, frame, global_shift):
                assert not state['ready']
                index = int(indices[state['count']])
                # Verify ordering against the actual input, not just a counter.
                np.testing.assert_array_equal(frame, source.read_raw(index))
                state['shift'] = tuple(global_shift)
                shifts.append(global_shift)
                result = super().displacement_tensor(frame, global_shift)
                residual = result[:, 140:270, 290:410] - result.new_tensor(global_shift)[:, None, None]
                field_rms.append(float(residual.square().mean().sqrt().cpu()))
                state['ready'] = True
                return result

        def project(backend, raw, shift, color):
            assert state['ready'] and color == 'mono'
            index = int(indices[state['count']])
            np.testing.assert_array_equal(raw, source.read_raw(index))
            score = max(selection.measurements[index, 0], 1e-12) if config.quality_weighting else 1.
            half = state['count'] % 2
            local = original_project(backend, raw, shift, color)
            # Only the displacement supplied to the same raw backprojector changes.
            for name, projected in [('local', local),
                    ('global_subpixel', original_project(backend, raw, state['shift'], color)),
                    ('global_integer', original_project(backend, raw, np.rint(state['shift']), color))]:
                sums[name][half] += score*projected[0]
                weights[name][half] += score*projected[1]
            state['count'] += 1
            state['ready'] = False
            if state['count'] % 256 == 0:
                print(json.dumps(dict(processed=state['count'], total=len(indices),
                                      seconds=time.perf_counter()-started)), flush=True)
            return local

        with patch('planetrecon.pipeline.circular_align.CircularMultiscaleRegistration', ProbeMatcher), \
                patch.object(TorchBackend, 'backproject', project):
            result = stack_source(source, config, preprocessing=selection)
        assert result.backend == 'cuda' and result.n_used == len(indices) == state['count']
        assert not result.incomplete
        np.savez_compressed(args.out/'diagnostics.npz', indices=indices, global_shifts=shifts,
                            field_rms=field_rms, quality=selection.measurements[indices, 0])
        report = dict(config=config.to_dict(), n_used=result.n_used,
            reference_sha256=state['reference_sha256'],
            reference_candidates=result.provenance['local_alignment']['template_candidates'],
            seconds=time.perf_counter()-started, gpu=torch.cuda.get_device_name(0),
            split='alternating selected frames in capture order; shared reference', variants={})
        for name in names:
            total, weight = sums[name].sum(axis=0), weights[name].sum(axis=0)
            image = np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)
            halves = np.divide(sums[name], weights[name], out=np.zeros_like(sums[name]), where=weights[name] > 0)
            if name == 'local':
                # Different summation grouping should affect only rounding.
                np.testing.assert_allclose(image, result.image, rtol=1e-12, atol=1e-10)
                quantum = (mapping['white']-mapping['black'])/65535
                decoded = pixels*quantum + mapping['black']
                supported = (pixels > 0) & (pixels < 65535)
                report['existing_export_parity'] = dict(quantum=quantum,
                    maximum_difference_adu=float(np.max(abs(image[supported]-decoded[supported]))),
                    rms_difference_adu=float(np.sqrt(np.mean((image[supported]-decoded[supported])**2))))
            provenance = deepcopy(result.provenance)
            provenance['experiment'] = dict(variant=name, shared_reference_sha256=state['reference_sha256'],
                displacement='production local field' if name == 'local' else name,
                note='Diagnostic override; original configuration retained for reproduction')
            product = replace(result, image=image, coverage=weight, validity=weight > 0, provenance=provenance)
            save_snapshot(args.out/f'{name}.npz', product)
            np.savez_compressed(args.out/f'{name}_halves.npz', images=halves, coverage=weights[name])
            export_result(product, args.out/f'{name}.png', ExportConfig('png16', mapping['black'], mapping['white']))
            report['variants'][name] = dict(full=measurements(image),
                split_difference=measurements(image, difference=(halves[0]-halves[1])/2))
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

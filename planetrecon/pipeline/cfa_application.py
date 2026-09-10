"""Application ownership of local CFA moments and completed-frame provenance."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import numpy as np
import scipy

from planetrecon.pipeline.cfa_checkpoint import array_digest, map_digest, validate_gpu_state
from planetrecon.pipeline.cfa_interpolation import LocalQuadraticAccumulator
from planetrecon.provenance import source_hash


MOMENTS = ('gram', 'noise', 'rhs', 'variance_sum')


def execution_identity(config):
    # Frozen builds have no package .py files to hash; bind the executable
    # containing their module archive as well as the numerical runtime.
    binary = None
    if getattr(sys, 'frozen', False):
        with Path(sys.executable).open('rb') as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
            binary = digest.hexdigest()
    return {'executable_sha256': binary, 'operator': 'local-cfa-radius4-noise-cap-1', 'package_sha256': source_hash(),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'python': sys.version, 'byteorder': sys.byteorder, 'threads': config.threads}


def registration_device(backend):
    if backend.name == 'cpu':
        return {'backend': 'cpu'}
    import torch
    p = torch.cuda.get_device_properties(0)
    return {'backend': 'cuda', 'name': p.name, 'capability': [p.major, p.minor],
            'multiprocessors': p.multi_processor_count,
            'torch': str(torch.__version__), 'cuda': torch.version.cuda}


class LocalCFA:
    def __init__(self, shape, color, config, backend, report, should_cancel):
        options = dict(should_cancel=should_cancel)
        if config.max_ram_bytes is not None:
            options['max_array_bytes'] = min(config.max_ram_bytes, 2*1024**3)
        if config.device == 'cpu':
            self.model = LocalQuadraticAccumulator(shape, color, **options)
        else:
            from planetrecon.pipeline.cfa_cuda import CudaLocalAccumulator
            if config.max_vram_bytes is not None:
                options['max_gpu_array_bytes'] = min(config.max_vram_bytes, 128*1024**2)
            self.model = CudaLocalAccumulator(shape, color, **options)
            if backend.name != 'cuda':
                self.model.gpu_enabled = False
                self.model.fallback_frame = 0
                self.model.fallback_reason = report.reason or 'CUDA unavailable at job start'
        self.frames = []
        self.registration_device = registration_device(backend)

    def add(self, index, raw, shift, quality):
        # The caller owns these arrays until add returns. GPU retry uses this
        # same raw frame and displacement, never a newly estimated local map.
        entry = {'index': int(index), 'quality': float(quality),
                 'raw_sha256': array_digest(raw), 'map_sha256': map_digest(shift, self.model.shape)}
        frames = self.frames+[entry]
        self.model.add(raw, shift, quality)
        self.frames = frames

    def state(self):
        state = {'frames': self.frames, 'registration_device': self.registration_device}
        if hasattr(self.model, 'gpu_enabled'):
            state['gpu_state'] = {key: getattr(self.model, key) for key in (
                'gpu_enabled', 'fallback_frame', 'fallback_reason', 'device_identity')}
            state['execution_history'] = self.model.execution_history
        else:
            state['execution_history'] = ['cpu']*self.model.n_used
        return deepcopy(state)

    def restore(self, saved, accum, weight, n_used, next_index, selection, backend):
        state = saved['local_cfa']
        frames = state.get('frames')
        if not isinstance(frames, list) or len(frames) != n_used:
            raise ValueError('local CFA checkpoint frame count mismatch')
        previous = -1
        for entry in frames:
            if (not isinstance(entry, dict) or set(entry) != {'index', 'quality', 'raw_sha256', 'map_sha256'}
                    or type(entry['index']) is not int or not previous < entry['index'] < next_index
                    or not selection.accepted[entry['index']]
                    or entry['quality'] != max(selection.measurements[entry['index'], 0], 1e-12)):
                raise ValueError('local CFA checkpoint selection mismatch')
            for key in ('raw_sha256', 'map_sha256'):
                value = entry[key]
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('invalid local CFA input digest')
            previous = entry['index']
        history = state.get('execution_history')
        if hasattr(self.model, 'gpu_enabled'):
            gpu_state = validate_gpu_state(state.get('gpu_state'), history, n_used)
            for key, value in gpu_state.items():
                setattr(self.model, key, value)
            self.model.expected_device_identity = gpu_state['device_identity']
            self.model.execution_history = history.copy()
        elif history != ['cpu']*n_used or 'gpu_state' in state:
            raise ValueError('invalid local CFA CPU execution history')
        # Remaining maps must use the saved registration execution. A CPU
        # fallback is sticky across resume; a different CUDA runtime is refused.
        if state.get('registration_device') != registration_device(backend):
            raise ValueError('local CFA registration device/runtime differs from checkpoint')
        self.registration_device = state['registration_device']
        for name in MOMENTS:
            setattr(self.model, name, saved['local_'+name])
        self.model.sums, self.model.weights = accum, weight
        self.model.n_used = n_used
        self.frames = deepcopy(frames)

    def describe(self, result, *, final=False):
        details = {'method': 'reference-coordinate quadratic, then affine under ordinary iid-noise variance cap, then ordinary RGB completion',
                   'radius_px': 4, 'preview': not final,
                   'preview_method': 'ordinary local CFA stack while interpolation moments accumulate',
                   'completed_frames': self.model.n_used,
                   'frame_map_manifest': deepcopy(self.frames),
                   'moment_execution_history': getattr(self.model, 'execution_history', ['cpu']*self.model.n_used),
                   'fallback_reason': getattr(self.model, 'fallback_reason', None),
                   'coverage': 'ordinary colour support; direct detector weights remain in cfa_direct_R/G/B',
                   'noise_model': 'independent equal-variance detector samples; relative variance is not calibrated photon/read noise'}
        if final:
            image, variance, degree, ordinary = self.model.finish()
            result.image, result.coverage, result.validity = image, ordinary.coverage, ordinary.validity
            result.layer_coverage = ordinary.layer_coverage
            effective = np.divide(1., variance, out=np.zeros_like(variance), where=(variance > 0) & result.validity)
            for c, name in enumerate('RGB'):
                result.layer_coverage['iid_effective_samples_'+name] = effective[..., c]
            details['fit_channel_counts'] = {name: int(np.count_nonzero(degree == code))
                for code, name in [(-1, 'unsupported'), (0, 'ordinary'), (1, 'affine'), (2, 'quadratic')]}
            details['iid_effective_samples'] = '1 / relative estimator variance, in separate layers; not frame count or direct CFA coverage'
        result.provenance['local_cfa_interpolation'] = details
        if details['fallback_reason'] and details['fallback_reason'] not in result.warnings:
            result.warnings.append(details['fallback_reason'])

"""Observed-data-only development inputs and explicit latent-cell representations."""
from dataclasses import fields
import json
from pathlib import Path
import h5py
import numpy as np
from planetrecon.config import SimConfig
from planetrecon.atmosphere import generate_screen, finite_exposure_psf
from planetrecon.optics import make_pupil, bin_box
from planetrecon.rng import screen_rng
from planetrecon.operators import SceneDetectorOperator
from tools.input_compatibility import archived_input_errors
from tools.scene_fft import SceneFFTBatch


TRAIN_SMALL = (0, 249, 499)
TRAIN_LARGE = (0, 55, 111, 166, 222, 249, 277, 333, 388, 444, 499)
SELECTION = (125, 374)
ASSESSMENT = (62, 187, 312, 437)


class ObservedSceneData:
    def __init__(self, path, indices, crop='feature'):
        if errors := archived_input_errors(path):
            raise ValueError(errors)
        if crop not in ('feature','bland'):
            raise ValueError('invalid crop')
        indices = tuple(sorted(set(indices)))
        with h5py.File(path) as f:
            raw = json.loads(f['config/json_utf8'][()])
            self.cfg = SimConfig(**{k.name: raw[k.name] for k in fields(SimConfig)})
            if not indices or any(int(i)!=i or i<0 or i>=self.cfg.n_frames for i in indices):
                raise ValueError('invalid frame indices')
            # No latent truth values or expected-image datasets are read.
            self.native_shape = f['object/full_latent_4x'].shape
            self.origin = tuple(f[f'object/{crop}_crop_origin'][...])
            self.images = dict(zip(indices, f[f'frames/{crop}_observed_e'][list(indices)].astype(float)))
            times = f['atmosphere/frame_time_s'][list(indices)]
            self.read_noise = float(f['config'].attrs['read_noise_e'])
        pupil, screen = make_pupil(self.cfg), generate_screen(self.cfg, screen_rng(self.cfg.seed))
        self.psfs = {i: finite_exposure_psf(pupil,screen,t,self.cfg.texp_s,
                     self.cfg.exposure_samples_j,self.cfg.wind_m_s) for i,t in zip(indices,times)}
        self.variances = {i: max(float(image.mean()),0.)+self.read_noise**2 for i,image in self.images.items()}

    def operators(self, indices, *, margin=None, factor=1):
        b, size = self.cfg.bin_factor, self.cfg.eval_size
        ox, oy = self.origin
        if margin is None:
            x0,y0,x1,y1 = 0,0,self.native_shape[1]//b,self.native_shape[0]//b
        else:
            if int(margin)!=margin or margin<0:
                raise ValueError('nonnegative integer detector margin required')
            x0,y0 = max(0,ox-margin),max(0,oy-margin)
            x1,y1 = min(self.native_shape[1]//b,ox+size+margin),min(self.native_shape[0]//b,oy+size+margin)
        base = [SceneDetectorOperator(((y1-y0)*b,(x1-x0)*b),(self.psfs[i],),b,
                                      (ox-x0,oy-y0),(size,size)) for i in indices]
        return [CellBasisOperator(op,factor) for op in base]


def expand_cells(x, factor):
    return np.repeat(np.repeat(x,factor,axis=0),factor,axis=1)/factor**2


def reduce_cells(x, factor):
    if x.ndim == 2:
        return bin_box(x,factor)/factor**2
    return np.stack([bin_box(x[...,c],factor)/factor**2 for c in range(x.shape[2])],axis=-1)


class CellBasisOperator:
    def __init__(self, base, factor):
        if int(factor)!=factor or factor<1 or any(n%factor for n in base.scene_shape[:2]):
            raise ValueError('cell factor must divide native scene dimensions')
        self.base,self.factor = base,int(factor)
        self.scene_shape = tuple(n//factor for n in base.scene_shape[:2])+base.scene_shape[2:]
        self.output_shape,self.valid_mask = base.output_shape,base.valid_mask

    def forward(self,x):
        return self.base.forward(expand_cells(x,self.factor))

    def adjoint(self,y):
        return reduce_cells(self.base.adjoint(y),self.factor)

    def detector_scene(self,x):
        base=self.base
        expanded=expand_cells(x,self.factor)
        detector=bin_box(expanded,base.bin_factor) if expanded.ndim==2 else np.stack([bin_box(expanded[...,c],base.bin_factor) for c in range(expanded.shape[2])],axis=-1)
        ox,oy=base.origin_xy;h,w=base.detector_shape
        return detector[oy:oy+h,ox:ox+w]


class CellFFTBatch:
    def __init__(self,operators,**kwargs):
        self.operators=tuple(operators)
        if not self.operators or len({o.factor for o in self.operators})!=1:
            raise ValueError('common cell factor required')
        self.factor=self.operators[0].factor
        self.native=SceneFFTBatch([o.base for o in self.operators],**kwargs)

    def forward(self,x):
        return self.native.forward(expand_cells(x,self.factor))

    def adjoint(self,ys):
        return reduce_cells(self.native.adjoint(ys),self.factor)

    def normal(self,x,weights):
        return reduce_cells(self.native.normal(expand_cells(x,self.factor),weights),self.factor)

    def cache_info(self):
        return self.native.cache_info()

    def clear_cache(self):
        self.native.clear_cache()


def prior_coefficient(sum_native_strength, n_frames, factor=1):
    if not np.isfinite(sum_native_strength) or sum_native_strength<=0 or n_frames<1 or factor<1:
        raise ValueError('positive prior, frame count and sampling factor required')
    return float(sum_native_strength)/(n_frames*factor**2)


def prediction_score(operators,images,variances,x,batch=None):
    predictions=batch.forward(x) if batch is not None else [o.forward(x) for o in operators]
    return float(np.mean([np.mean((p-y)**2/v) for p,y,v in zip(predictions,images,variances)]))


def select_candidate(rows, relative_tie=1e-8):
    if not rows or any(not r['numerical_passed'] or not np.isfinite(r['selection_score']) for r in rows):
        raise ValueError('all candidates must be numerically qualified with finite selection scores')
    minimum=min(r['selection_score'] for r in rows)
    tied=[r for r in rows if r['selection_score']<=minimum+relative_tie*max(abs(minimum),1e-30)]
    return max(tied,key=lambda r:r['sum_native_strength'])['sum_native_strength']

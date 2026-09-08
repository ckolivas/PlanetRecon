"""Bounded R10 development ablations, explicitly incapable of authorizing Q3.

Run with python -m planetrecon.science_audit --out NEW_DIRECTORY.
The reduced grids, frame count and optimizer budgets are diagnostics only.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import h5py
import numpy as np

from planetrecon import constants as C
from planetrecon.config import make_config
from planetrecon.estimators import e1, e2a0, frame_noise_variance, Regularisation
from planetrecon.evaluate import load_crop, _to_jsonable
from planetrecon.mfbd import PupilForward, tip_tilt_from_shifts, data_residual
from planetrecon.optics import centroid_px
from planetrecon.pipeline.align import phase_correlation_shift as phase_correlation
from planetrecon.provenance import experiment_manifest, sha256_bytes, write_manifest
from planetrecon.q2 import evaluate_q2_crop
from planetrecon.simulate import simulate
from planetrecon.validate import check_lowfreq_ranking


def run(directory: Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    protocol = dict(seeds=list(C.DEV_SEEDS), regimes=[4.,8.], crops=['feature','bland'],
        n_frames=8, n_diam=16, pupil_pad_factor=8., eval_size=32,
        stages=list(C.M_FIT_GRID), outer_iters=[1,1,1], alpha_iters=2,
        inits=['zero','subset'], holdout_role='model_selection',
        assessment_role='separate frames; phase-profiled residual with frozen selected training object',
        partition_fraction=C.Q2_HOLDOUT_FRAC,
        gates_enabled=False)
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=[]
    started=time.monotonic()
    for seed in C.DEV_SEEDS:
        for dr0 in (4.,8.):
            print(f'audit seed={seed} D/r0={dr0}',flush=True)
            cfg=make_config(seed,dr0,n_frames=8,n_diam=16,pupil_pad_factor=8.,eval_size=32)
            path=simulate(cfg,directory/'inputs')
            fwd=PupilForward.from_config(cfg)
            with h5py.File(path,'r') as file:
                # Projected physical screen phases and exposure averaged transfer
                # come from the simulator, not a fitted phase-generating model.
                coeff=file['/atmosphere/kl60_coeff'][...]
                basis_residual=file['/atmosphere/kl60_residual_rms'][...]
            for crop_name in protocol['crops']:
                cfg,crop,extras=load_crop(path,crop_name)
                sigma=frame_noise_variance(crop.expected,extras['read_noise_e'])
                lam=Regularisation().field_e2a(cfg,cfg.eval_size)
                closed=e1(crop.otf,crop.observed,sigma,lam)
                iterative,info=e2a0(crop.otf,crop.observed,sigma,lam)
                agreement=float(np.linalg.norm(closed-iterative)/max(np.linalg.norm(closed),1e-12))
                tt=tip_tilt_from_shifts(fwd,crop.shifts)
                actual=np.array([centroid_px(fwd.psf_det(a)) for a in tt])
                estimated=np.array([phase_correlation(crop.observed[0],v) for v in crop.observed])
                estimated += crop.shifts[0] # common origin for this diagnostic only
                controls={}
                for name,alphas in [('known_shift',tt),('estimated_shift',tip_tilt_from_shifts(fwd,estimated)),('no_shift',np.zeros_like(tt))]:
                    controls[name]=data_residual(fwd.otfs(alphas),crop.truth_e,crop.observed,sigma)
                fit=evaluate_q2_crop(cfg,crop,extras,holdout=True,m_grid=C.M_FIT_GRID,
                    outer_iters=(1,1,1),alpha_iters=2,frame_workers=1,tv_mu=0.,reference_star='E2b')
                record={'seed':seed,'dr0':dr0,'crop':crop_name,'status':'diagnostic',
                    'config':json.loads(cfg.json_utf8()),'input_sha256':sha256_bytes(path.read_bytes()),
                    'phase_basis_residual_rms':basis_residual.tolist(),
                    'E1_E2a0_relative_error':agreement,'E2a0':info,
                    'shift_range_px':[crop.shifts.min(axis=0).tolist(),crop.shifts.max(axis=0).tolist()],
                    'shift_to_pupil_max_centroid_error_px':float(np.linalg.norm(actual-crop.shifts,axis=1).max()),
                    'fixed_truth_object_shift_control_residuals':controls,'q2':fit}
                if coeff is not None:
                    snapshot=fwd.otfs(coeff)
                    record['snapshot_exposure_otf_relative_error']=float(np.linalg.norm(snapshot-crop.otf)/np.linalg.norm(crop.otf))
                dest=directory/f'audit-{seed}-{int(dr0)}-{crop_name}.json'
                dest.write_text(json.dumps(_to_jsonable(record),indent=2,allow_nan=False)+'\n')
                records.append({'seed':seed,'dr0':dr0,'crop':crop_name,'path':dest.name,
                    'sha256':sha256_bytes(dest.read_bytes()),'status':'diagnostic',
                    'fit_status':fit['status'],'closure':fit['C'],'E1_E2a0_relative_error':agreement,
                    'shift_error_px':record['shift_to_pupil_max_centroid_error_px']})
                print(f'  {crop_name}: {fit["status"]}; E1/E2a0={agreement:.3g}',flush=True)
    ranking=[]
    for seed in C.DEV_SEEDS:
        for dr0 in (4.,8.):
            cfg=make_config(seed,dr0,n_diam=16,pupil_pad_factor=8.,n_frames=1)
            ranking.append({'seed':seed,'dr0':dr0,'checks':[asdict(c) for c in check_lowfreq_ranking(cfg)]})
    manifest=experiment_manifest('r10-bounded-development-audit',status='diagnostic',
        protocol='Physical-screen reduced-grid development ablations; never a Gate-1/Q2 qualification.',
        results=records,seed_coverage=protocol,
        notes='No evaluation seeds, Q3, production atmospheric claims or GPU execution.',
        extra={'wall_s':time.monotonic()-started,'lowfreq_ranking':ranking,'q3_authorized':False})
    write_manifest(directory/'manifest.json',_to_jsonable(manifest))
    return manifest


if __name__ == '__main__':
    import argparse
    from planetrecon.runtime import apply_thread_limits
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    apply_thread_limits(2)
    run(args.out)

"""Fixed fractional-translation sweep of matching sampler and peak refinement."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.local_align import LocalRegistration
from tools.analytic_sampling_probe import components, detector_image
from tools.fractional_ap_trace import trace, observations
from tools.regularized_ap_fit import RegularizedAPFit
from tools.ap_noise_coupling import hashes as prior_hashes
from tools.joint_saturn_experiment import digest
from tools.reference_imprint_probe import probes

POLICIES = {'baseline': (False, False), 'cubic': (True, False),
            'refined': (False, True), 'cubic_refined': (True, True)}


def hashes():
    return dict(prior_hashes(), **{p: digest(p) for p in ('tools/fractional_ap_trace.py', 'tools/ap_fractional_phase.py')})


def evaluate(model, proxy, shift, truth, mask, policy):
    cubic, refine = POLICIES[policy]
    stages, records = trace(model.engine, proxy, shift, cubic=cubic, refine=refine)
    obs = observations(model.engine, stages, records, shift)
    field, stats = model.field('baseline', obs, shift)
    delta = field.cpu().numpy()-truth
    return field, dict(field_vector_rmse_px=float(np.sqrt(np.mean(np.sum(delta[:,mask]**2, axis=0)))),
        accepted_aps=int(np.count_nonzero(obs[1])), fit=stats,
        stage_field_rmse_px=[float((v-torch.as_tensor(truth, device=v.device))[:,mask].square().sum(0).mean().sqrt()) for v in stages],
        stage_measurement_rmse_px=[float(np.sqrt(np.mean(np.sum(r['measurements'][:,:2]**2, axis=1)))) for r in records],
        stage_accepted=[r['accepted'] for r in records])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    shape = (128,160)
    features = components('resolved', shape)
    reference = detector_image(features, shape, blur=1.)
    _, mask = probes(reference)
    model = RegularizedAPFit(reference, policies=['baseline'])
    shifts = [(x,y) for y in (-.5,-.25,0.,.25,.5) for x in (-.5,-.25,0.,.25,.5)]+[(1.,0.),(0.,1.),(-1.,-1.)]
    report = dict(hashes=hashes(), policies=POLICIES, shifts=shifts, rows=[])
    for shift in shifts:
        frame = detector_image(features, shape, shift, blur=1.)
        proxy = LocalRegistration.proxy(frame)
        truth = np.broadcast_to(np.array(shift)[:,None,None], (2,*shape))
        row = dict(shift=shift, variants={})
        for policy in POLICIES:
            field, row['variants'][policy] = evaluate(model, proxy, shift, truth, mask, policy)
            if policy == 'baseline':
                old, _ = model.field('baseline', model.observations(proxy, shift), shift)
                torch.testing.assert_close(field, old, atol=1e-10, rtol=0)
        report['rows'].append(row)
        print(shift, {k: round(v['field_vector_rmse_px'], 6) for k,v in row['variants'].items()}, flush=True)
    report['summary'] = {k: dict(field_rmse_px=float(np.sqrt(np.mean([r['variants'][k]['field_vector_rmse_px']**2 for r in report['rows'][:25]]))),
        guard_rejections=sum(not r['variants'][k]['fit']['field_guard_accepted'] for r in report['rows']),
        insufficient_points=sum(r['variants'][k]['fit']['fallback'] for r in report['rows'])) for k in POLICIES}
    # Select only as a follow-up diagnostic, never directly for deployment.
    candidates = [k for k in POLICIES if k != 'baseline' and not report['summary'][k]['guard_rejections']
                  and not report['summary'][k]['insufficient_points']]
    report['selected'] = min(candidates, key=lambda k: report['summary'][k]['field_rmse_px']) if candidates else None
    if report['hashes'] != hashes():
        raise ValueError('Sources changed')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

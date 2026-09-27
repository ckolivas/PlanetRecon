"""Validate the selected matching policy on true residual and deforming motion."""
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.local_align import LocalRegistration
from tools.analytic_sampling_probe import components, detector_image
from tools.ap_stability_validation import integrated
from tools.ap_oracle_observations import CASES
from tools.ap_fractional_phase import hashes, evaluate
from tools.regularized_ap_fit import RegularizedAPFit
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import image_errors, save_png, weighted_field_errors


def main():
    root, prior = Path('out/ap-fractional-validation'), Path('out/ap-noise-coupling')
    phase_path = Path('out/ap-fractional-phase/report.json')
    phase = json.loads(phase_path.read_text())
    if phase['hashes'] != hashes() or phase['selected'] is None:
        raise ValueError('No unchanged selected policy')
    prior_report = json.loads((prior/'report.json').read_text())
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    with np.load('out/ap-oracle-observations/geometry.npz') as data:
        reference, mask = data['reference'], data['mask']
    shape, count = reference.shape, 32
    features = components('resolved', shape)
    model = RegularizedAPFit(reference, policies=['baseline'])
    engine = model.engine
    selected = phase['selected']
    report = dict(hashes=hashes(), source_sha256=digest(__file__), selected=selected,
        phase_report_sha256=digest(phase_path), prior_report_sha256=digest(prior/'report.json'), count=count, cases={})
    # Nonzero residual checks guard against a policy that only suppresses shifts.
    residual_rows = []
    for shift in ((.3,-.4), (-.45,.2), (.7,.65)):
        frame = detector_image(features, shape, shift, blur=1.)
        truth = np.broadcast_to(np.array(shift)[:,None,None], (2,*shape)).copy()
        row = dict(true_shift=shift, supplied_shift=[0.,0.], variants={})
        for policy in ('baseline', selected):
            _, row['variants'][policy] = evaluate(model, LocalRegistration.proxy(frame), (0.,0.), truth, mask, policy)
        residual_rows.append(row)
    report['nonzero_residual'] = residual_rows
    for case, wavelengths in CASES.items():
        path = prior/f'{case}.npz'
        if digest(path) != prior_report['cases'][case]['raw_sha256']:
            raise ValueError('Prior case changed')
        with np.load(path) as data:
            truths = data['truth']
            prior_fields = {0: data['field_clean'], 2: data['field_a']}
            prior_images = {0: data['clean_clean'], 2: data['noisy_match']}
            prior_twins = {0: data['clean_clean'], 2: data['clean_a']}
            target = data['target']
        rng = np.random.default_rng(9033)
        keys = [f'{p}_noise{n}' for p in ('baseline', selected) for n in (0,2)]
        images, twins, fields, statistics = ({k: [] if name in ('fields','statistics') else np.zeros(shape) for k in keys}
                                           for name in ('images','twins','fields','statistics'))
        for index, ph in enumerate(2*np.pi*np.arange(count)/count):
            a,b = (0.,0.) if case=='stationary' else (1.5*np.sin(ph), 1.2*np.cos(ph))
            if case=='short_offset':
                a,b = a+.8,b-.5
            translation = (.35*np.sin(ph+.4), .4*np.cos(ph))
            clean = integrated(features, shape, a,b,translation,wavelengths)
            noisy = clean+rng.normal(0,2.,shape)
            shift = np.asarray(translation) if case=='short_offset' else truths[index][:,mask].mean(1)
            for noise, frame in ((0,clean),(2,noisy)):
                for policy in ('baseline', selected):
                    key = f'{policy}_noise{noise}'
                    if policy=='baseline':
                        field = torch.as_tensor(prior_fields[noise][index], device=engine.device)
                        stats = {}
                    else:
                        field, stats = evaluate(model, LocalRegistration.proxy(frame), shift, truths[index], mask, policy)
                    fields[key].append(field.cpu().numpy())
                    statistics[key].append(stats)
                    for src, accum in ((frame,images),(clean,twins)):
                        tensor = torch.as_tensor(src, device=engine.device)
                        accum[key] += sample(tensor,engine.yy+field[1],engine.xx+field[0]).cpu().numpy()/count
        values = {}
        for noise in (0,2):
            np.testing.assert_allclose(images[f'baseline_noise{noise}'], prior_images[noise], atol=1e-9, rtol=0)
            np.testing.assert_allclose(twins[f'baseline_noise{noise}'], prior_twins[noise], atol=1e-9, rtol=0)
        for key in keys:
            values[key] = dict(image=image_errors(images[key],target,mask), clean=image_errors(twins[key],target,mask),
                field=weighted_field_errors(np.asarray(fields[key]),truths,np.ones(count)/count,mask), statistics=statistics[key])
            save_png(root/f'{case}_{key}.png',images[key])
        save_png(root/f'{case}_target.png',target)
        artifact = root/f'{case}.npz'
        np.savez_compressed(artifact, **images, **{'clean_'+k:v for k,v in twins.items()},
                            **{'field_'+k:np.asarray(v) for k,v in fields.items()}, mask=mask, truth=truths, target=target)
        report['cases'][case] = dict(raw_sha256=digest(artifact), variants=values)
        print(case,{k:dict(field=v['field']['total_vector_rmse_px'],clean=v['clean']['rmse_adu']) for k,v in values.items()},flush=True)
    if report['hashes'] != hashes() or report['source_sha256'] != digest(__file__):
        raise ValueError('Sources changed')
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()

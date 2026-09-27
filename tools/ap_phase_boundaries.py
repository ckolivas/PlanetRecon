"""Use known exterior scene values to isolate finite-image proxy boundaries."""
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.pipeline.local_align import LocalRegistration
from tools.analytic_sampling_probe import components, detector_image
from tools.ap_fractional_phase import evaluate, hashes
from tools.regularized_ap_fit import RegularizedAPFit
from tools.reference_imprint_probe import probes
from tools.joint_saturn_experiment import digest


def install_proxy(engine, proxy):
    engine.reference = torch.as_tensor(proxy, device=engine.device)
    for layer in engine.layers:
        patch = engine.reference[layer['y'][:,None,None]+layer['py'], layer['x'][:,None,None]+layer['px']]
        patch = patch-(patch*layer['weight']).sum((-2,-1))[:,None,None]
        layer['templates'] = patch
        layer['strength'] = (patch.square()*layer['weight']).sum((-2,-1)).sqrt()


def main():
    torch.set_num_threads(4)
    shape, pad = (128,160), 48
    features = components('resolved',shape)
    reference = detector_image(features,shape,blur=1.)
    _,mask = probes(reference)
    extended = features.copy()
    extended[:,1:3] += pad
    def proxy(shift, padded):
        if not padded:
            return LocalRegistration.proxy(detector_image(features,shape,shift,blur=1.))
        big = detector_image(extended,tuple(n+2*pad for n in shape),shift,blur=1.)
        return LocalRegistration.proxy(big)[pad:-pad,pad:-pad].copy()
    report = dict(hashes=hashes(), source_sha256=digest(__file__), pad=pad, rows=[])
    for padded in (False,True):
        model = RegularizedAPFit(reference,device='cpu',policies=['baseline'])
        if padded:
            install_proxy(model.engine,proxy((0.,0.),True))
        for shift in ((0.,0.),(1.,0.),(0.,1.),(-1.,-1.),(.25,.25),(.5,.5)):
            row = dict(padded=padded,shift=shift,variants={})
            truth = np.broadcast_to(np.array(shift)[:,None,None],(2,*shape)).copy()
            for policy in ('baseline','cubic_refined'):
                _,row['variants'][policy] = evaluate(model,proxy(shift,padded),shift,truth,mask,policy)
            report['rows'].append(row)
            print(padded,shift,{k:v['field_vector_rmse_px'] for k,v in row['variants'].items()},flush=True)
    report['limits'] = [
        'Known exterior scene values are available only in this simulation; this is not a proposed padding fix for real captures.',
        'Reference matching templates change to the extended-scene proxy while AP geometry, fitting kernels and final field fit stay fixed.',
        'The original synthetic scene has nonzero signal at image boundaries; these results do not measure boundary bias in Saturn.']
    Path('out/ap-fractional-phase/boundaries.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()

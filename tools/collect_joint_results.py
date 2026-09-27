"""Collect the controlled joint blur/motion experiment and convergence limits."""
import argparse
import hashlib
import json
from pathlib import Path


def source(path):
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    result=dict(model=source('tools/joint_registration.py'),production_changed=False,
                brightness_normalization=False,output_filtering=False,cases=[],real_capture_replay=False,
                decision='Resolve interpolation-model false motion before a full captured-data replay or production integration.')
    identities=set();recipe=None;config=None
    for name in ('joint-fit-saturn-controls','joint-fit-texture-controls','joint-fit-saturn-repeat','joint-fit-texture-repeat'):
        path=Path('out')/name/'analysis.json'
        report=json.loads(path.read_text())
        if report['model_sha256']!=result['model']['sha256']:raise ValueError('Model differs')
        current={k:report['sharpening'][k] for k in ('wavelet','deconvolution','implementation_sha256')}
        if recipe is None:recipe=current;config=report['config']
        elif recipe!=current or config!=report['config']:raise ValueError('Recipe or configuration differs')
        for item in report['sharpening']['images'].values():
            for key in ('source','output'):
                if source(item[key])['sha256']!=item[key+'_sha256']:raise ValueError('Sharpening artifact changed')
        for scene,cases in report['scenes'].items():
            for name,case in cases.items():
                identity=(scene,name,case['seed'])
                if identity in identities:raise ValueError('Duplicate control')
                identities.add(identity)
                values={k:v for k,v in case.items() if k!='statistics'}
                values.update(scene=scene,condition=name,source=source(path),
                              baseline_max_difference_adu=report['maximum_coherent_baseline_difference_adu'])
                result['cases'].append(values)
    expected={(scene,case,seed) for scene in ('saturn','texture')
              for case in ('static_noise','static_blur_noise','motion_blur_noise','offset_motion_blur')
              for seed in (9017,9018)}
    if identities!=expected:raise ValueError('Incomplete controls')
    result.update(config=config,sharpening=recipe,synthetic_frames=sum(c['frames'] for c in result['cases']))
    path=Path('out/joint-convergence-v2.json')
    result['convergence_source']=source(path)
    result['convergence']=json.loads(path.read_text())
    if result['convergence']['model_sha256']!=result['model']['sha256']:raise ValueError('Probe model differs')
    if {(f['scene'],f['case']) for f in result['convergence']['frames']} != {
        (s,c) for s in ('saturn','texture') for c in ('stationary','motion','offset')}:
        raise ValueError('Incomplete convergence check')
    if any([r['budget'] for r in f['results']]!=[100,300,800] for f in result['convergence']['frames']):
        raise ValueError('Incomplete convergence budgets')
    result['translation_probes']={}
    for name,generator,noises in [('joint-translation-probe','scipy_spline',(0.,2.)),
                                   ('joint-translation-model-probe','model',(0.,))]:
        path=Path('out')/(name+'.json')
        probe=json.loads(path.read_text())
        if probe['model_sha256']!=result['model']['sha256']:raise ValueError('Translation model differs')
        expected={(s,shift,b,n) for s in ('saturn','texture')
                  for shift in ((.25,.5),(.6,-.4),(-.7,.35)) for b in (0.,1.) for n in noises}
        actual={(f['scene'],tuple(f['shift_xy']),f['blur'],f['noise']) for f in probe['frames']}
        if actual!=expected or len(probe['frames'])!=len(expected):raise ValueError('Incomplete translation probe')
        if probe.get('generator','scipy_spline')!=generator:raise ValueError('Wrong translation generator')
        result['translation_probes'][name]=dict(source=source(path),results=probe)
    result['limits']=[
        'The 100-iteration estimator is not generally converged; a six-frame budget probe is only a sensitivity check.',
        'Only one quarter of detector samples train the field; remaining samples validate predictions.',
        'The smoothness penalty expresses a motion prior, not calibrated posterior uncertainty.',
        'Independent validation accepts false motion caused by a shared interpolation-model mismatch.',
        'Isotropic additional Gaussian blur is a limited seeing model; the synthetic family partly matches it by construction.',
        'No output smoothing or normalization is added; sharper geometry need not imply lower residual noise.']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(result['synthetic_frames'],'control frames; six convergence probes; identical baseline stacks and sharpening verified')


if __name__=='__main__':main()

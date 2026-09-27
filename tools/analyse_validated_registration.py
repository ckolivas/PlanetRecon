"""Collect matched-control evidence for independent-pixel motion validation."""
import argparse
import hashlib
import json
from pathlib import Path


RUNS = [
    ('primary', 'validated-saturn-static'),
    ('primary', 'validated-saturn-motion'),
    ('primary', 'validated-saturn-other'),
    ('primary', 'validated-texture-primary'),
    ('short', 'validated-short'),
    ('repeat', 'validated-repeat'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_report(root):
    path = root/'analysis.json'
    if not path.exists():
        path = root/'report.json'
    return json.loads(path.read_text()), {'path': str(path), 'sha256': digest(path)}


def metrics(case):
    result = dict(case['variants']['local'])
    result.pop('image_vs_mean_blurred_scene')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--real', type=Path, default=Path('out/saturn-validated-field/analysis.json'))
    args = parser.parse_args()
    evidence = {'production_changed': False, 'brightness_normalization': False,
                'raw_output_filtering': False, 'synthetic_frames': 0, 'cases': [],
                'source_code_sha256': {p: digest(p) for p in [
                    'tools/validated_registration.py', 'tools/coherent_registration.py',
                    'tools/validate_local_warp_centring.py', 'tools/coherent_saturn_experiment.py']}}
    recipe = None
    for group, name in RUNS:
        root = Path('out')/name
        report, source = read_report(root)
        if not report['independent_pixel_validation'] or not report['patch_average']:
            raise ValueError('Unexpected registration configuration')
        sharp = report['sharpening']
        current_recipe = {k: sharp[k] for k in ('wavelet', 'deconvolution', 'implementation_sha256')}
        if recipe is None:
            recipe = current_recipe
        elif recipe != current_recipe:
            raise ValueError('Sharpening recipe changed between runs')
        for scene, cases in report['scenes'].items():
            if group == 'repeat':
                coherent_root = Path('out/coherent-repeat')
                production_root = Path('out/local-warp-validation-repeat')
            elif group == 'short':
                coherent_root = Path('out/coherent-area-short-0.01')
                production_root = Path('out/coherent-short-baseline')
            else:
                coherent_root = Path('out')/('coherent-validation' if scene == 'saturn'
                                             else 'coherent-validation-texture')
                production_root = Path('out/local-warp-validation')
            coherent, coherent_source = read_report(coherent_root)
            production, production_source = read_report(production_root)
            if {k: coherent['sharpening'][k] for k in current_recipe} != recipe:
                raise ValueError('Baseline sharpening recipe differs')
            for case_name, case in cases.items():
                old = coherent['scenes'][scene][case_name]
                current = production['scenes'][scene][case_name]
                for other in (old, current):
                    for key in ('seed', 'frame_count', 'a_px', 'b_px', 'blur_sigma_px',
                                'weights', 'oracle_global_shifts_xy'):
                        if other[key] != case[key]:
                            raise ValueError(f'Unmatched control: {key}')
                oracle = f'{scene}_{case_name}_noiseless_oracle.png'
                oracle_hash = digest(root/oracle)
                if any(digest(p/oracle) != oracle_hash for p in (coherent_root, production_root)):
                    raise ValueError('Oracle image differs between runs')
                evidence['synthetic_frames'] += case['frame_count']
                evidence['cases'].append({
                    'group': group, 'scene': scene, 'condition': case_name,
                    'seed': case['seed'], 'frame_count': case['frame_count'],
                    'motion_wavelengths_px': case['motion_wavelengths_px'],
                    'source': source, 'coherent_source': coherent_source,
                    'production_source': production_source, 'oracle_sha256': oracle_hash,
                    'production': metrics(current), 'coherent': metrics(old), 'validated': metrics(case),
                    'accepted_half_counts': {str(i): sum(s['accepted_halves'] == i
                                                       for s in case['coherent_fit']) for i in range(3)}})
    if len(evidence['cases']) != 26:
        raise ValueError('Incomplete synthetic controls')
    evidence['sharpening'] = recipe
    phase_path = Path('out/validated-detector-phase-replay.json')
    phase = json.loads(phase_path.read_text())
    if phase['model_sha256'] != evidence['source_code_sha256']['tools/validated_registration.py']:
        raise ValueError('Detector-phase check used a different model')
    evidence['detector_phase_source'] = {'path': str(phase_path), 'sha256': digest(phase_path)}
    evidence['detector_phase'] = phase
    if args.real.exists():
        evidence['real_source'] = {'path': str(args.real), 'sha256': digest(args.real)}
        evidence['real'] = json.loads(args.real.read_text())
        real = evidence['real']
        if real['hashes']['validation_code'] != phase['model_sha256'] or real['n_used'] != 5738:
            raise ValueError('Unexpected real replay model or frame count')
        if {k: real['sharpening'][k] for k in recipe} != recipe:
            raise ValueError('Real replay sharpening differs from synthetic controls')
    evidence['limits'] = [
        'Synthetic shears, blur and independent Gaussian noise are controls, not a calibrated atmosphere or sensor model.',
        'Checkerboard halves have disjoint detector samples; independence does not cover spatially correlated detector noise.',
        'The forward model uses an approximate reference blur bank with sigma 0 to 2 pixels and a common contrast nuisance fit.',
        'The fixed real reference may share frames with the stack; this experiment does not create an independent reference.',
        'Static clean-twin error isolates geometric damage. Sharpened error also includes noise and nonlinear processing.',
        'The real capture has no known geometric truth; variation and edge width alone cannot establish accuracy.',
        'Only separately labelled AS-response comparison copies are filtered. The raw candidate stack is unfiltered.',
        'Parameters were fixed before these evaluations; this remains one real capture and a limited synthetic family.']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2)+'\n')
    print(f'{len(evidence["cases"])} controls, {evidence["synthetic_frames"]} frames; real replay present: {"real" in evidence}')


if __name__ == '__main__':
    main()

"""Measure the fixed-translation native drizzle control with exact sharpening."""
import json
from pathlib import Path

from tools.alignment_noise_experiment import read_png, measurements
from tools.analyse_refined_registration import ring_edges
from tools.analyse_joint_saturn import sharp_identity
from tools.joint_saturn_experiment import digest


def main():
    root = Path('out/saturn-native-drizzle')
    report = json.loads((root/'report.json').read_text())
    prior = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    report['sharpening'] = sharp_identity(root, prior)
    for name, values in report['variants'].items():
        path = root/'sharpened'/f'{name}_sharpened.png'
        pixels = read_png(path)[0]
        values.update(path=str(path), sha256=digest(path), variation=measurements(pixels), ring_edges=ring_edges(pixels))
        print(name, [v['highpass_percent'] for v in values['variation'].values()],
              [v['width_10_90_px'] for v in values['ring_edges'].values()])
    report['limits'] = [
        'Fixed global translations only; no local deformation, AP recombination or resolution enlargement.',
        'Area-overlap square footprints at native scale are explicit test kernels, not claims about AS internals.',
        'A unit square footprint equals bilinear sampling for translations; point placement equals rounded translations.',
        'At exact integer placement these kernels preserve an impulse and cannot generate the previously measured broad AS response.',
        'Fine-scale variation includes detail and artifacts; descriptive ring widths are not calibrated resolution.']
    report['developer_source'] = dict(date='2024-01-18', author='Emil Kraaikamp (MvZ)', post=39,
        url='https://www.cloudynights.com/forums/topic/907378-jupiter-at-20-degrees-11624-not-what-you-think-dont-miss-panel-26/page/2/',
        description='Distinguishes integer regular stacking, subpixel enlarged drizzle and temporal Bayer channel reconstruction; not a current-version implementation audit.')
    (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    Path('results/registration/native-mono-drizzle.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

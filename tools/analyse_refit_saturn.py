"""Measure a matched-frame refit ablation without comparing unequal frame counts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QFont
from PySide6.QtCore import Qt

from tools.alignment_noise_experiment import read_png, measurements
from tools.analyse_refined_registration import ring_edges, edge_sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    root = args.out
    report = json.loads((root/'report.json').read_text())
    policies = list(report['config']['policies'])
    paths = {name:root/'sharpened'/f'{name}_sharpened.png' for name in policies}
    if report['n_used']==5738:
        previous = json.loads(Path('out/saturn-validated-field/report.json').read_text())
        for key in ('reference','indices','quality','global_shifts','model_code'):
            if previous['hashes'][key] != report['hashes'][key]:
                raise ValueError('Previous full-count comparison inputs differ')
        policies.append('previous_validated')
        paths['previous_validated'] = Path('out/saturn-validated-field/sharpened/coherent_sharpened.png')
    sharp = json.loads((root/'sharpened/sharpening_report.json').read_text())
    prior = json.loads(Path('out/saturn-validated-field/sharpened/sharpening_report.json').read_text())
    for key in ('wavelet','deconvolution','implementation_sha256'):
        if sharp[key] != prior[key]:
            raise ValueError('Sharpening recipe differs from the established comparison')
    report['sharpening'] = sharp
    arrays = {}
    report['comparison'] = {}
    for name in policies:
        path = paths[name]
        a = read_png(path)[0]
        arrays[name] = a
        values = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      variation=measurements(a),ring_edges=ring_edges(a))
        report['comparison'][name] = values
        print(name,'variation',*[round(v['highpass_percent'],5) for v in values['variation'].values()],
              'ring widths',*[round(v['width_10_90_px'],5) for v in values['ring_edges'].values()],flush=True)
    centre = np.rint(center_of_mass(np.maximum(arrays['coherent']-.02*arrays['coherent'].max(),0))).astype(int)
    report['paired_edge_widths_vs_coherent'] = {}
    for name in policies[1:]:
        # The existing three-image helper includes an AS slot. Supply the same
        # baseline there and retain only the relevant candidate/baseline pair.
        images = dict(production_local=arrays[name],existing_bilinear=arrays['coherent'],AS_manual64=arrays['coherent'])
        paired = edge_sensitivity(images,{key:centre for key in images})
        report['paired_edge_widths_vs_coherent'][name] = {side:{
            'paired_windows':value['paired_windows'],
            'candidate_minus_coherent_width_px_min_median_max':value['local_minus_global_width_px_min_median_max'],
            'fraction_candidate_wider':value['fraction_local_wider']} for side,value in paired.items()}
    report['limits'] = [
        'All methods use identical selected frames, weights, global shifts and the same reference.',
        'A subset result cannot be directly compared to the 5738-frame AS or PR results.',
        'No normalization or added output filter is used. Every method uses the exact user sharpening recipe.',
        'Full-data refitting follows independent motion gating; the final refitted field is not itself held out.',
        'Fine-scale variation contains detail and artifacts as well as noise; widths are not calibrated resolution.',
        'Overlapping edge windows are a sensitivity check, not independent replicates.']
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    app = QGuiApplication.instance() or QGuiApplication([])
    canvas = QImage(2000,1530,QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setPen(QColor('white'));painter.setFont(QFont('Sans',18))
    labels = dict(coherent='Full-frame coherent fit',half_average='Half-frame fits averaged, no extra gate',
              validated='Previous independent-pixel gated average',full_any='Motion gate, then full-frame refit',
              full_scaled='Full-frame fit scaled by accepted-half count',
              previous_validated='Previous independent-pixel gated average',
              regional_any='Regional gate: either half accepts',regional_both='Regional gate: both halves accept')
    cy,cx = centre
    for i,name in enumerate(policies):
        label = labels[name]
        image = QImage(str(paths[name]))
        crop = image.copy(int(cx-250),int(cy-115),500,230)
        crop = crop.scaled(1000,460,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.FastTransformation)
        x,y = (i%2)*1000,(i//2)*510
        painter.drawText(x+12,y+32,label);painter.drawImage(x,y+45,crop)
    painter.drawText(1012,1100,f'Identical {report["n_used"]} frames and weights')
    painter.drawText(1012,1140,'Wavelet 27/0/0/0; adaptive deconvolution 15.6')
    painter.drawText(1012,1180,'No added output filtering or normalization')
    painter.end()
    if not canvas.save(str(root/'comparison.png')):
        raise OSError('comparison.png')


if __name__ == '__main__':
    main()

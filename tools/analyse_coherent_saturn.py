"""Compare the real coherent-field replay with fixed existing stack controls."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('out/saturn-coherent-field'))
    args=parser.parse_args(); out=args.out
    report=json.loads((out/'report.json').read_text())
    paths={
        'production_raw':Path('2024-09-27-1154_3-CK-R-SatPs.png'),
        'coherent_raw':out/'sharpened/coherent_sharpened.png',
        'production_matched':Path('out/saturn-matched-transfer/sharpened/pr_original_as_transfer_sharpened.png'),
        'coherent_matched':out/'sharpened/coherent_as_transfer_sharpened.png',
        'global_matched':Path('out/saturn-refined-registration/sharpened/existing_bilinear_as_transfer_sharpened.png'),
        'AS':Path('out/saturn-matched-transfer/sharpened/as_manual64_sharpened.png')}
    arrays={name:read_png(path)[0] for name,path in paths.items()}
    centres={name:np.asarray(center_of_mass(np.maximum(a-a.max()*.02,0))) for name,a in arrays.items()}
    report['comparison']={}
    for name,a in arrays.items():
        report['comparison'][name]={'path':str(paths[name]),'sha256':hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            'centroid_yx':centres[name].tolist(),'variation':measurements(a),'ring_edges':ring_edges(a)}
        print(name,'variation',*[round(v['highpass_percent'],5) for v in report['comparison'][name]['variation'].values()],
              'ring widths',*[round(v['width_10_90_px'],5) for v in report['comparison'][name]['ring_edges'].values()],flush=True)
    report['edge_sensitivity']={}
    for name in ('production_matched','coherent_matched'):
        images={'production_local':arrays[name],'existing_bilinear':arrays['global_matched'],'AS_manual64':arrays['AS']}
        common={'production_local':np.rint(centres['global_matched']).astype(int),
                'existing_bilinear':np.rint(centres['global_matched']).astype(int),
                'AS_manual64':np.rint(centres['AS']).astype(int)}
        report['edge_sensitivity'][name]={'each_image_centroid':edge_sensitivity(images),
                                        'common_PR_centroid':edge_sensitivity(images,common)}
    report['sharpening']=json.loads((out/'sharpened/sharpening_report.json').read_text())
    report['limits']=[
        'Same 5738 frames, scalar weights, reference and global shifts; no per-frame or final-stack normalization.',
        'Raw coherent stack has no additional pixel filter. Only named matched comparison copies get AS-measured sigma-1 radius-3 smoothing.',
        'All sharpened outputs use Wavelet 27/0/0/0 then Adaptive Deconvolution 15.6 with Contrast Adaptive.',
        'Rejected fitted fields fall back to global translation; their frames still contribute to the stack.',
        'Variation includes real detail and artifacts; ring widths include geometry and processing, not calibrated resolution.',
        'The 75 overlapping windows are a sensitivity check, not independent statistical replicates.',
        'Both individual-image and common PR centroids are reported to expose window-recentring effects.',
        'This is one real capture; AS is a comparison output, not known geometric truth.']
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    labels=['Current PR, unfiltered','Experimental coherent field, unfiltered',
            'Current PR + measured AS response','Coherent field + measured AS response',
            'Global alignment + measured AS response','Supplied AutoStakkert, manual 64-frame reference']
    app=QGuiApplication.instance() or QGuiApplication([])
    canvas=QImage(2000,1530,QImage.Format.Format_RGB32);canvas.fill(QColor('black'))
    painter=QPainter(canvas);painter.setPen(QColor('white'));painter.setFont(QFont('Sans',18))
    for i,((name,path),label) in enumerate(zip(paths.items(),labels)):
        cy,cx=np.rint(centres[name]).astype(int)
        crop=QImage(str(path)).copy(int(cx-250),int(cy-115),500,230)
        crop=crop.scaled(1000,460,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.FastTransformation)
        x,y=(i%2)*1000,(i//2)*510
        painter.drawText(x+12,y+32,label);painter.drawImage(x,y+45,crop)
    painter.end()
    if not canvas.save(str(out/'comparison.png')):raise OSError('comparison.png')


if __name__=='__main__':main()

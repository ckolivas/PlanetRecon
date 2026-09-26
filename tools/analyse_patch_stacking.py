"""Summarize controlled patch stacks after the exact PlanetaryTools replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass

from tools.alignment_noise_experiment import measurements, read_png
from tools.patch_stacking_experiment import combine_patches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--as-sharpened', type=Path, required=True)
    parser.add_argument('--pr-sharpened', type=Path, required=True)
    args = parser.parse_args()
    from planetrecon.runtime import apply_thread_limits
    apply_thread_limits(8)
    report = json.loads((args.experiment/'report.json').read_text())
    report['date'] = '2026-09-27'
    report['sharpening'] = json.loads((args.experiment/'sharpened'/'sharpening_report.json').read_text())
    report['comparison'] = {}
    for name, path in [('AS_manual64', args.as_sharpened), ('PR_production', args.pr_sharpened)]:
        im, _ = read_png(path)
        report['comparison'][name] = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                          sharpened=measurements(im))
    for name in report['variants']:
        path = args.experiment/'sharpened'/f'{name}_sharpened.png'
        im, _ = read_png(path)
        report['variants'][name]['sharpened'] = measurements(im)
    with np.load(args.experiment/'measurements.npz') as data:
        scalar_weights = data['global_weights'].copy()
        original_selection = data['original_selected'].copy()
        shifts = data['local_shifts']-data['global_shifts'][:,None,:]
        report['local_residual_magnitude_percentiles'] = np.percentile(np.linalg.norm(shifts,axis=-1), [0,50,90,99,100]).tolist()
        report['screened_indices_sha256'] = hashlib.sha256(data['indices'].tobytes()).hexdigest()
    with np.load(args.experiment/'selections.npz') as data:
        selection_names, selection_masks = data['names'].copy(), data['masks'].copy()
        for name, mask in zip(data['names'], data['masks']):
            report['variants'][str(name)]['unique_frames_in_union'] = int(np.any(mask,axis=1).sum())
            report['variants'][str(name)]['selected_indices_per_ap_sha256'] = hashlib.sha256(mask.tobytes()).hexdigest()
    with np.load(args.experiment/'global_control.npz') as data:
        ref = data['image']
    with np.load(args.experiment/'patch_stacks.npz') as data:
        centres, footprint = data['centres'].copy(), data['footprint'].copy()
        _, coverage = combine_patches(np.ones_like(data['means'][0]),np.ones_like(data['means'][0]),
                                      data['centres'],data['footprint'],np.zeros_like(ref))
    cy,cx = np.round(center_of_mass(np.maximum(ref-ref.max()*.02,0))).astype(int)
    report['metric_roi_ap_coverage_fraction'] = {
        'upper':float(np.mean(coverage[cy-55:cy-30,cx-40:cx+40]>1e-12)),
        'lower':float(np.mean(coverage[cy+35:cy+60,cx-40:cx+40]>1e-12))}
    # Effective count from frame weights after AP overlap, ignoring the
    # additional covariance changes caused by different subpixel resamplings.
    for name, mask in zip(selection_names, selection_masks):
        normalized = np.column_stack([mask,original_selection])*scalar_weights[:,None]
        normalized /= normalized.sum(axis=0)
        covariance = normalized.T@normalized
        effective = {}
        for region, ylo in [('upper', cy-55), ('lower', cy+35)]:
            yy,xx = np.indices((25,80)); yy += ylo; xx += cx-40
            dy,dx = yy[None]-centres[:,0,None,None]+32, xx[None]-centres[:,1,None,None]+32
            good = (dy>=0)&(dy<65)&(dx>=0)&(dx<65)
            blend = (footprint[dy.clip(0,64),dx.clip(0,64)]*good).reshape(len(centres),-1)
            coverage = blend.sum(axis=0)
            blend = np.vstack([blend,np.maximum(1.-coverage,0.)])/np.maximum(coverage,1.)
            variance = np.sum(blend*(covariance@blend),axis=0)
            effective[region] = np.percentile(1/variance,[10,50,90]).tolist()
        report['variants'][str(name)]['overlap_frame_weight_effective_count_p10_median_p90'] = effective
    report['limits'] = [
        'Controlled experimental mechanisms, not a reproduction of AutoStakkert internal algorithms.',
        'Same 5738 observations per AP; different APs may select different frame IDs.',
        'Pool is the 25476 PR-screened frames, not all 27689 capture frames.',
        'Single 65px scale with 48 overlapping circular APs and fixed production 64-frame reference.',
        'Original scalar global quality weights retained; no brightness normalization or output denoising.',
        'Smoothing is used only to rank quality and measure alignment; original raw pixels are accumulated.',
        'Registered variants have an extra bilinear resampling of each completed patch.',
        'Sequential variants resample global alignment and local alignment separately.',
        'AP overlap can increase effective frame count beyond 5738 by mixing different selections; reported frame-weight counts ignore resampling covariance.',
        'Outside patch support, products use the original global stack.',
        'Fine-scale variation includes real structure and artifacts; it is not a direct sensor-noise measurement.',
        'No new AutoStakkert Global-mode stack was generated by this experiment.']
    (args.experiment/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    for name, item in list(report['comparison'].items())+list(report['variants'].items()):
        m=item['sharpened']
        print(name,*(round(m[part]['highpass_percent'],4) for part in ('upper','lower')))


if __name__ == '__main__':
    main()

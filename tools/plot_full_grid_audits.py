"""Render the full-grid numerical and model evidence with system Matplotlib."""
import argparse
import json
import gzip
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run(directory):
    root = Path(__file__).resolve().parents[1]/'results'
    def read(path):
        file = root/path
        raw = file.read_bytes()
        return json.loads(gzip.decompress(raw) if file.suffix == '.gz' else raw)
    dev = read('r10-full-gate1-admm-development/report.json')
    corrected = read('r10-full-gate1-admm-development-v2/report.json')
    boundary = read('r10-full-crop-model-audit/report.json')
    noise = read('r10-full-noise-weight-sensitivity/report.json')
    noise_cases = [read('r10-full-noise-weight-sensitivity/'+c['path']) for c in noise['cases']]
    colors = {'feature': '#247b87', 'bland': '#b76d3f'}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    fig.suptitle('500-frame, 128×128 scientific controls — experimental solver', fontsize=16)
    ax = axes[0, 0]
    for i, (report, path) in enumerate(((dev, 'r10-full-gate1-admm-development'),
                                      (corrected, 'r10-full-gate1-admm-development-v2'))):
        cases = [read(path+'/'+c['path']) for c in report['cases']]
        for b, offset, color in [(0, -.17, '#8fabc4'), (1, .17, '#247b87')]:
            infos = [m['_info'][est] for c in cases for m in c['runs'][b]['result']['metrics'].values()
                     for est in ('E2a', 'A1o')]
            count = sum(info['converged'] for info in infos)
            ax.bar(i+offset, count, width=.3, color=color, label=('10k cap' if b == 0 else '20k cap') if i == 0 else None)
            ax.text(i+offset, count+2, f'{count}/{len(infos)}', ha='center', fontsize=8)
    ax.set_xticks([0, 1], ['Initial candidate', 'Real-Fourier correction'])
    ax.set(ylabel='Constrained solves passing declared checks', ylim=(0, 132), title='Complete development family; strict checks')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    regions = ['whole_crop', 'metric_window_squared', 'interior_border_8', 'interior_border_16', 'interior_border_32']
    for crop, color in colors.items():
        rows = [c for c in boundary['cases'] if c['crop'] == crop]
        values = np.array([[c['regions'][r]['model_mean_standardized_square'] for r in regions] for c in rows])
        ax.semilogy(range(5), np.median(values, axis=0), 'o-', color=color, label=crop)
        ax.fill_between(range(5), values.min(axis=0), values.max(axis=0), color=color, alpha=.12)
    ax.axhline(1., color='#777', ls='--', lw=1)
    ax.set_xticks(range(5), ['Full crop', 'Metric\nwindow²', 'Trim 8', 'Trim 16', 'Trim 32'])
    ax.set(ylabel='Model residual² / expected noise variance', title='Forward mismatch falls away from boundaries')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for gap, offset, color in [('G1', -.12, '#247b87'), ('G2', .12, '#b76d3f')]:
        for i, (report, regime) in enumerate(((dev, '4.0'), (dev, '8.0'), (corrected, '4.0'), (corrected, '8.0'))):
            table = report['tables']['20000/feature'][regime][gap]
            ax.scatter(np.full(len(table['per_seed_g']), i+offset), table['per_seed_g'], color=color, alpha=.5, s=15)
            ax.plot(i+offset, table['median'], '_', color=color, ms=17, mew=3, label=gap if i == 0 else None)
    ax.axhline(0., color='#777', lw=1)
    ax.set_xticks(range(4), ['Initial 4', 'Initial 8', 'Corrected 4', 'Corrected 8'])
    ax.set(ylabel='Relative gap (feature crop)', title='Descriptive gaps; incomplete numerical qualification', xlabel='Candidate / D/r0')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for crop, offset in [('feature', -.12), ('bland', .12)]:
        for i, (regime, p) in enumerate(((4., 10), (4., 100), (8., 10), (8., 100))):
            values = [s['weighted_fits'][-1]['metrics']['E_H']/s['scalar_metrics']['E_H']
                      for c in noise_cases if c['crop'] == crop and c['dr0'] == regime
                      for s in c['subsets'] if s['p'] == p]
            ax.scatter(np.full(len(values), i+offset), values, color=colors[crop], s=24, label=crop if i == 0 else None)
    ax.axhline(1., color='#777', ls='--', lw=1)
    ax.set_xticks(range(4), ['4 / S10', '4 / S100', '8 / S10', '8 / S100'])
    ax.set(ylabel='E_H with pixel weights / scalar weights', xlabel='D/r0 / subset', title='Noise weighting worsens this mismatched model')
    ax.legend(fontsize=8, title='Bland: ill-conditioned', title_fontsize=8)
    for ax in axes.ravel():
        ax.grid(axis='y', alpha=.18)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory/'full-grid.png', dpi=150)
    fig.savefig(directory/'full-grid.svg')
    svg = directory/'full-grid.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out)

"""Render the committed convergence studies with system Matplotlib."""
import argparse
import json
import gzip
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]/'results'
    def read(path):
        file = root/path
        return json.loads(gzip.decompress(file.read_bytes()) if file.suffix == '.gz' else file.read_bytes())
    physical = read('r10-full-grid-physics-audit/report.json')
    before = read('r10-constraint-operator-audit-v3/report.json')
    after = read('r10-constraint-operator-audit-v5/report.json')
    joint_paths = ['r10-joint-budget-audit', 'r10-joint-budget-audit-v2', 'r10-joint-budget-audit-v3']
    joint = [[read(f'{p}/{r["path"]}') for r in read(f'{p}/report.json')['cases']] for p in joint_paths]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    fig.suptitle('Development convergence evidence — bounded studies', fontsize=16)
    a = axes[0, 0]
    limits = {'padding_doubling': .005, 'grid_doubling': .02, 'exposure_quadrature_doubling': .005,
              'tilt_grid_doubling': .05, 'lowfreq_strehl_subharmonics': .02, 'lowfreq_tilt_subharmonics': .1}
    ratios = [max(c['value']/lim for r in physical['cases'] for c in r['checks'] if c['name'] == name)
              for name, lim in limits.items()]
    a.barh(['Padding', 'Pupil grid', 'Exposure quadrature', 'Grid tilt', 'Low-frequency Strehl', 'Low-frequency tilt'], ratios, color='#267a8c')
    for i, ratio in enumerate(ratios):
        a.text(ratio+.015, i, f'{ratio:.2g}', va='center', fontsize=8)
    a.axvline(1., color='#b23a48', linestyle='--')
    a.set(xlabel='Worst measured error / acceptance tolerance', title='60 full-grid physical controls')
    a = axes[0, 1]
    for report, label, color in [(before, 'Before restart', '#b23a48'), (after, 'Restart + tighter stopping', '#267a8c')]:
        vals = [max(s['relative_image_error'] for s in c['solves'] if s['budget'] == 512) for c in report['cases']]
        a.semilogy(range(6), vals, 'o-', color=color, label=label)
    a.axhline(1e-4, color='#555', linestyle='--', label='Oracle image tolerance')
    a.set_xticks(range(6), ['1001 B', '1001 F', '1002 B', '1002 F', '1003 B', '1003 F'])
    a.set(ylabel='Relative image error', title='Object error against independent optimizer', xlabel='B: band-limited; F: full support')
    a.legend(fontsize=8)
    a = axes[1, 0]
    for i, cases in enumerate(joint):
        valid = [sum(c['budgets'][b]['fit']['status'] == 'valid' for c in cases) for b in range(2)]
        x = i+np.array([-.17, .17])
        a.bar(x, valid, width=.3, color=['#8fabc4', '#267a8c'])
        for xx, vv in zip(x, valid):
            a.text(xx, vv+.15, str(vv), ha='center', fontsize=9)
    a.set_xticks(range(3), ['Old derivative\n4/32, 12/96', 'Corrected derivative\n4/32, 12/96', 'Diverse phase starts\n12/96, 36/192'])
    a.set(ylim=(0, 13), ylabel='Cases with valid selected fit and assessment', title='MFBD status (final refinement: 12/12)')
    a = axes[1, 1]
    cases = joint[-1]
    labels = [f"{c['seed']%1000}:{int(c['dr0'])}{c['crop'][0].upper()}" for c in cases]
    for b, label, color in [(0, '12 outer / 96 phase', '#8fabc4'), (1, '36 outer / 192 phase', '#267a8c')]:
        a.plot(range(12), [c['budgets'][b]['wall_s'] for c in cases], 'o-', label=label, color=color)
    a.set_xticks(range(12), labels, rotation=45)
    a.set(ylabel='Measured CPU wall time (seconds)', xlabel='Seed suffix : seeing regime / crop', title='Per-case fit cost; 8 frames, 32×32 crop')
    a.legend(fontsize=8)
    for a in axes.ravel():
        a.grid(axis='y', alpha=.15)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out/'convergence.png', dpi=150)
    fig.savefig(args.out/'convergence.svg')
    svg = args.out/'convergence.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__ == '__main__':
    main()

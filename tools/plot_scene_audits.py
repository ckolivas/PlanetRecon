"""Standalone scientific figures for archived scene-operator/solver controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(model_dir, solver_dir, output):
    model = json.loads((model_dir/'report.json').read_text())
    solver = json.loads((solver_dir/'report.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    labels, extended, legacy = [], [], []
    for row in model['cases']:
        for c in row['crops']:
            labels.append(f"{row['seed']}\n{int(row['dr0'])} · {c['crop'][0].upper()}")
            extended.append(c['models']['extended']['stack_max_standardized_square'])
            legacy.append(c['models']['legacy_circular']['stack_max_standardized_square'])
    x = np.arange(len(labels))
    axes[0].scatter(x, legacy, label='Legacy crop-periodic', color='#b45b3b', marker='x')
    axes[0].scatter(x, extended, label='Extended optical scene', color='#207f85')
    axes[0].axhline(1e-6, color='0.5', linestyle=':', label='Numerical tolerance')
    axes[0].set(yscale='log', ylabel='Worst pixel: stack bias² / stack noise variance',
                title='Full development sequences: 3,000 frames', xticks=x, xticklabels=labels)
    axes[0].tick_params(axis='x', labelsize=7)
    axes[0].legend(loc='center left', fontsize=8)
    names = []
    for i, row in enumerate(solver['cases']):
        names.append(row['case'].replace('/', '\n')+f"\n{row['runs'][0]['maxiter']}/{row['runs'][1]['maxiter']} caps")
        for k, info in enumerate(row['runs']):
            axes[1].scatter(i+(k-.5)*.16, info['relative_solution_error_bound'],
                            color=('#3976aa', '#cf843a')[k], marker='o' if info['converged'] else 'x',
                            label=("Smaller cap", "Larger cap")[k] if i == 0 else None)
    axes[1].axhline(1e-5, color='0.5', linestyle=':', label='Certificate tolerance')
    axes[1].set(yscale='linear', ylim=(0, 1.1e-5), xticks=range(len(names)), xticklabels=names,
                ylabel='Certified relative solution-distance upper bound',
                title='Fixed-prior, three-frame numerical pilot')
    axes[1].legend(fontsize=8)
    fig.suptitle('Extended optical model and constrained solver: scoped numerical checks', fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=170)
    fig.savefig(output.with_suffix('.svg'))
    svg = output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--solver', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    plot(a.model, a.solver, a.out)

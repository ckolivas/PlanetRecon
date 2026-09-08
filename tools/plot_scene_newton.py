"""Render numerical endpoint evidence and accepted Newton-CG progress."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.plot_scene_endpoints import endpoint_bounds


def plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    labels, bounds, colours = [], [], []
    for method, folder in [('Reference', 'p2-selection-endpoints-reference'),
                           ('L-BFGS-B', 'p2-selection-endpoints-lbfgsb'),
                           ('Newton-CG', 'p2-selection-endpoints-newton')]:
        for count, bound, passed in endpoint_bounds(results/folder):
            labels.append(f'{method}\n{count} frames'); bounds.append(bound)
            colours.append('#21735a' if passed else '#b67926')
    axes[0].bar(labels, bounds, color=colours)
    axes[0].set_yscale('log'); axes[0].set_ylim(min(min(bounds), 1e-5)/5, max(bounds)*5)
    for i, bound in enumerate(bounds): axes[0].text(i, bound*1.3, f'{bound:.3g}', ha='center', fontsize=8)
    axes[0].tick_params(axis='x', labelsize=8)
    axes[0].set_title('Independent CPU certificates\nWorst bound across the two declared caps')
    axes[0].set_ylabel('Relative scene-distance upper bound')
    root = results/'p2-selection-endpoints-newton'
    report = json.loads((root/'report.json').read_text())
    if not report['source_input_unchanged']: raise ValueError('changed source/input identity')
    for row in report['rows']:
        for fit in row['runs']:
            trace = fit.get('trace', [])
            if not trace: continue
            axes[1].semilogy([v['hessian_products'] for v in trace],
                             [v['relative_solution_error_bound'] for v in trace],
                             label=f"{row['n_used']} frames, cap {fit['max_products']}")
    axes[1].set_title('Accepted Newton-CG updates\nFinal certificates recomputed independently')
    axes[1].set_xlabel('Inner / line search / refresh Hessian products')
    axes[1].set_ylabel('Running relative distance bound')
    for axis in axes:
        axis.axhline(1e-5, color='#a43939', linestyle='--', label='Required: 1e-5')
        axis.grid(axis='y', alpha=.2); axis.legend(fontsize=8)
    fig.suptitle('Same objective; bounds are not measured image errors; cap units differ across solvers', fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('results'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); plot(a.results, a.output)

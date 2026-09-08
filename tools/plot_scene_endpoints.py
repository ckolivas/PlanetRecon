"""Render archived endpoint certificates and CPU verification costs."""
import argparse
import json
from pathlib import Path


def endpoint_bounds(root):
    report = json.loads((root/'report.json').read_text())
    values = []
    for row in report['rows']:
        bounds = []
        for fit in row['runs']:
            if 'reference_certificate' not in fit:
                records = sorted(root.glob(f"incomplete-{row['fraction']}-{fit['maxiter']}-*.json"))
                if not records: raise ValueError('missing incomplete fit certificate')
                fit = json.loads(records[-1].read_text())
            bounds.append(fit['reference_certificate']['relative_solution_error_bound'])
        values.append((row['n_used'], max(bounds), row['numerical_passed']))
    return values


def plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    labels, bounds, colours = [], [], []
    for method, folder in [('Reference', 'p2-selection-endpoints-reference'), ('L-BFGS-B', 'p2-selection-endpoints-lbfgsb')]:
        for count, bound, passed in endpoint_bounds(results/folder):
            labels.append(f'{method}\n{count} frames'); bounds.append(bound)
            colours.append('#21735a' if passed else '#b67926')
    axes[0].bar(labels, bounds, color=colours)
    axes[0].set_yscale('log')
    axes[0].set_ylim(1e-6, max(bounds)*5)
    axes[0].axhline(1e-5, color='#a43939', linestyle='--', label='Required: 1e-5')
    for i, bound in enumerate(bounds): axes[0].text(i, bound*1.3, f'{bound:.3g}', ha='center', fontsize=9)
    axes[0].set_ylabel('Relative scene-distance upper bound')
    axes[0].set_title('Independent CPU certificates\nWorse bound across both caps; not measured image error')
    axes[0].legend(fontsize=9)
    rows = json.loads((results/'p5-parallel-reference/report.json').read_text())['rows']
    for i, row in enumerate(rows):
        bars = axes[1].bar([i*3, i*3+1], [row['normal_s'], row['linear_s']], color=['#487b9c', '#728197'])
        axes[1].bar_label(bars, fmt='%.1f s', padding=4)
    axes[1].set_xticks([0, 1, 3, 4], ['Normal\n1 worker', 'Adjoint\n1 worker', 'Normal\n8 workers', 'Adjoint\n8 workers'])
    axes[1].set_ylim(0, max(r['normal_s'] for r in rows)*1.2)
    axes[1].set_ylabel('Elapsed operation time (seconds)')
    axes[1].set_title('Independent CPU operations, 500 frames\nSame ordered sums; workload-dependent timings')
    fig.suptitle('Full-count numerical evidence — scientific reconstruction quality remains unqualified', fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('results'))
    p.add_argument('--output', type=Path, default=Path('results/p2-selection-endpoints-comparison/overview.png'))
    a = p.parse_args(); plot(a.results, a.output)

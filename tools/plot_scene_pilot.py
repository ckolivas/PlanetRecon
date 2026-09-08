"""Render archived numerical pilot evidence without loading image or truth pixels."""
import argparse
import json
from pathlib import Path


def plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    performance = json.loads((results / 'p5-shared-fft/report.json').read_text())
    family = json.loads((results / 'p2-development-numerical-pilot/report.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    rows = performance['rows']
    labels = ['Reference CPU', 'Shared CPU\nno cache', 'Shared CPU\ncache', 'Shared CUDA\ncache']
    values = [r['solve']['wall_s'] for r in rows]
    bars = axes[0].bar(labels, values, color=['#728197', '#487b9c', '#487b9c', '#21735a'])
    axes[0].bar_label(bars, fmt='%.2f s', padding=4)
    axes[0].set_ylim(0, max(values) * 1.2)
    axes[0].set_ylabel('Iterative solve time (seconds)')
    axes[0].set_title('Three-frame performance pilot\nSame objective; setup / CPU check excluded')
    cases = family['cases']
    bounds = [max(r['reference_certificate']['relative_solution_error_bound'] for r in c['runs']) for c in cases]
    labels = [f"{c['seed']} / {c['dr0']:g}\n{c['crop']}" for c in cases]
    axes[1].bar(range(len(cases)), bounds, color='#21735a')
    axes[1].axhline(1e-5, color='#a43939', linestyle='--', label='Required bound: 1e-5')
    axes[1].set_xticks(range(len(cases)), labels, rotation=65, ha='right', fontsize=8)
    axes[1].set_ylim(0, 1.25e-5)
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[1].set_ylabel('Certified relative scene-distance bound')
    axes[1].set_title('12-case numerical pilot: 11 / 500 frames\nWorse bound across 750 / 1500 iteration caps')
    axes[1].legend(fontsize=8)
    fig.suptitle('Known-transfer numerical evidence — scientific image quality remains unqualified', fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('results/p2-development-numerical-pilot/overview.png'))
    args = parser.parse_args()
    plot(args.results, args.output)

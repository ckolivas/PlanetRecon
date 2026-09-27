"""Plot the audited AP-oracle record using system Python and Matplotlib."""
import json
from pathlib import Path

import numpy as np

CASES = ["stationary", "long_motion", "short_motion", "short_offset"]


def plot(root, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Measured APs', 'True APs, same gates', 'True APs, all available', 'True APs, no taper']
    names = ['measured', 'truth_gated', 'truth_all', 'truth_untapered']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x, width = np.arange(len(CASES)), .19
    for i, (name, label) in enumerate(zip(names, labels)):
        axes[0].bar(x+(i-1.5)*width,
            [report['cases'][c]['variants'][name]['field']['total_vector_rmse_px'] for c in CASES], width, label=label)
        axes[1].bar(x+(i-1.5)*width,
            [report['cases'][c]['variants'][name]['sharpened_versus_noisy_oracle']['rmse_adu'] for c in CASES], width, label=label)
    for ax in axes:
        ax.set_xticks(x, ['Stationary', 'Long motion', 'Short motion', 'Short + offset'], rotation=15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Motion error: vector RMS, pixels')
    axes[1].set_ylabel('Sharpened error vs same noisy frames aligned by truth\nRMSE in linear 0–1 image units')
    axes[0].legend(fontsize=8)
    fig.suptitle('Known-motion AP substitution: 32 identical frames per case\nWavelet 27/0/0/0 + Adaptive Deconvolution 15.6, Contrast Adaptive')
    fig.savefig(root/'comparison.png', dpi=160)
    plt.close(fig)



if __name__ == "__main__":
    root = Path("out/ap-oracle-observations")
    plot(root, json.loads((root/"analysis.json").read_text()))

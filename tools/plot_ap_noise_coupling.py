"""Plot audited matching and noise-coupling results with system Matplotlib."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('out/ap-noise-coupling')
    report = json.loads((root/'analysis.json').read_text())
    cases = ['stationary', 'long_motion', 'short_motion', 'short_offset']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(4)
    for i, (name, label) in enumerate([('clean', 'No added noise'), ('a', 'Noise A'), ('b', 'Noise B')]):
        axes[0].bar(x+(i-1)*.23, [report['cases'][c]['matching'][name]['field']['total_vector_rmse_px'] for c in cases], .23, label=label)
    for i, (name, label) in enumerate([('paired_same', 'Match own noise'), ('paired_cross', 'Match independent noise'),
            ('paired_no_noise_match', 'Match noiseless frame'), ('paired_oracle', 'Known true motion')]):
        axes[1].bar(x+(i-1.5)*.18, [report['cases'][c]['sharpened_error_linear_0_1'][name]['rmse_adu'] for c in cases], .18, label=label)
    for ax in axes:
        ax.set_xticks(x, ['Stationary', 'Long motion', 'Short motion', 'Short + offset'], rotation=15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1]*1.23)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Motion error: vector RMS, pixels')
    axes[1].set_ylabel('Sharpened RMSE vs noiseless target, linear 0–1')
    fig.suptitle('Zero-noise matching and paired noise exchange\n32 motion poses; paired outputs use the same 64 noisy frames')
    fig.savefig(root/'comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

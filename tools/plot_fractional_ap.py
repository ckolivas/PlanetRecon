"""Plot audited phase sweep and noisy-motion validation with system Matplotlib."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('out/ap-fractional-validation')
    report = json.loads((root/'analysis.json').read_text())
    phase = report['phase_screen']
    fig, axes = plt.subplots(1,3,figsize=(15,4.6), constrained_layout=True)
    policies = ['baseline','cubic','refined','cubic_refined']
    axes[0].bar(np.arange(4), [phase['summary'][k]['field_rmse_px'] for k in policies])
    axes[0].set_xticks(np.arange(4), ['Current','Cubic','Refined peak','Both'],rotation=20)
    axes[0].set_ylabel('False motion, vector RMS pixels')
    axes[0].set_title('Noiseless fractional translations')
    cases = ['stationary','long_motion','short_motion','short_offset']
    for i,(key,label) in enumerate([('baseline_noise2','Current'),('cubic_refined_noise2','Both')]):
        axes[1].bar(np.arange(4)+(i-.5)*.34,[report['cases'][c]['variants'][key]['field']['total_vector_rmse_px'] for c in cases],.34,label=label)
        axes[2].bar(np.arange(4)+(i-.5)*.34,[report['cases'][c]['variants'][key]['sharpened_error_linear_0_1']['rmse_adu'] for c in cases],.34,label=label)
    for ax in axes[1:]:
        ax.set_xticks(np.arange(4),['Stationary','Long','Short','Short + offset'],rotation=20)
        ax.legend(fontsize=8)
        ax.set_ylim(0,ax.get_ylim()[1]*1.18)
    axes[1].set_ylabel('Motion error, vector RMS pixels')
    axes[1].set_title('Moving/noisy controls')
    axes[2].set_ylabel('Sharpened RMSE vs noiseless target, linear 0–1')
    axes[2].set_title('Same exact sharpening recipe')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Matching sampler and fractional peak refinement; final stacking sampler unchanged')
    fig.savefig(root/'comparison.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()

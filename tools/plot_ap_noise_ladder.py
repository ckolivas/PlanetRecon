"""Plot the audited factorial dose response with the system Matplotlib."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('out/ap-noise-ladder')
    report=json.loads((root/'analysis.json').read_text())
    cases=['stationary','long_motion','short_motion','short_offset']
    labels={'baseline':'Current','cubic':'Cubic only','refined':'Refinement only','cubic_refined':'Both'}
    fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    for col,case in enumerate(cases):
        axes[0,col].set_title(case.replace('_',' ').capitalize())
        for row,metric in enumerate(['field','sharpened']):
            entries=report['cases'][case]['factorial'][metric]
            for policy,label in labels.items():
                axes[row,col].plot([x['sigma_adu'] for x in entries],
                    [x['values'][policy]-x['values']['baseline'] for x in entries],marker='o',label=label)
            axes[row,col].axhline(0,color='black',linewidth=.6)
            axes[row,col].grid(alpha=.2)
            axes[row,col].spines[['top','right']].set_visible(False)
            axes[row,col].set_xlabel('Added noise sigma, ADU')
    axes[0,0].set_ylabel('Motion RMSE difference from current, pixels\nBelow zero is better')
    axes[1,0].set_ylabel('Sharpened RMSE difference, linear 0–1\nBelow zero is better')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Matching-only noise ladder: 16 paired poses per case, fixed noise realization\nWavelet 27/0/0/0 + Adaptive Deconvolution 15.6, Contrast Adaptive')
    fig.savefig(root/'comparison.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()

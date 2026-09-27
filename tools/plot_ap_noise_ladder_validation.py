"""Show development versus unused-phase/fresh-noise refinement effects."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('out/ap-noise-ladder-validation')
    report=json.loads((root/'analysis.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,case in zip(axes.flat,['stationary','long_motion','short_motion','short_offset']):
        rows=report['cases'][case]['relative_change_percent']['sharpened']
        for label in ('development','validation'):
            ax.plot([r['sigma_adu'] for r in rows],[r[label] for r in rows],marker='o',label=label.capitalize())
        ax.axhline(0,color='black',linewidth=.6)
        ax.grid(alpha=.2)
        ax.set_title(case.replace('_',' ').capitalize())
        ax.set_xlabel('Added noise sigma, ADU')
        ax.set_ylabel('Sharpened RMSE change vs current, %\nBelow zero is better')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle('Refinement only: fresh noise and unused phases\n16 poses per stack; same exact sharpening; no parameter retuning')
    fig.savefig(root/'comparison.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()

"""Render local operator-profile timings and measured Torch allocation peaks."""
import argparse
import json
from pathlib import Path
import statistics


def plot(directory,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((directory/'report.json').read_text())
    if report['source_input_unchanged'] is not True:
        raise ValueError('source/input verification failed')
    rows=report['rows'];labels=[r['mode'].replace('-','\n') for r in rows]
    medians=[statistics.median(r['normal_s']) for r in rows]
    allocated=[r['torch_peak_allocated_bytes']/1024**3 for r in rows]
    reserved=[r['torch_peak_reserved_bytes']/1024**3 for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    axes[0].bar(labels,medians,color='#367e85')
    for i,row in enumerate(rows):
        axes[0].scatter([i]*len(row['normal_s']),row['normal_s'],color='#182d39',s=20)
        axes[0].text(i,max(row['normal_s'])*1.04,f'{medians[i]:.3f} s',ha='center')
    axes[0].set_ylim(0,max(medians)*1.25);axes[0].set_ylabel('Seconds per normal product')
    axes[0].set_title('Three local measurements; bars show medians')
    axes[1].bar(labels,reserved,label='Reserved',color='#b8c4c7')
    axes[1].bar(labels,allocated,label='Allocated',color='#367e85')
    axes[1].set_ylabel('GiB');axes[1].set_title('Torch allocator peaks; excludes driver memory')
    axes[1].legend()
    for ax in axes: ax.grid(axis='y',alpha=.2)
    fig.suptitle('500-frame exact-operator profile — no fitted-scene convergence claim')
    fig.savefig(output,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();plot(a.directory,a.output)

"""Plot final independent certificates and accepted projection-ablation progress."""
import argparse
import json
from pathlib import Path


def plot(directory,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((directory/'report.json').read_text())
    if report['source_input_unchanged'] is not True:
        raise ValueError('source/input verification failed')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    labels=[];bounds=[];colours=[]
    for row in report['rows']:
        for fit in row['runs']:
            cert=fit.get('reference_certificate')
            if cert is None: continue
            label=f"{row['mode'].replace('window_','')}\ncap {fit['max_products']}"
            labels.append(label);bounds.append(cert['relative_solution_error_bound'])
            colours.append('#21735a' if cert['feasible'] and bounds[-1]<=1e-5 else '#b67926')
            trace=fit['trace']
            axes[1].semilogy([r['hessian_products'] for r in trace],
                             [r['relative_solution_error_bound'] for r in trace],label=label.replace('\n',', '))
    axes[0].bar(labels,bounds,color=colours);axes[0].set_yscale('log')
    for i,bound in enumerate(bounds): axes[0].text(i,bound*1.2,f'{bound:.3g}',ha='center',fontsize=9)
    axes[0].set_ylim(min([1e-5,*bounds])/5,max([1e-5,*bounds])*4)
    axes[0].set_title('Fresh independent CPU certificate')
    axes[1].set_title('Accepted feasible updates')
    axes[1].set_xlabel('Hessian products');axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(1e-5,color='#a43939',linestyle='--');ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Relative distance upper bound')
    fig.suptitle(f"{report['rows'][0]['n_used']} observed frames, same window inverse and objective; bounds are not measured image errors")
    fig.savefig(output,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();plot(a.directory,a.output)

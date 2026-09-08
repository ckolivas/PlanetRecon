"""Plot the two prospective cropped-FFT endpoint studies with distinct cap units."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.plot_scene_endpoints import endpoint_bounds


def plot(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    labels=[];bounds=[];colours=[]
    for column,method in enumerate(('reference','newton'),1):
        root=results/('p2-cropped-'+method)
        report=json.loads((root/'report.json').read_text())
        if not report['source_input_unchanged']: raise ValueError('changed source/input identity')
        for count,bound,passed in endpoint_bounds(root):
            labels.append(f'{method}\n{count} frames');bounds.append(bound)
            colours.append('#21735a' if passed else '#b67926')
        ax=axes[column]
        for block in report['rows']:
            if block['fraction']!=100: continue
            for fit in block['runs']:
                trace=fit.get('trace',[])
                key='iteration' if method=='reference' else 'hessian_products'
                cap=fit['maxiter'] if method=='reference' else fit['max_products']
                ax.semilogy([r[key] for r in trace],[r['relative_solution_error_bound'] for r in trace],label=f'cap {cap}')
        ax.set_title(f'{method.capitalize()}: 500-frame progress')
        ax.set_xlabel('Iterations' if method=='reference' else 'Hessian products')
        ax.legend(fontsize=8)
    axes[0].bar(labels,bounds,color=colours);axes[0].set_yscale('log')
    axes[0].set_ylim(min([1e-5,*bounds])/5,max([1e-5,*bounds])*4)
    for i,bound in enumerate(bounds): axes[0].text(i,bound*1.2,f'{bound:.3g}',ha='center',fontsize=8)
    axes[0].set_title('Worst independent bound across both caps')
    for ax in axes:
        ax.axhline(1e-5,color='#a43939',linestyle='--');ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Relative distance upper bound')
    fig.suptitle('Same objective and exact-crop backend; cap units differ; bounds are not measured image errors')
    fig.savefig(output,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();plot(a.results,a.output)

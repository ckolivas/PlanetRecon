"""Plot full and reduced linear probes separately from scene certification."""
import argparse
import hashlib
import json
from pathlib import Path


def plot(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,digest in json.loads((root/'checksums.json').read_text()).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('archive checksum mismatch: '+name)
    report=json.loads((root/'report.json').read_text())
    if not report['source_input_unchanged'] or report['scene_updated']:
        raise ValueError('unchanged frozen scene required')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for row in report['rows']:
        system,mode=row['mode'].split('-')
        axis=axes[int(system=='reduced')];trace=row['probe']['trace']
        counts=[v['hessian_products'] for v in trace]
        axis.semilogy([0]+counts,[1.]+[v['recursive_relative_residual'] for v in trace],label=mode)
        if counts:
            axis.scatter([counts[-1]],[row['independent_system_relative_residual']],marker='x',s=70,color='black',zorder=3)
    for axis,title in zip(axes,['Full linear system','Fixed free-variable system']):
        axis.set_title(title);axis.set_xlabel('Exact Hessian products')
        axis.set_ylabel('Linear residual norm / initial norm')
        axis.axhline(1.,color='gray',linestyle=':',linewidth=1)
        axis.plot([],[],color='black',marker='x',linestyle='none',label='Independent CPU terminal check')
        axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.suptitle('Saved iteration 142; scene unchanged; these residuals are not scene-error bounds',fontsize=11)
    fig.savefig(output,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.root,a.output)

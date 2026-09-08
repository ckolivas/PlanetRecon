"""Summarize independently verified probes and exploratory projection geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.scene_projection_geometry import projection_geometry


def summarize(directory, reference):
    report = json.loads((directory/'report.json').read_text())
    if report['source_input_unchanged'] is not True:
        raise ValueError('original study source/input verification failed')
    protocol = json.loads((directory/'protocol.json').read_text())
    x = {}
    for fraction, path, key in [(5, reference/'stages/5/budget-750.npz', 'image'),
                                (100, reference/'iterations/100-750.npz', 'x')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol['input_hashes'][str(path)]:
            raise ValueError('frozen state checksum mismatch')
        with np.load(path, allow_pickle=False) as data: x[fraction] = data[key].copy()
    rows = []; hashes = {}
    for row in report['rows']:
        key = f"{row['fraction']}-{row['state']}-{row['mode']}"
        path = directory/('correction-'+key+'.npz')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data: direction = -data['correction']
        initial = x[row['fraction']] if row['state']=='late' else np.zeros_like(direction)
        rows.append({name:row[name] for name in ('fraction','state','mode','independent_reduced_relative_residual',
                                                'independent_full_relative_residual','product_relative_error')} |
                    {'unit_step_geometry':projection_geometry(initial,direction)})
    spectra = []
    for item in report['spectra']:
        for frequency in item['frequencies']:
            energy = frequency['independent_energy']
            spectra.append({'fraction':item['fraction'], **frequency,
                            'periodic_relative_energy_error':abs(frequency['periodic_symbol']/energy-1),
                            'window_relative_energy_error':abs(frequency['window_symbol']/energy-1)})
    return {'status':report['status'], 'probe_rows':rows, 'fourier_checks':spectra,
            'initials':report['initials'], 'correction_sha256':hashes,
            'analysis_source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                     [Path('tools/analyze_scene_window_states.py'), Path('tools/scene_projection_geometry.py')]},
            'scope':'Projection geometry is exploratory, computed after the frozen study. Unit steps were not accepted, scored or applied. No scene or convergence claim.'}


def plot(directory, destination):
    import matplotlib.pyplot as plt
    report = json.loads((directory/'report.json').read_text())
    fig, axes = plt.subplots(2,2,figsize=(10,7),sharex=True)
    for row in report['rows']:
        ax = axes[0 if row['fraction']==5 else 1,0 if row['state']=='zero' else 1]
        trace = row['probe']['trace']
        ax.semilogy([t['iteration'] for t in trace],[t['recursive_relative_residual'] for t in trace],label=row['mode'])
        ax.set_title(f"{25 if row['fraction']==5 else 500} frames — {row['state']} state")
    for ax in axes.flat:
        ax.grid(alpha=.25); ax.set_ylabel('Reduced residual / initial'); ax.legend()
    for ax in axes[-1]: ax.set_xlabel('Hessian products')
    fig.suptitle('Frozen linear probes: these curves do not certify a fitted image')
    fig.tight_layout(); fig.savefig(destination,dpi=150); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--reference',type=Path,default=Path('out/p2-selection-endpoints-reference'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--plot',type=Path)
    a=p.parse_args();a.out.write_text(json.dumps(summarize(a.directory,a.reference),indent=2)+'\n')
    if a.plot: plot(a.directory,a.plot)

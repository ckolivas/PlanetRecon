"""Plot archived frozen-iterate probes; residual traces are not certificates."""
import argparse
import hashlib
import json
from pathlib import Path


def read_study(path):
    checks = json.loads((path/'checksums.json').read_text())
    for name, digest in checks.items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest() != digest:
            raise ValueError('archive checksum mismatch: '+name)
    report = json.loads((path/'report.json').read_text())
    if report['status'] != 'valid' or not report['source_input_unchanged'] or report['scene_updated']:
        raise ValueError('a valid unchanged frozen-iterate diagnostic is required')
    return json.loads((path/'protocol.json').read_text()), report


def plot(studies, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [read_study(path) for path in studies]
    for protocol, _ in records[1:]:
        for key in ('input_sha256', 'checkpoint_sha256', 'original_protocol_sha256', 'manifest_sha256'):
            if protocol[key] != records[0][0][key]: raise ValueError('different frozen objectives')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for _, report in records:
        for row in report['rows']:
            trace = row['probe']['trace']
            n = [r['hessian_evaluations'] for r in trace]
            axes[0].plot([0]+n, [1.]+[r['recursive_relative_residual'] for r in trace], label=row['mode'])
            axes[1].semilogy(n, [r['direction_rayleigh'] for r in trace], label=row['mode'])
    axes[0].axhline(1., color='gray', linestyle=':', linewidth=1)
    axes[0].set_ylabel('Recursive residual norm / initial norm')
    axes[0].set_title('Linear correction progress (not a certificate)')
    axes[1].set_ylabel('Directional Rayleigh quotient of H')
    axes[1].set_title('Sampled directions (not spectral endpoints)')
    for axis in axes:
        axis.set_xlabel('Hessian products'); axis.grid(alpha=.2); axis.legend()
    fig.suptitle('Same saved iteration 142; 500 frames; scene never updated')
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('studies', type=Path, nargs='+'); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); plot(a.studies, a.output)

"""Trace AP decisions under the completed reference-only interventions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.io.ser import SERSource
from planetrecon.pipeline.local_align import LocalRegistration
from tools.coherent_saturn_experiment import inputs
from tools.joint_saturn_experiment import CAPTURE, digest
from tools.local_warp_trace import trace
from tools.reference_imprint_probe import probes, variants, ReferenceStates, engine_for, source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    _, indices, _, shifts, reference, hashes = inputs()
    hashes.update(source_hashes())
    hashes['tools/reference_imprint_gates.py'] = digest(__file__)
    hashes['tools/local_warp_trace.py'] = digest('tools/local_warp_trace.py')
    torch.set_num_threads(4)
    selected = np.linspace(0, len(indices)-1, 512, dtype=int)
    positions = selected[np.linspace(0, 511, 16, dtype=int)]
    patterns, mask = probes(reference)
    report = dict(hashes=hashes, positions=positions.tolist(), methods={})
    with SERSource(CAPTURE) as source:
        for method in ('production', 'coherent'):
            engine = engine_for(reference, method, 'cuda:0')
            states = ReferenceStates(engine, reference, patterns)
            rows = []
            for pos in positions:
                proxy = LocalRegistration.proxy(source.read_raw(int(indices[pos])))
                fields, measurements, row = {}, {}, dict(position=int(pos), variants={})
                for key in variants():
                    states.select(key)
                    stages, records = trace(engine, proxy, shifts[pos])
                    field = engine.displacement(proxy, shifts[pos], lambda: None)
                    if method == 'production':
                        torch.testing.assert_close(field, stages[-1], atol=1e-10, rtol=0)
                    fields[key] = field
                    values = np.concatenate([r['measurements'] for r in records])
                    measurements[key] = values
                    item = dict(stage_acceptance=[r['accepted'] for r in records],
                        stage_minimum_jacobian=[r['minimum_jacobian'] for r in records],
                        confident_ap_count=int(np.count_nonzero(values[:, 4])), ap_count=len(values))
                    if method == 'coherent':
                        item['spline_guard'] = engine.stats[-1].copy()
                    if key != 'base':
                        before = measurements['base']
                        item['changed_active_aps'] = int(np.count_nonzero((values[:, 4]>0)!=(before[:, 4]>0)))
                        item['ap_displacement_difference_rms_px'] = float(np.sqrt(np.mean(np.sum((values[:, :2]-before[:, :2])**2, axis=1))))
                        item['field_difference_rms_px'] = float((field-fields['base'])[:, mask].square().sum(0).mean().sqrt())
                    row['variants'][key] = item
                rows.append(row)
            report['methods'][method] = rows
            print(method, 'completed', len(rows), 'frames', flush=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

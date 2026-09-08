"""Account exact optimizer normal calls for the explicitly supported solver versions.

Initialization/final gradients are included; constructor setup, objective forward
calls and independent CPU verification are separate. Counts are not wall-time or
accuracy equivalence between solvers.
"""

def normal_work(fit):
    solver=fit.get('solver')
    if solver=='extended_scene_projected_acceleration_v4':
        trace=fit['trace']
        steps=fit['n_iter']-fit['resume_from_iteration']
        if steps<0: raise ValueError('invalid resumed iteration count')
        # Each completed iteration evaluates one gradient; every callback follows
        # an additional certificate product. Entry/exit each recompute a gradient.
        total=steps+len(trace)+2
        progress=[r['iteration']-fit['resume_from_iteration']+i+2 for i,r in enumerate(trace)]
    elif solver=='extended_scene_projected_newton_cg_v2':
        if fit['initial_final_gradient_evaluations']!=2:
            raise ValueError('unsupported initialization/finalization count')
        total=fit['hessian_products']+2
        progress=[r['hessian_products']+1 for r in fit['trace']]
    else:
        raise ValueError('unsupported solver version for exact work accounting')
    return {'optimizer_normal_products':total,'accepted_trace_normal_products':progress,
            'initial_final_included':True,
            'scope':'Current optimizer attempt only; excludes setup, forward objectives and independent CPU verification. Equal counts do not imply equal time or accuracy.'}


if __name__=='__main__':
    import argparse
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from tools.study_io import file_hash,write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise ValueError('use a new analysis output')
    report_path=args.directory/'report.json';protocol_path=args.directory/'protocol.json'
    report=json.loads(report_path.read_text());protocol=json.loads(protocol_path.read_text())
    if not report['source_input_unchanged']: raise ValueError('unchanged source/input study required')
    for name in ('tools/scene_quadratic.py','tools/scene_projected_newton.py'):
        if protocol['identities']['dependencies'].get(name)!=file_hash(Path(name)):
            raise ValueError('work accounting requires the tested solver source: '+name)
    rows=[]
    for block in report['rows']:
        for fit in block['runs']:
            row={'fraction':block['fraction'],'nominal_cap':fit.get('maxiter',fit.get('max_products')),
                 'work_available':'solver' in fit}
            if row['work_available']:row.update(normal_work(fit))
            rows.append(row)
    write_json(args.out,{'rows':rows,'source_report_sha256':file_hash(report_path),
                        'source_protocol_sha256':file_hash(protocol_path),'analysis_source_sha256':file_hash(Path(__file__)),
                        'qualification_changed':False})

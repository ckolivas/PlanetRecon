"""Summarize observed inner-solve progress without treating it as convergence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.study_io import file_hash,write_json


def summarize(report, target=.1):
    if not report['source_input_unchanged']: raise ValueError('unchanged study identity required')
    rows=[]
    for block in report['rows']:
        for fit in block['runs']:
            trace=fit.get('trace',[])
            observed=[r for r in trace if r.get('inner_trace')]
            terminal=[r['inner_trace'][-1]['recursive_relative_residual'] for r in observed]
            rows.append({'fraction':block['fraction'],'max_products':fit.get('max_products'),
                         'accepted_updates':len(trace),'updates_with_inner_trace':len(observed),
                         'updates_reaching_inner_target':sum(v<=target for v in terminal),
                         'inner_terminal_residual_min':min(terminal) if terminal else None,
                         'inner_terminal_residual_max':max(terminal) if terminal else None,
                         'accepted_inner_products':sum(r['inner_products'] for r in trace),
                         'accepted_line_search_products':sum(r['line_search_products'] for r in trace),
                         'unaccepted_inner_products_recorded':len(fit.get('unaccepted_inner_trace',[])),
                         'safeguard_updates':sum(r['step_kind']=='projected_majorizer' for r in trace),
                         'converged':fit.get('converged',False),
                         'reference_certificate':fit.get('reference_certificate'),
                         'reason':fit.get('reason')})
    return {'rows':rows,'inner_relative_target':target,'qualification_changed':False,
            'scope':'Descriptive trace analysis; inner targets, residuals and active counts are not scene certificates. Missing historical inner traces are not inferred.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists(): raise ValueError('use a new analysis output')
    result=summarize(json.loads(a.report.read_text()))
    result['source_report_sha256']=file_hash(a.report)
    write_json(a.out,result)

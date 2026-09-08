from tools.analyse_newton_inner import summarize


def test_historical_missing_inner_traces_are_not_inferred():
    r={'source_input_unchanged':True,'rows':[{'fraction':5,'runs':[{'max_products':750,'converged':False,
       'trace':[{'inner_products':3,'line_search_products':1,'step_kind':'newton'}]}]}]}
    row=summarize(r)['rows'][0]
    assert row['accepted_updates']==1 and row['updates_with_inner_trace']==0
    assert row['inner_terminal_residual_min'] is None and not row['converged']


def test_inner_targets_never_promote_outer_convergence():
    r={'source_input_unchanged':True,'rows':[{'fraction':100,'runs':[{'max_products':750,'converged':False,
       'unaccepted_inner_trace':[{'recursive_relative_residual':.2}],
       'trace':[{'inner_products':2,'line_search_products':1,'step_kind':'newton',
                 'inner_trace':[{'recursive_relative_residual':.4},{'recursive_relative_residual':.09}]}]}]}]}
    result=summarize(r);row=result['rows'][0]
    assert row['updates_reaching_inner_target']==1 and row['unaccepted_inner_products_recorded']==1
    assert not row['converged'] and not result['qualification_changed']

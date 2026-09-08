from tools.audit_scene_family import complete_cases


def test_family_requires_every_crop_seed_regime_once():
    rows=[{'seed':s,'dr0':d,'crop':c} for s in (1001,1002,1003) for d in (4.,8.) for c in ('feature','bland')]
    assert complete_cases(rows)
    assert not complete_cases(rows[:-1])
    assert not complete_cases(rows+rows[:1])
    assert not complete_cases(rows[:-1]+rows[:1])

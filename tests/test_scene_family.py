from tools.audit_scene_family import complete_cases


def test_family_requires_every_crop_seed_regime_once():
    rows=[{'seed':s,'dr0':d,'crop':c} for s in (1001,1002,1003) for d in (4.,8.) for c in ('feature','bland')]
    assert complete_cases(rows)
    assert not complete_cases(rows[:-1])
    assert not complete_cases(rows+rows[:1])
    assert not complete_cases(rows[:-1]+rows[:1])


def test_resume_creates_unstarted_case_and_continues_timed_out_stage(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    import numpy as np
    import tools.audit_scene_family as audit
    from tools.scene_study import CellBasisOperator,TRAIN_LARGE
    from planetrecon.operators import SceneDetectorOperator
    from tools.study_io import file_hash
    class Data:
        cfg=SimpleNamespace(seed=1001,dr0=4.,n_frames=500)
        def __init__(self,*args):
            self.images={i:np.ones((4,4)) for i in TRAIN_LARGE}
            self.variances={i:1. for i in TRAIN_LARGE}
        def operators(self,indices,**kwargs):
            return [CellBasisOperator(SceneDetectorOperator((4,4),(np.ones((1,1)),),1,(0,0),(4,4)),1) for i in indices]
    path=tmp_path/'input.h5';path.write_bytes(b'bounded mocked input')
    protocol={'input_sha256':{path.name:file_hash(path)},'case_budget_s':20.,'sum_native_strength':.003,
              'device':'cpu','cache_bytes':0,'budgets':[10,20],'tolerance':1e-5,'image_tolerance':1e-4}
    monkeypatch.setattr(audit,'ObservedSceneData',Data)
    original=audit.solve
    def expire(problem,**kwargs):
        return original(problem,**{**kwargs,'deadline':time.monotonic()-1})
    monkeypatch.setattr(audit,'solve',expire)
    payload=(path,'feature',tmp_path/'study',protocol,time.monotonic()+30,True)
    failed=audit.case(payload)
    assert not failed['numerical_passed']
    root=tmp_path/'study'/'1001-4-feature'
    assert len(list(root.glob('incomplete-*.json')))==2
    assert not list((root/'stages').glob('budget-*.npz'))
    monkeypatch.setattr(audit,'solve',original)
    completed=audit.case(payload)
    assert completed['numerical_passed']
    assert len(list(root.glob('incomplete-*.json')))==2
    assert len(list((root/'stages').glob('budget-*.npz')))==2

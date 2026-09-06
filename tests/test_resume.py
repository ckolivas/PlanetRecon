from dataclasses import replace
from pathlib import Path
import os
import numpy as np
import pytest
from planetrecon.io.ser import write_ser, SERSource, COLOR_RGGB
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon import resume


@pytest.mark.parametrize('bayer',[False,True])
def test_exact_resume_with_rejections_and_no_double_count(tmp_path,bayer):
    rng=np.random.default_rng(12)
    frames=rng.integers(0,200,(11,12,16),dtype='u2');frames[3]=65535
    path=write_ser(tmp_path/'in.ser',frames,color_id=COLOR_RGGB if bayer else 0)
    cfg=ReconstructionConfig(device='cpu',threads=2,batch_frames=2,max_shift_px=100)
    state=tmp_path/'state.npz'
    with SERSource(path) as src:whole=stack_source(src,cfg)
    cancel=False
    def event(result,info):
        nonlocal cancel
        if info['n_processed']>=4:cancel=True
    with SERSource(path) as src:
        partial=stack_source(src,cfg,on_event=event,should_cancel=lambda:cancel,state_checkpoint=state)
    assert partial.incomplete and partial.n_rejected==1
    with SERSource(path) as src:
        continued=stack_source(src,cfg,resume_from=state,state_checkpoint=state)
    for attr in ('image','coverage','validity'):
        np.testing.assert_array_equal(getattr(continued,attr),getattr(whole,attr))
    assert continued.n_used==whole.n_used and continued.n_rejected==whole.n_rejected
    with SERSource(path) as src:
        repeated=stack_source(src,cfg,resume_from=state)
    np.testing.assert_array_equal(repeated.image,whole.image)


def test_resume_rejects_changed_input_config_and_alias(tmp_path):
    path=write_ser(tmp_path/'in.ser',np.ones((4,8,8),dtype='u2'))
    cfg=ReconstructionConfig(device='cpu',threads=2,batch_frames=2)
    state=tmp_path/'state.npz'
    with SERSource(path) as src:
        stack_source(src,cfg,state_checkpoint=state)
        with pytest.raises(ValueError,match='cannot replace'):
            stack_source(src,cfg,state_checkpoint=path)
        with pytest.raises(ValueError,match='mismatch'):
            stack_source(src,replace(cfg,batch_frames=1),resume_from=state)
    st=path.stat()
    with path.open('r+b') as fh:fh.seek(190);fh.write(b'\x02')
    os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
    with SERSource(path) as src:
        with pytest.raises(ValueError,match='mismatch'):stack_source(src,cfg,resume_from=state)


def test_failed_state_write_preserves_prior(tmp_path,monkeypatch):
    path=tmp_path/'state.npz';path.write_bytes(b'prior')
    def fail(*args,**kwargs):raise OSError('No space left on device')
    monkeypatch.setattr(np,'savez_compressed',fail)
    state={key:None for key in resume.ARRAYS}
    state.update(n_used=0,n_rejected=0,reference_index=None,next_index=0)
    with pytest.raises(OSError):resume.save(path,{},state)
    assert path.read_bytes()==b'prior' and list(tmp_path.iterdir())==[path]


def test_cli_resumable_state(tmp_path):
    from planetrecon.cli import main
    path=write_ser(tmp_path/'in.ser',np.ones((4,8,8),dtype='u2'))
    state=tmp_path/'state.npz'
    args=['--threads','2','stack','--path',str(path),'--out',str(tmp_path/'out'),'--device','cpu']
    assert main(args+['--state-checkpoint',str(state)])==0
    assert main(args+['--resume',str(state)])==0
    with pytest.raises(SystemExit):main(args+['--state-checkpoint',str(tmp_path/'out/stack.npz')])


@pytest.mark.skipif(not Path('/proc/self/stat').exists(),reason='local process-crash integration uses Linux procfs')
def test_owned_worker_exits_after_parent_crash(tmp_path):
    import subprocess,sys,time
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    script=tmp_path/'owner.py'
    ready=tmp_path/'ready'
    script.write_text('''import multiprocessing,sys,time,os
sys.path.insert(0,sys.argv[2])
from planetrecon.jobs import _watch_parent
from pathlib import Path
def child(path):
    _watch_parent()
    Path(path).write_text(str(os.getpid()))
    time.sleep(60)
if __name__=='__main__':
    proc=multiprocessing.get_context('spawn').Process(target=child,args=(sys.argv[1],))
    proc.start()
    time.sleep(60)
''')
    owner=subprocess.Popen([sys.executable,str(script),str(ready),str(root)])
    pid=None
    try:
        deadline=time.monotonic()+10
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists()
        pid=int(ready.read_text());owner.kill();owner.wait(timeout=5)
        def live():
            try:
                stat=Path(f'/proc/{pid}/stat').read_text()
                return stat.split(') ',1)[1][0]!='Z'
            except FileNotFoundError:return False
        deadline=time.monotonic()+5
        while live() and time.monotonic()<deadline:time.sleep(.02)
        assert not live()
    finally:
        if owner.poll() is None:owner.kill();owner.wait(timeout=5)
        if pid is not None:
            try:os.kill(pid,9)
            except ProcessLookupError:pass

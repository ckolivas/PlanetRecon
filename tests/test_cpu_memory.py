import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

pytestmark=pytest.mark.skipif(sys.platform!='linux',reason='Linux RLIMIT_AS qualification')


def test_kernel_denies_allocation_and_restores_limit_in_isolated_process():
    # Do not lower the test runner's address-space limit or allocate large RAM.
    script=textwrap.dedent('''
        import json,resource
        from planetrecon.memory import cpu_memory_limit,virtual_bytes
        original=resource.getrlimit(resource.RLIMIT_AS)
        cap=virtual_bytes()+16*1024**2
        denied=False
        with cpu_memory_limit(cap) as report:
            try: value=bytearray(64*1024**2)
            except MemoryError: denied=True
            assert denied
        assert resource.getrlimit(resource.RLIMIT_AS)==original
        assert report['enforced'] and report['effective_bytes']==cap
        print(json.dumps(report))
    ''')
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=20,check=True)
    assert json.loads(result.stdout)['enforced']


def test_cpu_stack_under_ceiling_and_immediate_rerun():
    script=textwrap.dedent('''
        import resource,numpy as np
        from dataclasses import replace
        from planetrecon.memory import virtual_bytes
        from planetrecon.io.source import ArraySource
        from planetrecon.reconstruction import ReconstructionConfig
        from planetrecon.pipeline.baseline import stack_source
        from planetrecon.runtime import apply_thread_limits
        apply_thread_limits(2)
        cfg=ReconstructionConfig(device='cpu',threads=2,batch_frames=1)
        frames=np.arange(3*16*24,dtype='u2').reshape(3,16,24)+1
        whole=stack_source(ArraySource(frames),cfg)
        original=resource.getrlimit(resource.RLIMIT_AS)
        limited=stack_source(ArraySource(frames),replace(cfg,max_ram_bytes=virtual_bytes()+128*1024**2))
        np.testing.assert_array_equal(limited.image,whole.image)
        assert limited.provenance['cpu_memory_budget']['enforced']
        assert resource.getrlimit(resource.RLIMIT_AS)==original
        try: stack_source(ArraySource(frames),replace(cfg,max_ram_bytes=1024))
        except MemoryError: pass
        else: raise AssertionError('impossibly small cap was accepted')
        again=stack_source(ArraySource(frames),cfg)
        np.testing.assert_array_equal(again.image,whole.image)
    ''')
    subprocess.run([sys.executable,'-c',script],check=True,timeout=30)


def test_owned_worker_reports_small_limit_and_cleans_up(tmp_path):
    import time
    import numpy as np
    from planetrecon.io.ser import write_ser
    from planetrecon.jobs import start_stack_job
    from planetrecon.reconstruction import ReconstructionConfig
    source=write_ser(tmp_path/'in.ser',np.ones((3,8,12),dtype='u2'))
    handle=start_stack_job(source,ReconstructionConfig(device='cpu',threads=2,max_ram_bytes=1024))
    directory=Path(handle.events.directory)
    events=[]
    try:
        deadline=time.monotonic()+10
        while handle.state not in ('failed','completed') and time.monotonic()<deadline:
            events.extend(handle.poll(.05))
        assert handle.state=='failed'
        assert any('memory cap' in event.payload.get('message','') for event in events)
    finally:handle.close()
    assert not handle.process.is_alive() and not directory.exists()

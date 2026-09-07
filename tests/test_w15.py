import multiprocessing
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import time

import numpy as np
import pytest

from planetrecon.event_transport import FileEventQueue
from planetrecon.io.source import open_source, ArraySource
from planetrecon.io.ser import SERSource, SERHeader, pack_header, SER_FILE_ID, write_ser
from planetrecon.jobs import JobEvent, JobHandle
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def avi(path, frames, *, top_down=False, compressed=False):
    n, h, w = frames.shape[:3]
    channels = 3 if frames.ndim == 4 else 1
    def chunk(tag, data):
        return tag + struct.pack('<I', len(data)) + data + b'\0' * (len(data) & 1)
    sh = bytearray(56)
    sh[:8] = b'vidsDIB '
    struct.pack_into('<4I', sh, 20, 1, 25, 0, n)
    fmt = struct.pack('<IiiHHIIiiII', 40, w, -h if top_down else h, 1, channels*8,
        1 if compressed else 0, 0, 0, 0, 256 if channels == 1 else 0, 0)
    if channels == 1:
        fmt += b''.join(bytes((i,i,i,0)) for i in range(256))
    headers = chunk(b'LIST', b'hdrl' + chunk(b'LIST', b'strl' + chunk(b'strh', bytes(sh)) + chunk(b'strf', fmt)))
    data = []
    for frame in frames:
        rows = frame if top_down else frame[::-1]
        if channels == 3:
            rows = rows[..., ::-1]
        data.append(chunk(b'00db', b''.join(row.tobytes() + b'\0'*((-w*channels)%4) for row in rows)))
    payload = b'AVI ' + headers + chunk(b'LIST', b'movi' + b''.join(data))
    path.write_bytes(chunk(b'RIFF', payload))
    return path


@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('top_down', [False, True])
def test_avi_exact_and_indexed(tmp_path, rgb, top_down):
    shape = (3, 7, 9, 3) if rgb else (3, 7, 9)
    frames = (np.arange(np.prod(shape)) % 239).astype('u1').reshape(shape)
    path = avi(tmp_path/'fixture.avi', frames, top_down=top_down)
    with open_source(path) as src:
        for i in [2,0,1]:
            np.testing.assert_array_equal(src.read_raw(i), frames[i])
        assert src.timestamps() is None
        assert src.metadata().units == 'decoded-code-value'
        assert src._index.seek(0,2) == 3*8
        out = stack_source(src, ReconstructionConfig(frame_preselection=False, device='cpu',threads=2, max_shift_px=100))
        assert out.n_used == 3 and any('linearity' in w for w in out.warnings)


def test_avi_independent_ffmpeg_readback(tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('independent development decoder unavailable')
    frames = np.random.default_rng(5).integers(0,240,(2,8,10,3),dtype='u1')
    path = avi(tmp_path/'independent.avi',frames)
    decoded = subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','-'])
    np.testing.assert_array_equal(np.frombuffer(decoded,'u1').reshape(frames.shape),frames)


def test_avi_reject_and_input_changes(tmp_path):
    frames=np.ones((2,8,9),dtype='u1')
    path=avi(tmp_path/'x.avi',frames,compressed=True)
    with pytest.raises(ValueError,match='supported AVI pixels'):
        open_source(path)
    avi(path,frames)
    with open_source(path) as src:
        path.write_bytes(path.read_bytes()[:-1])
        with pytest.raises(OSError,match='changed'):
            src.read_raw(0)
    with pytest.raises(ValueError):
        open_source(path)


def test_ser_disappearance_truncation_and_sparse_large_input(tmp_path):
    path=tmp_path/'large.ser'
    header=SERHeader(SER_FILE_ID,0,0,0,1024,1024,16,4096,'','','',0,0)
    with path.open('wb') as fh:
        fh.write(pack_header(header))
        fh.truncate(178 + header.frame_bytes*4096) # 8 GiB logical capture, no bulk allocation
    with SERSource(path) as src:
        assert src.read_raw(4095).nbytes == 2*1024**2
        with path.open('r+b') as fh:
            fh.truncate(178)
        with pytest.raises(OSError,match='changed'):
            src.read_raw(4095)
    write_ser(path,np.ones((2,8,8),dtype='u2'))
    with SERSource(path) as src:
        path.unlink()
        with pytest.raises(FileNotFoundError):
            src.read_raw(0)


def test_batch_memory_and_pre_read_cancellation():
    src=ArraySource(np.ones((13,10,10),dtype='u2'))
    batches=list(src.iter_batches(100,max_bytes=600))
    assert [len(i) for i,_ in batches] == [3,3,3,3,1]
    src.read_raw=lambda index: pytest.fail('must not read after cancellation')
    assert list(src.iter_batches(2,should_cancel=lambda:True)) == []


def _large_sender(events):
    events.put(JobEvent('large',1,'snapshot',{'image':np.ones((4096,4096))}))


def test_killed_payload_sender_cannot_block_poll(tmp_path):
    ctx=multiprocessing.get_context('spawn')
    events=FileEventQueue.create(ctx,2)
    proc=ctx.Process(target=_large_sender,args=(events,))
    handle=JobHandle('large',proc,events,ctx.Event())
    proc.start()
    time.sleep(.1)
    proc.kill(); proc.join(5)
    start=time.monotonic()
    messages=handle.poll(timeout=.05)
    assert time.monotonic()-start < 2
    assert messages[-1].kind == 'error'
    spool=Path(events.directory)
    handle.close()
    assert not spool.exists()


def test_event_queue_full_cleans_unpublished_payloads():
    ctx=multiprocessing.get_context('spawn')
    events=FileEventQueue.create(ctx,2)
    try:
        events.put_nowait(JobEvent('x',1,'progress'))
        events.put_nowait(JobEvent('x',2,'progress'))
        with pytest.raises(queue.Full):
            events.put_nowait(JobEvent('x',3,'progress'))
        assert len(list(Path(events.directory).glob('*.event'))) == 2
        assert events.get(timeout=1).seq == 1
        assert events.get(timeout=1).seq == 2
        assert not list(Path(events.directory).glob('*.event'))
        assert not list(Path(events.directory).glob('*.tmp'))
    finally:
        events.close()

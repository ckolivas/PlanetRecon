import multiprocessing
from pathlib import Path
import sys

import pytest

from planetrecon import spool
from planetrecon.event_transport import FileEventQueue

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux lease qualification')


def hold_queue(events, ready, stop):
    events.put_nowait({'payload': 'owned by worker'})
    ready.set()
    stop.wait(15)


def test_spawn_worker_lease_survives_owner_release_and_reaps_after_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(spool, 'spool_root', lambda: tmp_path)
    ctx = multiprocessing.get_context('spawn')
    events = FileEventQueue.create(ctx, 2)
    directory = Path(events.directory)
    ready, stop = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=hold_queue, args=(events, ready, stop))
    proc.start()
    try:
        assert ready.wait(10)
        # Model abrupt owner death: its OS lease closes; the child's remains.
        spool.release(events._lease)
        events._lease = None
        assert spool.reap(tmp_path) == [] and directory.exists()
        proc.kill()
        proc.join(5)
        assert not proc.is_alive()
        assert spool.reap(tmp_path) == [directory]
        assert not directory.exists()
    finally:
        if proc.is_alive(): proc.kill()
        proc.join(5)
        events.close()


def test_startup_reaps_only_abandoned_spools(tmp_path, monkeypatch):
    monkeypatch.setattr(spool, 'spool_root', lambda: tmp_path)
    live, lease = spool.create()
    stale, old_lease = spool.create()
    spool.release(old_lease)
    (Path(stale) / 'incomplete.tmp').write_bytes(b'partial snapshot')
    unknown = tmp_path / 'events-unrecognised'
    unknown.mkdir()
    outside = tmp_path / 'unrelated'
    outside.mkdir()
    (tmp_path / 'events-link').symlink_to(outside, target_is_directory=True)
    new, new_lease = spool.create()
    try:
        assert Path(live).exists() and Path(new).exists()
        assert not Path(stale).exists()
        assert unknown.exists() and outside.exists()
    finally:
        spool.release(lease)
        spool.release(new_lease)


def test_spool_root_rejects_nonprivate_directory(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(spool.tempfile, 'gettempdir', lambda: str(tmp_path))
    root = tmp_path / f'planetrecon-spools-{os.getuid()}'
    root.mkdir(mode=0o755)
    with pytest.raises(ValueError, match='private directory'):
        spool.spool_root()

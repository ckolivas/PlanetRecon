"""Linux event-spool leases, held independently by owner and spawned worker.

Reap only directories whose leases can be acquired exclusively. PID reuse and
parent crashes cannot make an active worker's payload directory look abandoned.
Other platforms retain ordinary owner-close cleanup until locally qualified.
"""
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile


def spool_root():
    if sys.platform != 'linux':
        return None
    root = Path(tempfile.gettempdir()) / f'planetrecon-spools-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('event spool root must be a private directory owned by this user')
    return root


def acquire(directory):
    if sys.platform != 'linux':
        return None
    import fcntl
    fd = os.open(Path(directory) / '.lease', os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def release(fd):
    if fd is not None:
        os.close(fd)


def reap(root):
    """Remove inactive generated spools without reading their payloads."""
    import fcntl
    removed = []
    for path in Path(root).glob('events-*'):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            continue
        try:
            fd = os.open(path / '.lease', os.O_RDWR | os.O_NOFOLLOW)
        except (FileNotFoundError, OSError):
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            shutil.rmtree(path)
            removed.append(path)
        finally:
            os.close(fd)
    return removed


def create():
    root = spool_root()
    if root is None:
        return tempfile.mkdtemp(prefix='planetrecon-events-'), None
    reap(root)
    directory = tempfile.mkdtemp(prefix='events-', dir=root)
    # Hold a shared lock before publishing .lease so another starter cannot
    # reap the directory in the interval between file creation and locking.
    import fcntl
    fd = os.open(Path(directory) / '.initial', os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        os.rename(Path(directory) / '.initial', Path(directory) / '.lease')
    except BaseException:
        os.close(fd)
        shutil.rmtree(directory)
        raise
    return directory, fd

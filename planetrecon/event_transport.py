"""Bounded disk payloads; only small atomic filename messages cross the pipe.

An interrupted child can abandon a temporary payload, never a partial array
pickle in the GUI pipe. The owner removes the private spool after joining it.
"""
from pathlib import Path
import pickle
import queue
import shutil
import tempfile
import uuid


class FileEventQueue:
    def __init__(self, queue, directory, slots):
        self.queue = queue
        self.directory = str(directory)
        self.slots = slots

    @classmethod
    def create(cls, ctx, maxsize):
        return cls(ctx.Queue(maxsize=maxsize), tempfile.mkdtemp(prefix='planetrecon-events-'), ctx.BoundedSemaphore(maxsize))

    def put(self, event, block=True, timeout=None):
        if not self.slots.acquire(block, timeout):
            raise queue.Full
        path = Path(self.directory) / (uuid.uuid4().hex + '.event')
        tmp = path.with_suffix('.tmp')
        try:
            with tmp.open('xb') as fh:
                pickle.dump(event, fh, protocol=5)
            tmp.replace(path)
            self.queue.put(path.name, block=block, timeout=timeout)
        except BaseException:
            self.slots.release()
            tmp.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            raise

    def put_nowait(self, event):
        self.put(event, block=False)

    def get(self, block=True, timeout=None):
        name = self.queue.get(block=block, timeout=timeout)
        self.slots.release()
        if Path(name).name != name:
            raise ValueError('invalid event filename')
        path = Path(self.directory) / name
        try:
            with path.open('rb') as fh:
                return pickle.load(fh)
        finally:
            path.unlink(missing_ok=True)

    def get_nowait(self):
        return self.get(block=False)

    def cancel_join_thread(self):
        self.queue.cancel_join_thread()

    def close(self):
        self.queue.close()
        shutil.rmtree(self.directory, ignore_errors=True)

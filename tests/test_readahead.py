import threading

import numpy as np
import pytest

from planetrecon.io.readahead import Readahead
from planetrecon.io.source import ArraySource


class CountingSource(ArraySource):
    """Records reads and fails on request; reads must never overlap."""

    def __init__(self, frames, fail=None):
        super().__init__(frames)
        self.reads, self.fail, self.active = [], fail, threading.Lock()

    def read_raw(self, index):
        assert self.active.acquire(blocking=False), 'overlapping source reads'
        try:
            self.reads.append(int(index))
            if index == self.fail:
                raise OSError('injected read failure')
            return super().read_raw(index)
        finally:
            self.active.release()


def frames(n=40):
    return np.arange(n * 6, dtype=np.uint16).reshape(n, 2, 3)


def test_prefetched_reads_match_direct_reads_in_any_access_pattern():
    data = frames()
    with Readahead(CountingSource(data), frames_ahead=4) as reader:
        reader.expect([3, 5, 7, 9, 11, 13])
        assert np.array_equal(reader.read_raw(3), data[3])
        assert np.array_equal(reader.read_raw(9), data[9])  # 5 and 7 are passed over and dropped.
        assert np.array_equal(reader.read_raw(5), data[5])  # No longer ahead: read directly.
        assert np.array_equal(reader.read_raw(20), data[20])  # Never announced.
        assert np.array_equal(reader.read_raw(11), data[11])
        reader.expect([1, 2])  # A new announcement replaces the old one.
        assert np.array_equal(reader.read_raw(13), data[13])
        assert np.array_equal(reader.read_raw(2), data[2])


def test_batches_follow_the_order_and_stop_on_cancellation():
    data = frames()
    source = CountingSource(data)
    reader = Readahead(source)
    order = np.array([0, 2, 3, 8, 9, 10, 30])
    got = list(reader.batches(order, 3))
    assert [list(i) for i, _ in got] == [[0, 2, 3], [8, 9, 10], [30]]
    assert all(np.array_equal(f, data[i]) for indices, batch in got for i, f in zip(indices, batch))
    assert sorted(source.reads) == list(order)
    seen = []
    for indices, batch in reader.batches(np.arange(40), 4, should_cancel=lambda: len(seen) >= 6):
        seen.extend(indices.tolist())
    assert seen == list(range(8))
    assert reader._worker is None


def test_read_failure_surfaces_at_the_frame_that_failed():
    data = frames()
    reader = Readahead(CountingSource(data, fail=6))
    reader.expect(range(10))
    for index in range(6):
        assert np.array_equal(reader.read_raw(index), data[index])
    with pytest.raises(OSError, match='injected'):
        reader.read_raw(6)
    reader.stop()

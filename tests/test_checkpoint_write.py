"""Large checkpoint cancellation stops writing before the GUI must kill it."""
import io
import json

import numpy as np
import pytest

from planetrecon.io.checkpoint_write import save_compressed


def test_checked_npz_preserves_all_values_and_scalar_metadata():
    arrays = {'metadata': json.dumps({'frames': 12}),
              'moments': np.random.default_rng(7).normal(size=(23, 3, 6, 6)),
              'strided': np.arange(90.).reshape(9, 10)[:, ::2]}
    stream = io.BytesIO()
    save_compressed(stream, should_cancel=lambda: False, **arrays)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as actual:
        assert set(actual.files) == set(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(actual[name], expected)
        assert json.loads(str(actual['metadata'])) == {'frames': 12}


def test_cancellation_interrupts_a_large_compressed_member():
    cancelled = False
    writes = []
    class Stream(io.BytesIO):
        def write(self, data):
            nonlocal cancelled
            writes.append(len(data))
            result = super().write(data)
            if len(data) >= 1024*1024:
                cancelled = True
            return result
    stream = Stream()
    data = np.random.default_rng(41).integers(0, 256, size=8*1024*1024, dtype=np.uint8)
    with pytest.raises(InterruptedError, match='compression/write'):
        save_compressed(stream, should_cancel=lambda: cancelled, moments=data)
    assert cancelled and max(writes) == 1024*1024
    assert stream.tell() < 2*1024*1024  # Stop within the member, not after all 8 MiB.
    assert not stream.closed  # The atomic-file owner retains cleanup responsibility.


@pytest.mark.parametrize('kind', ['application', 'fixed'])
def test_cancelled_write_preserves_published_state_and_cleans_temporary(tmp_path, monkeypatch, kind):
    from planetrecon.io import checkpoint_write
    from test_cfa_local_resume import inputs, advance
    from planetrecon.pipeline.cfa_checkpoint import FixedLocalRun, ARRAYS
    manifest, raw, maps = inputs(tmp_path)
    run = FixedLocalRun(manifest)
    advance(run, raw, maps, 1)
    path = tmp_path/'state.npz'
    stop = False
    if kind == 'fixed':
        run.model.should_cancel = lambda: stop
        save = lambda: run.save(path)
    else:
        from planetrecon import resume
        state = {'accum': run.model.sums, 'weight': run.model.weights,
                 'reference': raw[0], 'reference_index': 0,
                 'demosaic_accum': run.model.sums, 'demosaic_weight': run.model.weights,
                 'n_used': 1, 'n_rejected': 0, 'next_index': 1, 'local_cfa': {}}
        state.update({'local_'+name: getattr(run.model, name) for name in ('gram', 'noise', 'rhs', 'variance_sum')})
        save = lambda: resume.save(path, {'test': 'same input'}, state, should_cancel=lambda: stop)
    save()
    previous = path.read_bytes()
    original = checkpoint_write._CheckedWriter.write
    calls = 0
    def cancel_in_zip(self, data):
        nonlocal stop, calls
        calls += 1
        if calls == 4:
            stop = True
        return original(self, data)
    monkeypatch.setattr(checkpoint_write._CheckedWriter, 'write', cancel_in_zip)
    with pytest.raises(InterruptedError, match='compression/write'):
        save()
    assert calls >= 4 and path.read_bytes() == previous
    assert not list(tmp_path.glob('.state.npz-*.tmp'))
    if kind == 'fixed':
        restored = FixedLocalRun.load(path, manifest)
        for name in ARRAYS:
            np.testing.assert_array_equal(getattr(restored.model, name), getattr(run.model, name))

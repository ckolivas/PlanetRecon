"""Fixed-input local interpolation resumes exactly and never publishes partial frames."""
import hashlib
import json

import numpy as np
import pytest

import tools.cfa_local_resume as module
from tools.cfa_local_resume import FixedLocalRun, make_manifest, array_digest, map_digest, canonical


def inputs(tmp_path, color='RGGB'):
    shape = (10, 12)
    y, x = np.indices(shape, dtype=float)
    raw = [30+np.sin(x/2+i*.2)+np.cos(y/3-i*.1) for i in range(4)]
    maps = [(.2*i+.12*np.sin(y/3), -.15*i+.1*np.cos(x/4)) for i in range(4)]
    source = tmp_path/'input.ser'; source.write_bytes(b'fixed input identity')
    manifest = make_manifest(source, hashlib.sha256(source.read_bytes()).hexdigest(), shape, color,
                             [4, 7, 10, 20], [.8, 1., 1.1, 1.2],
                             [array_digest(v) for v in raw], [map_digest(v, shape) for v in maps], chunk_rows=7)
    return manifest, raw, maps


def advance(run, raw, maps, stop):
    while run.model.n_used < stop:
        i = run.model.n_used
        run.add(run.manifest['indices'][i], raw[i], maps[i])


@pytest.mark.parametrize('color', ['RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('cut', [0, 1, 3, 4])
def test_interruption_at_every_phase_matches_uninterrupted_bitwise(tmp_path, color, cut):
    manifest, raw, maps = inputs(tmp_path, color)
    full = FixedLocalRun(manifest); advance(full, raw, maps, 4)
    part = FixedLocalRun(manifest); advance(part, raw, maps, cut)
    checkpoint = tmp_path/'state.npz'; part.save(checkpoint)
    resumed = FixedLocalRun.load(checkpoint, manifest)
    advance(resumed, raw, maps, 4)
    assert resumed.model.n_used == 4
    for a, b in zip(full.finish()[:3], resumed.finish()[:3]):
        np.testing.assert_array_equal(a, b)
    for key in module.ARRAYS:
        np.testing.assert_array_equal(getattr(full.model, key), getattr(resumed.model, key))


@pytest.mark.parametrize('change', ['source', 'selection', 'quality', 'raw', 'map', 'region', 'chunk', 'neighbours'])
def test_resume_rejects_changed_input_or_execution_policy(tmp_path, change):
    manifest, raw, maps = inputs(tmp_path)
    run = FixedLocalRun(manifest); advance(run, raw, maps, 1)
    path = tmp_path/'state.npz'; run.save(path)
    if change == 'source': manifest['source_sha256'] = '1'*64
    elif change == 'selection': manifest['indices'][-1] += 1
    elif change == 'quality': manifest['qualities'][-1] += .1
    elif change in ('raw', 'map'): manifest[change+'_digests'][-1] = '1'*64
    elif change == 'region': manifest['options']['region'][0][1] -= 1
    elif change == 'chunk': manifest['options']['chunk_rows'] += 1
    elif change == 'neighbours': manifest['options']['max_neighbour_entries'] += 1
    with pytest.raises(ValueError, match='mismatch|array header'):
        FixedLocalRun.load(path, manifest)


@pytest.mark.parametrize('wrong', ['raw', 'map', 'index'])
def test_wrong_next_frame_never_changes_state(tmp_path, wrong):
    manifest, raw, maps = inputs(tmp_path)
    run = FixedLocalRun(manifest); advance(run, raw, maps, 1)
    before = {name: getattr(run.model, name).copy() for name in module.ARRAYS}
    args = [7, raw[1], maps[1]]
    if wrong == 'raw': args[1] = raw[1]+.01
    if wrong == 'map': args[2] = (maps[1][0]+.01, maps[1][1])
    if wrong == 'index': args[0] = 4
    with pytest.raises(ValueError): run.add(*args)
    assert run.model.n_used == 1
    for name in module.ARRAYS: np.testing.assert_array_equal(before[name], getattr(run.model, name))
    with pytest.raises(ValueError, match='before all'): run.finish()


def test_cancelled_frame_can_resume_without_double_counting(tmp_path):
    manifest, raw, maps = inputs(tmp_path)
    run = FixedLocalRun(manifest); advance(run, raw, maps, 1)
    path = tmp_path/'state.npz'; run.save(path)
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 16
    run.model.should_cancel = cancelled
    with pytest.raises(InterruptedError): run.add(7, raw[1], maps[1])
    run.model.should_cancel = None
    run.save(path)
    resumed = FixedLocalRun.load(path, manifest); advance(resumed, raw, maps, 4)
    full = FixedLocalRun(manifest); advance(full, raw, maps, 4)
    for a, b in zip(full.finish()[:3], resumed.finish()[:3]): np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('failure', ['write', 'cancel', 'replace'])
def test_failed_checkpoint_publication_preserves_previous_file(tmp_path, monkeypatch, failure):
    manifest, raw, maps = inputs(tmp_path)
    run = FixedLocalRun(manifest); advance(run, raw, maps, 1)
    path = tmp_path/'state.npz'; run.save(path); original = path.read_bytes()
    advance(run, raw, maps, 2)
    if failure == 'replace':
        def fail(*args): raise OSError('replace failed')
        monkeypatch.setattr(module.os, 'replace', fail)
    else:
        save = module.np.savez_compressed
        def interrupted(stream, **kwargs):
            save(stream, **kwargs)
            if failure == 'write': raise OSError('write failed')
            run.model.should_cancel = lambda: True
        monkeypatch.setattr(module.np, 'savez_compressed', interrupted)
    with pytest.raises((OSError, InterruptedError)): run.save(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('.state.npz-*.tmp'))
    assert FixedLocalRun.load(path, manifest).model.n_used == 1


@pytest.mark.parametrize('corrupt', ['array', 'count', 'history'])
def test_corrupt_checkpoint_is_not_accepted(tmp_path, corrupt):
    manifest, raw, maps = inputs(tmp_path)
    run = FixedLocalRun(manifest); advance(run, raw, maps, 1)
    path = tmp_path/'state.npz'; run.save(path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in module.ARRAYS}
        metadata = json.loads(str(stored['metadata']))
    if corrupt == 'array': arrays['sums'][0, 0, 0] += 1
    elif corrupt == 'count': metadata['n_used'] = 99
    else: metadata['execution_history'] = ['cuda']
    if corrupt != 'array':
        metadata.pop('metadata_sha256')
        metadata['metadata_sha256'] = hashlib.sha256(canonical(metadata).encode()).hexdigest()
    np.savez_compressed(path, metadata=canonical(metadata), **arrays)
    with pytest.raises(ValueError): FixedLocalRun.load(path, manifest)


def test_checkpoint_cannot_replace_source_or_unrelated_npz(tmp_path):
    manifest, raw, maps = inputs(tmp_path)
    run = FixedLocalRun(manifest); advance(run, raw, maps, 1)
    with pytest.raises(ValueError): run.save(manifest['source_path'])
    unrelated = tmp_path/'user.npz'; np.savez(unrelated, data=np.ones(3))
    before = unrelated.read_bytes()
    with pytest.raises((ValueError, KeyError)): run.save(unrelated)
    assert unrelated.read_bytes() == before
    link = tmp_path/'source.npz'; link.symlink_to(manifest['source_path'])
    with pytest.raises(ValueError): run.save(link)


def test_unsupported_operator_or_runtime_cannot_masquerade_as_current_policy(tmp_path):
    manifest, _, _ = inputs(tmp_path)
    manifest['operator']['radius'] = 5
    with pytest.raises(ValueError, match='manifest changed'): FixedLocalRun(manifest)
    manifest, _, _ = inputs(tmp_path)
    manifest['execution']['numpy'] = 'different runtime'
    with pytest.raises(ValueError, match='manifest changed'): FixedLocalRun(manifest)


def test_huge_corrupt_array_header_is_refused_before_payload_loading(tmp_path, monkeypatch):
    import io
    import zipfile
    manifest, raw, maps = inputs(tmp_path)
    run = FixedLocalRun(manifest); advance(run, raw, maps, 1)
    path = tmp_path/'state.npz'; run.save(path)
    with zipfile.ZipFile(path) as source:
        members = {name: source.read(name) for name in source.namelist()}
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'descr': '<f8', 'fortran_order': False, 'shape': (10**12,)})
    members['gram.npy'] = header.getvalue()
    with zipfile.ZipFile(path, 'w') as target:
        for name, content in members.items(): target.writestr(name, content)
    def forbidden(*args, **kwargs): pytest.fail('decompressed arrays before verifying headers')
    monkeypatch.setattr(module.np, 'load', forbidden)
    with pytest.raises(ValueError, match='array header'): FixedLocalRun.load(path, manifest)

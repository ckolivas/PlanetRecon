"""W13 format read-back, mapping, persistence and interrupted-publication tests."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import numpy as np
import pytest
import tifffile

import planetrecon.export as exporter
from planetrecon.export import ExportConfig, export_result, parse_tiff_description
from planetrecon.result import ReconstructionResult, load_snapshot, save_snapshot


def result(image, **changes):
    image = np.asarray(image, dtype=np.float64)
    r = ReconstructionResult(image, np.ones_like(image), np.ones_like(image, dtype=bool),
                             'adu', 'mono' if image.ndim == 2 else 'RGB', 'cpu', 'float64',
                             'final', False, reference_epoch='1.25', n_used=7,
                             provenance={'config': {'value': None}, 'source': {'path': '月.ser'}},
                             layer_coverage={'ring': np.ones(image.shape[:2])})
    return replace(r, **changes)


def png_chunks(path):
    raw = path.read_bytes()
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'
    offset, chunks = 8, {}
    while offset < len(raw):
        n = struct.unpack_from('>I', raw, offset)[0]
        kind = raw[offset+4:offset+8]
        payload = raw[offset+8:offset+8+n]
        assert zlib.crc32(kind + payload) == struct.unpack_from('>I', raw, offset+8+n)[0]
        chunks.setdefault(kind, []).append(payload)
        offset += n + 12
    assert offset == len(raw) and chunks[b'IEND'] == [b'']
    return chunks


def tiff_ifd_count(path):
    raw = path.read_bytes()
    order = '<' if raw[:2] == b'II' else '>'
    offset = struct.unpack_from(order+'I', raw, 4)[0]
    pages = 0
    while offset:
        pages += 1
        count = struct.unpack_from(order+'H', raw, offset)[0]
        offset = struct.unpack_from(order+'I', raw, offset + 2 + 12*count)[0]
    return pages


def independent_tiff(path):
    """Read classic uncompressed TIFF tags/strips without using the encoder library."""
    raw = path.read_bytes()
    order = '<' if raw[:2] == b'II' else '>'
    assert struct.unpack_from(order+'H', raw, 2)[0] == 42
    offset = struct.unpack_from(order+'I', raw, 4)[0]
    count = struct.unpack_from(order+'H', raw, offset)[0]
    tags = {}
    for i in range(count):
        entry = offset + 2 + 12*i
        tag, typ, n = struct.unpack_from(order+'HHI', raw, entry)
        if typ not in (3, 4):
            continue
        code, size = ('H', 2) if typ == 3 else ('I', 4)
        pos = entry+8 if size*n <= 4 else struct.unpack_from(order+'I', raw, entry+8)[0]
        tags[tag] = struct.unpack_from(order+code*n, raw, pos)
    assert tags[259] == (1,)  # no compression
    bits, samples = tags[258][0], tags.get(277, (1,))[0]
    fmt = tags.get(339, (1,))[0]
    assert tags[274] == (1,)  # top-left
    ids = []
    for i in range(count):
        ids.append(struct.unpack_from(order+'H', raw, offset + 2 + 12*i)[0])
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    shape = (tags[257][0], tags[256][0]) + ((samples,) if samples > 1 else ())
    data = b''.join(raw[start:start+length] for start, length in zip(tags[273], tags[279]))
    image = np.frombuffer(data, dtype=order+('f' if fmt == 3 else 'u')+str(bits//8)).reshape(shape)
    return image, tags


def collapse_replicated_rgb(array):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 3 and np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2]):
        return array[..., 0]
    return array


@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('encoding', ['png16', 'tiff16', 'tiff32', 'tiff32_raw'])
def test_encodings_preserve_values_and_metadata(tmp_path, rgb, encoding):
    a = np.array([[-4., 0., 1., 254., 255., 256., 257., 65534., 65535., 70000.]])
    if rgb:
        a = np.stack([a, a[:, ::-1], a / 2], axis=-1)
    r = result(a)
    p = tmp_path / ('科学.png' if encoding == 'png16' else '科学.tiff')
    cfg = ExportConfig(encoding, *([] if encoding in ('tiff32', 'tiff32_raw') else [0., 65535.]))
    report = export_result(r, p, cfg)
    expected = ((a / 100100).astype(np.float32) if encoding == 'tiff32' else
                a.astype(np.float32) if encoding == 'tiff32_raw' else
                np.floor(np.clip(a, 0, 65535)+.5).astype(np.uint16))
    meta = json.loads(report.sidecar.read_text())
    assert meta['image_sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert meta['result']['reference_epoch'] == '1.25'
    assert meta['result']['provenance'] == r.provenance
    assert meta['rendering'] == 'scientific-linear'
    assert meta['resampling'] == 'none'
    if encoding == 'png16':
        chunks = png_chunks(p)
        assert struct.unpack('>IIBBBBB', chunks[b'IHDR'][0]) == (10, 1, 16, 2 if rgb else 0, 0, 0, 0)
        assert struct.unpack('>I', chunks[b'gAMA'][0]) == (100000,)
        embedded = json.loads(chunks[b'iTXt'][0].split(b'\0', 5)[-1])
        # Qt/libpng independently decodes all 16 bits, including truecolour RGB.
        QtGui = pytest.importorskip('PySide6.QtGui')
        q = QtGui.QImage(str(p)).convertToFormat(QtGui.QImage.Format.Format_RGBA64)
        assert not q.isNull()
        rgba = np.frombuffer(q.constBits(), dtype=np.uint16).reshape(1, 10, 4)
        np.testing.assert_array_equal(rgba[..., :3] if rgb else rgba[..., 0], expected)
    else:
        actual, tags = independent_tiff(p)
        assert tiff_ifd_count(p) == 1
        assert tags[258] == ((32 if encoding in ('tiff32', 'tiff32_raw') else 16),) * (3 if rgb else 1)
        assert tags.get(339, (1,)) == ((3,) * (3 if rgb else 1) if encoding in ('tiff32', 'tiff32_raw') else (1,))
        np.testing.assert_array_equal(actual, expected)
        with tifffile.TiffFile(p) as tf:
            embedded = parse_tiff_description(tf.pages[0].description)
            assert not tf.pages[0].is_shaped
            assert len(tf.pages) == 1
            assert tf.pages[0].description.startswith('PlanetRecon\n')
        if encoding == 'tiff16':
            Image = pytest.importorskip('PIL.Image')
            with Image.open(p) as im:
                im.load()
                assert im.size == (10, 1)
            QtGui = pytest.importorskip('PySide6.QtGui')
            q = QtGui.QImage(str(p)).convertToFormat(QtGui.QImage.Format.Format_RGBA64)
            assert not q.isNull()
            rgba = np.frombuffer(q.constBits(), dtype=np.uint16).reshape(1, 10, 4)
            np.testing.assert_array_equal(rgba[..., :3] if rgb else rgba[..., 0], expected)
    assert embedded == {k:v for k,v in meta.items() if k != 'image_sha256'}
    assert report.coverage_path is not None and report.coverage_path.exists()
    with tifffile.TiffFile(report.coverage_path) as tf:
        np.testing.assert_array_equal(tf.pages[0].asarray(), collapse_replicated_rgb(r.validity))
        np.testing.assert_array_equal(tf.pages[1].asarray(), collapse_replicated_rgb(r.coverage))
        assert json.loads(tf.pages[2].description)['layer'] == 'ring'
        np.testing.assert_array_equal(tf.pages[2].asarray(), r.layer_coverage['ring'])
    np.testing.assert_array_equal(r.image, a)


@pytest.mark.parametrize('encoding', ['png16', 'tiff16', 'tiff32', 'tiff32_raw'])
def test_invalid_and_nonfinite_samples_have_explicit_masks(tmp_path, encoding):
    r = result([[[np.nan, np.inf, -np.inf], [1, 2, 3]], [[4, 5, 6], [-1, 20, 10]]])
    r.validity[1, 0, 0] = False
    r.coverage[1, 0, 1] = 0
    cfg = ExportConfig(encoding, *([] if encoding in ('tiff32', 'tiff32_raw') else [0, 10]))
    report = export_result(r, tmp_path / ('a.png' if encoding == 'png16' else 'a.tif'), cfg)
    counts = report.metadata['counts']
    assert counts['invalid_samples'] == 5 and counts['invalid_pixels'] == 2
    assert counts['nonfinite_samples'] == 3
    if encoding not in ('tiff32', 'tiff32_raw'):
        assert counts['clipped_low_samples'] == counts['clipped_high_samples'] == counts['clipped_pixels'] == 1
    else:
        pixels, _ = independent_tiff(report.path)
        assert np.isnan(pixels[0, 0]).all() and np.isnan(pixels[1, 0, :2]).all()
        scale = 28.6 if encoding == 'tiff32' else 1.
        np.testing.assert_allclose(pixels[1, 1, :2]*scale, [-1., 20.], rtol=1e-7)


def test_tiff_files_are_single_page_and_not_tifffile_shaped(tmp_path, caplog):
    import logging
    r = result([[1., 2., 3.]])
    caplog.set_level(logging.ERROR)
    for encoding, extra in (('tiff16', {'black': 0, 'white': 65535}), ('tiff32', {})):
        p = tmp_path / f'{encoding}.tif'
        export_result(r, p, ExportConfig(encoding, **extra))
        with tifffile.TiffFile(p) as tf:
            assert len(tf.pages) == 1
            assert not tf.pages[0].is_shaped
            assert tf.pages[0].asarray().shape == (1, 3)
        assert tiff_ifd_count(p) == 1
        with tifffile.TiffFile(p) as tf:
            assert tf.pages[0].description.startswith('PlanetRecon\n')
            parse_tiff_description(tf.pages[0].description)
    assert 'corrupted file' not in caplog.text
    assert 'invalid shaped series' not in caplog.text
    Image = pytest.importorskip('PIL.Image')
    with Image.open(tmp_path / 'tiff16.tif') as im:
        im.load()
        np.testing.assert_array_equal(np.array(im), [[1, 2, 3]])
    nested = result([[4., 5.]], provenance={'calibration': {'bias': {'shape': [8, 8], 'dtype': '<f8'}}})
    export_result(nested, tmp_path / 'nested.tif')
    with tifffile.TiffFile(tmp_path / 'nested.tif') as tf:
        assert '"shape":' in tf.pages[0].description
        assert tf.pages[0].description.startswith('PlanetRecon\n')
        assert not tf.pages[0].is_shaped
        assert parse_tiff_description(tf.pages[0].description)['result']['provenance']['calibration']['bias']['shape'] == [8, 8]


def test_mapping_rounding_display_gamma_and_spatial_mask(tmp_path):
    r = result(np.array([[[10, 10+20*.25, 30], [10+10/65535, 10, 30]]]),
               coverage=np.ones((1, 2)), validity=np.ones((1, 2), bool))
    report = export_result(r, tmp_path / 'gamma.tif', ExportConfig('tiff16', 10, 30, 2))
    image, _ = independent_tiff(report.path)
    assert image[0, 0].tolist() == [0, 32768, 65535]
    assert report.metadata['rendering'] == 'display-rendered'
    assert report.metadata['mapping']['shared_across_channels'] is True
    linear = export_result(result([[.5, 1.5, 2.5]]), tmp_path / 'ties.tif', ExportConfig('tiff16', 0, 65535))
    np.testing.assert_array_equal(independent_tiff(linear.path)[0], [[1, 2, 3]])


@pytest.mark.parametrize('kwargs', [dict(encoding='bad'), dict(encoding='png16'),
    dict(encoding='tiff16', black=1, white=1), dict(encoding='png16', black=0, white=np.inf),
    dict(encoding='tiff32_raw', black=0, white=1), dict(encoding='tiff32', black=0), dict(display_gamma=2),
    dict(encoding='png16', black=0, white=1, display_gamma=np.nan)])
def test_bad_export_options(kwargs):
    with pytest.raises(ValueError):
        ExportConfig(**kwargs)


def test_bad_shapes_and_float_overflow_never_publish(tmp_path):
    p = tmp_path / 'x.tif'
    for r in (result([[1e39]]), result([[1.]], channel_order='BGR'),
              result([[1.]], validity=np.ones((1,1))), result([[1.]], coverage=np.array([[-1.]]))):
        with pytest.raises(ValueError):
            export_result(r, p, ExportConfig('tiff32_raw'))
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('failure', ['encoder', 'mask', 'sidecar', 'commit'])
def test_failed_replacement_keeps_old_image_companions_and_result(tmp_path, monkeypatch, failure):
    p = tmp_path / 'x.png'
    cfg = ExportConfig('png16', 0, 10)
    r = result([[2., 8.]])
    old = export_result(r, p, cfg)
    before = {f.name:f.read_bytes() for f in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        export_result(r, p, cfg)
    original_publish = exporter._publish
    def publish(staged, destination, overwrite=False):
        if ((failure == 'commit' and destination == p) or
            (failure == 'mask' and destination.suffix == '.tif') or
            (failure == 'sidecar' and destination.suffix == '.json')):
            raise OSError('simulated disk full')
        original_publish(staged, destination, overwrite)
    def broken_encoder(stream, *args):
        stream.write(b'partial')
        raise OSError('simulated disk full')
    monkeypatch.setattr(exporter, '_publish', publish)
    if failure == 'encoder':
        monkeypatch.setattr(exporter, '_write_png', broken_encoder)
    with pytest.raises(OSError, match='disk full'):
        export_result(r, p, cfg, overwrite=True)
    assert {f.name:f.read_bytes() for f in tmp_path.iterdir()} == before
    np.testing.assert_array_equal(r.image, [[2., 8.]])
    assert old.sidecar.exists()


def test_racing_destination_is_not_overwritten(tmp_path, monkeypatch):
    p = tmp_path / 'x.tif'
    original = exporter._publish
    def publish(staged, destination, overwrite=False):
        if destination == p:
            p.write_bytes(b'another writer')
        return original(staged, destination, overwrite)
    monkeypatch.setattr(exporter, '_publish', publish)
    with pytest.raises(FileExistsError):
        export_result(result([[1.]]), p)
    assert p.read_bytes() == b'another writer'
    assert list(tmp_path.iterdir()) == [p]


def test_process_interruption_keeps_old_generation_and_allows_retry(tmp_path):
    p = tmp_path / 'x.tif'
    old = export_result(result([[1.]]), p)
    before = p.read_bytes()
    script = '''
import os, sys
from pathlib import Path
import planetrecon.export as ex
from planetrecon.result import ReconstructionResult
import numpy as np
p = Path(sys.argv[1])
original = ex._publish
def stop(staged, destination, overwrite=False):
    if destination == p:
        os._exit(17)
    original(staged, destination, overwrite)
ex._publish = stop
r = ReconstructionResult(np.ones((1,1))*2, np.ones((1,1)), np.ones((1,1),bool), 'adu', 'mono', 'cpu', 'float64', 'final', False)
ex.export_result(r, p, overwrite=True)
'''
    proc = subprocess.run([sys.executable, '-c', script, str(p)], timeout=15)
    assert proc.returncode == 17 and p.read_bytes() == before and old.sidecar.exists()
    new = export_result(result([[3.]]), p, overwrite=True)
    assert new.sidecar != old.sidecar
    assert independent_tiff(p)[0][0,0] * new.metadata['mapping']['white'] == pytest.approx(3.)


def test_full_resolution_intermediate_export_preserves_processing(tmp_path):
    from planetrecon.calibration import Calibration
    from planetrecon.io.source import ArraySource
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.reconstruction import ReconstructionConfig
    from planetrecon.jobs import _write_checkpoint
    frame = 10 + np.random.default_rng(7).random((514, 8))
    source = ArraySource(np.stack([frame]*3), bit_depth=32)
    cfg = ReconstructionConfig(frame_preselection=False, device='cpu', threads=2, batch_frames=1)
    snapshots = []
    def callback(r, info):
        if snapshots:
            return
        save_snapshot(tmp_path/'live.npz', r)
        _write_checkpoint(str(tmp_path), 'worker', cfg, r)
        snap = load_snapshot(tmp_path/'worker.npz')
        report = export_result(snap, tmp_path/'live.tif')
        snapshots.append(report)
        assert report.metadata['result']['incomplete'] and r.n_used == 1
        assert snap.image.shape == (514, 8)
        assert snap.provenance['config'] == cfg.to_dict()
        assert snap.provenance['calibration']['gain_e_per_adu'] == 2
    final = stack_source(source, cfg, calibration=Calibration(gain_e_per_adu=2), on_event=callback)
    assert final.n_used == 3 and not final.incomplete
    assert load_snapshot(tmp_path/'live.npz').incomplete
    assert independent_tiff(tmp_path/'live.tif')[0].shape == (514, 8)
    np.testing.assert_allclose(final.image, 2*frame)


def test_cli_stack_export_and_checkpoint(tmp_path):
    from planetrecon.cli import main
    from planetrecon.io.ser import write_ser
    frame = (10 + np.random.default_rng(1).random((9, 12))*30).astype(np.uint16)
    path = tmp_path/'fixture.ser'
    write_ser(path, np.stack([frame]*2))
    args = ['--threads', '2', 'stack', '--no-frame-preselection', '--path', str(path), '--out', str(tmp_path/'out'), '--device', 'cpu',
            '--export', 'png16', '--black', '0', '--white', '65535', '--checkpoint', str(tmp_path/'live.npz')]
    assert main(args) == 0
    loaded = load_snapshot(tmp_path/'out/stack.npz')
    assert loaded.n_used == 2 and loaded.provenance['input_identity']['sample_sha256']
    assert loaded.provenance['input_identity']['full_file_checksum'] is False
    assert not load_snapshot(tmp_path/'live.npz').incomplete
    assert main(['export', '--path', str(tmp_path/'live.npz'), '--out', str(tmp_path/'float.tif')]) == 0
    np.testing.assert_allclose(independent_tiff(tmp_path/'float.tif')[0], loaded.image / (1.43*loaded.image.max()), rtol=1e-7)
    assert main(['export', '--path', str(tmp_path/'live.npz'), '--out', str(tmp_path/'raw.tif'), '--encoding', 'tiff32_raw']) == 0
    np.testing.assert_array_equal(independent_tiff(tmp_path/'raw.tif')[0], loaded.image.astype(np.float32))
    with pytest.raises(SystemExit):
        main(args)
    assert main(args + ['--overwrite']) == 0


def test_atomic_snapshot_failure_and_legacy_rejection(tmp_path, monkeypatch):
    p = tmp_path/'snapshot.npz'
    save_snapshot(p, result([[2.]]))
    before = p.read_bytes()
    def broken(*args, **kwargs):
        raise OSError('no space')
    monkeypatch.setattr(np, 'savez_compressed', broken)
    with pytest.raises(OSError):
        save_snapshot(p, result([[3.]]))
    assert p.read_bytes() == before and list(tmp_path.iterdir()) == [p]
    np.savez(tmp_path/'legacy.npz', image=np.ones((1, 1)))
    with pytest.raises(ValueError, match='metadata'):
        load_snapshot(tmp_path/'legacy.npz')


def test_preview_stride_is_tracked_and_export_requires_full_resolution(tmp_path):
    r = result(np.ones((1025, 8)))
    preview = r.copy_preview()
    assert preview.spatial_stride == 3
    preview.provenance['config']['value'] = 'edited'
    assert r.provenance['config']['value'] is None
    assert preview.copy_preview(max_side=128).spatial_stride == 9
    with pytest.raises(ValueError, match='full-resolution'):
        export_result(preview, tmp_path/'preview.tif')


def test_checkpoint_cannot_replace_capture_even_through_hardlink(tmp_path):
    import os
    from planetrecon.cli import main
    capture = tmp_path/'input.ser'
    capture.write_bytes(b'protected recording')
    checkpoint = tmp_path/'snapshot.npz'
    os.link(capture, checkpoint)
    for p in (capture, checkpoint):
        with pytest.raises(SystemExit):
            main(['stack', '--no-frame-preselection', '--path', str(capture), '--checkpoint', str(p)])
    assert capture.read_bytes() == b'protected recording'

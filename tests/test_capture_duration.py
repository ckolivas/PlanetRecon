from dataclasses import replace
import numpy as np
import pytest

from planetrecon.geometry.pose import capture_timing
from planetrecon.io.ser import SERSource, write_ser
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.preprocess_cache import preprocess_source, load_cache
from planetrecon.reconstruction import ReconstructionConfig


def test_ser_duration_uses_exact_trailer_span_with_irregular_intervals(tmp_path):
    base = 638_900_000_000_000_000
    ticks = np.array([base, base+1, base+102, base+1_234_567], dtype=np.int64)
    path = write_ser(tmp_path/'timed.ser', np.zeros((4, 16, 16), dtype='u2'), timestamps=ticks)
    with SERSource(path) as source:
        timing = capture_timing(source, cadence_s=99.)
        assert timing['duration_s'] == pytest.approx(.1234567, abs=1e-14)
        assert timing['minimum_interval_s'] == pytest.approx(1e-7, abs=1e-14)
        assert timing['origin'] == 'measured'
        assert source.metadata().as_dict()['extras']['capture_timing']['value'] == timing
        cfg = ReconstructionConfig(device='cpu', threads=2)
        selected = preprocess_source(source, cfg)
        assert selected.summary['timing'] == timing
        _, report = load_cache(source, replace(cfg, cadence_s=12.))
        assert report['timing'] == timing


@pytest.mark.parametrize('timestamps', [[10., 10.], [10., 9.]])
def test_invalid_timestamps_do_not_create_a_duration(timestamps):
    source = ArraySource(np.zeros((2, 16, 16)), timestamps=np.array(timestamps))
    timing = capture_timing(source, cadence_s=1.)
    assert timing['status'] == 'invalid' and timing['duration_s'] is None


def test_missing_timestamps_require_explicit_cadence_and_single_frame_has_zero_span(tmp_path):
    path = write_ser(tmp_path/'untimed.ser', np.zeros((4, 16, 16), dtype='u2'), datetime_utc=638_900_000_000_000_000)
    with SERSource(path) as source:
        assert capture_timing(source)['duration_s'] is None
        assert capture_timing(source, cadence_s=.1)['duration_s'] == pytest.approx(.3)
        cfg = ReconstructionConfig(device='cpu', threads=2, cadence_s=.1)
        preprocess_source(source, cfg)
        _, report = load_cache(source, replace(cfg, cadence_s=.2))
        assert report['timing']['duration_s'] == pytest.approx(.6)
    one = ArraySource(np.zeros((1, 16, 16)), timestamps=np.array([100.]))
    assert capture_timing(one)['duration_s'] == 0.


def test_gui_shows_measured_capture_duration():
    pytest.importorskip('PySide6')
    from planetrecon.gui.app import MainWindow, create_app
    app = create_app(['timing-test'])
    window = MainWindow()
    try:
        source = ArraySource(np.zeros((4, 16, 16)), timestamps=np.array([0., .1, .2, .32]))
        window._set_input({'source_metadata': source.metadata().as_dict(),
                           'capture_timing': capture_timing(source),
                           'input_image': source.read_raw(0), 'input_view': 'mono'})
        assert 'duration: 0.32 s (measured)' in window.source_label.text()
    finally:
        window._shutdown()
        window.window.close()
        app.processEvents()

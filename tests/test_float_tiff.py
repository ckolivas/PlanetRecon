"""Editor-compatible float exports retain reversible, unclipped linear data."""
import struct

import numpy as np
import pytest
import tifffile

from planetrecon.export import ExportConfig, export_result
from test_w13 import result


@pytest.mark.parametrize('rgb', [False, True])
def test_adu_float_is_displayable_and_inverts_to_original_values(tmp_path, rgb):
    data = np.array([[-20., 0., 300., 1200.]])
    if rgb:
        data = data[..., None] * [1., .7, .3]
    r = result(data)
    saved = export_result(r, tmp_path/'preview.tif')
    with tifffile.TiffFile(saved.path) as file:
        pixels = file.asarray()
        profile = file.pages[0].tags['InterColorProfile'].value
    assert pixels.dtype == np.float32
    assert pixels.max() == pytest.approx(1/1.43) and pixels.min() < 0
    assert saved.metadata['mapping']['white'] == 1716.
    np.testing.assert_allclose(pixels*1716, data, rtol=1e-7, atol=2e-5)
    assert saved.metadata['counts']['clipped_pixels'] == 0
    assert saved.metadata['mapping']['input_units'] == 'adu'
    assert saved.metadata['rendering'] == 'preview-mapped'
    assert saved.metadata['mapping']['gamma'] == 1.
    assert saved.metadata['transfer']['function'] == 'sRGB'
    assert saved.metadata['transfer']['gamma'] is None
    assert profile[16:20] == (b'RGB ' if rgb else b'GRAY')
    # Independently evaluate the ICC decode curve, then encode to a display.
    # The old linear profile turned 0.25 into 0.537, bleaching the midtones.
    tags = {name: profile[offset:offset+size] for name,offset,size in
            (struct.unpack_from('>4sII',profile,132+12*i)
             for i in range(struct.unpack_from('>I',profile,128)[0]))}
    for name in ((b'rTRC',b'gTRC',b'bTRC') if rgb else (b'kTRC',)):
        curve = tags[name]
        assert curve[:4] == b'para' and struct.unpack_from('>H',curve,8)[0] == 3
        g, a, b, c, d = np.array(struct.unpack_from('>5i',curve,12)) / 65536
        codes = np.array([0., .02, .04, .25, .5, .75, 1.])
        light = np.where(codes >= d, (a*codes+b)**g, c*codes)
        display = np.where(light <= .0031308, 12.92*light, 1.055*light**(1/2.4)-.055)
        np.testing.assert_allclose(display, codes, atol=3e-5, rtol=0)
    if rgb:
        from PySide6.QtGui import QColorSpace
        colour = QColorSpace.fromIccProfile(profile)
        assert colour.isValid() and colour.transferFunction() == QColorSpace.TransferFunction.SRgb
    np.testing.assert_array_equal(r.image, data)


@pytest.mark.parametrize('gain', [None, 2.])
def test_automatic_white_caps_at_detector_range_in_result_units(tmp_path, gain):
    r = result([[0., 200.]], provenance={'source':{'units':'adu','bit_depth':8},
                                       'config':{'gain_e_per_adu':gain}})
    if gain is not None:
        r.units = 'e-'
        r.image *= gain
    saved = export_result(r, tmp_path/'capped.tif')
    assert saved.metadata['mapping']['white'] == 255*(gain or 1.)
    assert tifffile.imread(saved.path).max() == pytest.approx(200/255)


def test_manual_float_levels_preserve_hdr_negative_values_and_masks(tmp_path):
    r = result([[-10., 20., 200., np.inf]])
    saved = export_result(r, tmp_path/'hdr.tif', ExportConfig('tiff32', 10., 110.))
    actual = tifffile.imread(saved.path)
    np.testing.assert_allclose(actual[0,:3], [-.2,.1,1.9])
    assert np.isnan(actual[0,3])
    assert saved.metadata['mapping']['clipping'] == 'none'
    assert saved.metadata['counts']['clipped_pixels'] == 0
    assert saved.metadata['counts']['invalid_samples'] == 1
    np.testing.assert_allclose(actual[0,:3]*100+10, r.image[0,:3], atol=1e-5)


def test_raw_float_option_retains_original_units_and_no_editing_profile(tmp_path):
    r = result([[-2., 350., 90000.]])
    saved = export_result(r, tmp_path/'raw.tif', ExportConfig('tiff32_raw'))
    with tifffile.TiffFile(saved.path) as file:
        np.testing.assert_array_equal(file.asarray(), r.image.astype(np.float32))
        assert 'InterColorProfile' not in file.pages[0].tags
    assert saved.metadata['mapping'] is None

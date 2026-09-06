import numpy as np
import pytest

from planetrecon.detector import cfa_labels, nearest_debayer_preview


@pytest.mark.parametrize('pattern',['RGGB','GRBG','GBRG','BGGR'])
@pytest.mark.parametrize('origin',[(0,0),(1,0),(0,1),(1,1)])
def test_display_uses_only_nearest_measured_channel_samples(pattern,origin):
    # Unique samples catch accidental averaging, wrong channel, and wrong phase.
    raw=np.arange(7*9,dtype='u2').reshape(7,9)+100
    labels=cfa_labels(7,9,pattern,origin)
    rgb=nearest_debayer_preview(raw,pattern,origin)
    assert rgb.dtype==raw.dtype
    for channel,index in zip('RGB',range(3)):
        sites=np.argwhere(labels==channel)
        for y in range(7):
            for x in range(9):
                d=np.sum((sites-[y,x])**2,axis=1)
                nearest=sites[d==d.min()]
                assert rgb[y,x,index] in raw[nearest[:,0],nearest[:,1]]
    for stride in (2,3,4):
        np.testing.assert_array_equal(nearest_debayer_preview(raw,pattern,origin,stride),rgb[::stride,::stride])


def test_worker_inspection_delivers_rgb_bayer_preview(tmp_path):
    import time
    from planetrecon.io.ser import write_ser,COLOR_RGGB
    from planetrecon.jobs import start_stack_job
    from planetrecon.reconstruction import ReconstructionConfig
    raw=np.arange(9*11,dtype='u2').reshape(9,11)+100
    path=write_ser(tmp_path/'in.ser',raw[None],color_id=COLOR_RGGB)
    handle=start_stack_job(path,ReconstructionConfig(device='cpu',threads=2),inspect_only=True)
    events=[]
    try:
        end=time.monotonic()+10
        while handle.state not in ('completed','failed') and time.monotonic()<end:
            events.extend(handle.poll(.05))
        assert handle.state=='completed'
        payload=next(e.payload for e in events if e.kind=='completed')
        assert payload['input_view']=='nearest-neighbour Bayer RGB'
        np.testing.assert_array_equal(payload['input_image'],nearest_debayer_preview(raw,'RGGB'))
    finally:handle.close()


@pytest.mark.parametrize('pattern',['RGGB','GRBG','GBRG','BGGR'])
@pytest.mark.parametrize('origin',[(0,0),(1,0),(0,1),(1,1)])
def test_bayer_result_display_samples_support_before_decimation(pattern, origin):
    from planetrecon.gui.preview import display_result_preview
    from planetrecon.result import ReconstructionResult
    labels = cfa_labels(7,9,pattern,origin)
    valid = np.stack([labels == channel for channel in 'RGB'],axis=2)
    image = np.arange(7*9*3,dtype=float).reshape(7,9,3)+100
    coverage = np.where(valid,image/10,0)
    r = ReconstructionResult(image,coverage,valid,'adu','RGB','cpu','float64',
                             'final',False,provenance={'color_mode':pattern})
    before = [a.copy() for a in (image,coverage,valid)]
    full = display_result_preview(r,max_side=20)
    assert full.validity.all()
    for channel in range(3):
        sites = np.argwhere(valid[...,channel])
        for y in range(7):
            for x in range(9):
                dist = np.sum((sites-[y,x])**2,axis=1)
                nearest = sites[dist == dist.min()]
                assert full.image[y,x,channel] in image[nearest[:,0],nearest[:,1],channel]
                assert full.coverage[y,x,channel] == full.image[y,x,channel]/10
    for max_side in (5,3):
        small = display_result_preview(r,max_side=max_side)
        step = small.spatial_stride
        assert step in (2,3)
        np.testing.assert_array_equal(small.image,full.image[::step,::step])
        np.testing.assert_array_equal(small.validity,full.validity[::step,::step])
    for current, original in zip((image,coverage,valid),before):
        np.testing.assert_array_equal(current,original)
    assert 'display_sampling' not in r.provenance


def test_bayer_display_uses_shifted_support_and_preserves_large_holes():
    from planetrecon.gui.preview import display_result_preview
    from planetrecon.result import ReconstructionResult
    # Jupiter's registered result: R/B on one parity, G on the other.
    y,x = np.indices((17,19))
    green = (x+y)%2 == 1
    valid = np.stack([~green,green,~green],axis=2)
    valid[5:12,5:12] = False
    image = np.broadcast_to([10.,20.,30.],valid.shape).copy()
    r = ReconstructionResult(image,valid.astype(float),valid,'adu','RGB','cpu','float64',
                             'final',False,provenance={'color_mode':'RGGB'})
    assert not r.copy_preview(max_side=10).validity[...,1].any()
    preview = display_result_preview(r,max_side=10)
    assert preview.validity[:2].all()
    assert not preview.validity[3:6,3:6].any()
    np.testing.assert_array_equal(preview.image[:2],np.broadcast_to([10.,20.,30.],preview.image[:2].shape))
    r.provenance['color_mode'] = 'RGB'
    unchanged = display_result_preview(r,max_side=10)
    np.testing.assert_array_equal(unchanged.validity,valid[::2,::2])

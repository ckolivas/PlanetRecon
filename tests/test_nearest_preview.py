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

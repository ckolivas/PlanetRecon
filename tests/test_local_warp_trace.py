from copy import copy
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.backends.torch_circular import TorchCircularRegistration
from tools.local_warp_trace import trace, smooth_motion


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('moving', [False, True])
def test_trace_prefixes_match_production(device, moving):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    reference=100+30*gaussian_filter(np.random.default_rng(34).normal(size=(80,96)),1.)
    y,x=np.indices(reference.shape)
    frame=(map_coordinates(reference,[y+.25*np.sin(x/20),x-.4*np.cos(y/18)],order=3,mode='reflect')
           if moving else reference)
    shift=(.1,-.2) if moving else (0.,0.)
    engine=TorchCircularRegistration(CircularMultiscaleRegistration(reference),device=device)
    proxy=LocalRegistration.proxy(frame)
    stages,records=trace(engine,proxy,shift)
    assert len(stages)==len(engine.layers)==len(records)
    for j,stage in enumerate(stages):
        prefix=copy(engine)
        prefix.layers=engine.layers[-(j+1):]
        expected=prefix.displacement(proxy,shift,lambda:None)
        np.testing.assert_allclose(stage.cpu().numpy(),expected.cpu().numpy(),atol=1e-9,rtol=0)
        assert np.isfinite(records[j]['measurements']).all()
        if not moving:
            np.testing.assert_allclose(stage.cpu().numpy(),0.,atol=1e-12)


def test_motion_regularization_retains_translation_and_does_not_mix_axes():
    y,x=np.indices((64,80))
    field=np.stack([np.full((64,80),.3),-.2+.5*np.cos(2*np.pi*x/20)])
    original=field.copy()
    result=smooth_motion(torch.as_tensor(field),2.).numpy()
    np.testing.assert_array_equal(field,original)
    np.testing.assert_allclose(result[0],.3,atol=1e-14)
    np.testing.assert_allclose(result.mean((1,2)),[.3,-.2],atol=1e-14)
    assert np.std(result[1]) < np.std(field[1])

import numpy as np
import pytest
from scipy.ndimage import fourier_shift, gaussian_filter
import torch

from planetrecon.pipeline.align import correlation_filter, correlation_peak
from tools.refined_registration_experiment import continuous_peak, project


@pytest.mark.parametrize('xy', [(0., 0.), (.43, -.27), (-2.38, 1.42)])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_continuous_peak_recovers_known_fourier_translation(xy, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    reference = gaussian_filter(np.random.default_rng(16).normal(size=(65, 81)), 1., mode='wrap')
    frame = np.fft.ifft2(fourier_shift(np.fft.fft2(reference), xy[::-1])).real
    cross = np.fft.fft2(frame)*np.conj(np.fft.fft2(reference))*correlation_filter(reference.shape)
    prior = correlation_peak(np.fft.ifft2(cross).real)
    refined, valid = continuous_peak(torch.as_tensor(cross[None], device=device), [prior])
    assert valid.all()
    np.testing.assert_allclose(refined[0], xy, rtol=0, atol=1e-8)


def test_unidentifiable_flat_frame_falls_back_without_nan():
    refined, valid = continuous_peak(torch.zeros((1, 17, 21), dtype=torch.complex128), [[.2, .3]])
    assert not valid.any()
    np.testing.assert_array_equal(refined[0], [.2, .3])


def test_cubic_recovers_translated_bandlimited_detail_better_than_bilinear():
    y, x = np.indices((81, 101))
    reference = 100+20*np.cos(2*np.pi*x/10)+15*np.cos(2*np.pi*y/9)
    xy = (.4, -.3)
    frame = np.fft.ifft2(fourier_shift(np.fft.fft2(reference), xy[::-1])).real
    outputs = project(frame, (xy, xy))
    roi = np.s_[12:-12, 12:-12]
    errors = [np.sqrt(np.mean((image[roi]-reference[roi])**2)) for image, _ in outputs]
    assert errors[2] < errors[0]*.1
    constant = project(np.full_like(frame, 17.), (xy, xy))
    for image, weight in constant:
        np.testing.assert_allclose(image[roi]/weight[roi], 17., atol=1e-12)

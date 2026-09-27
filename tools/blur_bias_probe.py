"""Separate blur-only false motion from noisy stationary-frame errors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.blur_matched_registration import BlurMatchedRegistration
from tools.validate_local_warp_centring import texture_scene, image_errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    reference = np.load('out/saturn-controlled-alignment/reference.npy')
    report = dict(model_sha256=hashlib.sha256(Path('tools/blur_matched_registration.py').read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(), frames=[])
    for name, scene in [('saturn', reference), ('texture', texture_scene(reference.shape))]:
        engine = BlurMatchedRegistration(scene, CircularMultiscaleRegistration(scene))
        mask = binary_erosion(scene > .08*scene.max(), iterations=3)
        mask[:8] = mask[-8:] = False
        mask[:, :8] = mask[:, -8:] = False
        for sigma in (0., .5, 1., 1.5, 2.):
            clean = gaussian_filter(scene, sigma) if sigma else scene
            for noise, seed in [(0., 9017), (2., 9017), (2., 9018)]:
                frame = clean+np.random.default_rng(seed).normal(0, noise, scene.shape)
                fields = engine.variants(frame, (0.,0.), lambda:None, known_sigma=sigma)
                values = {}
                for key, field in fields.items():
                    warped = sample(torch.as_tensor(clean, device=engine.device),
                                    engine.yy+field[1], engine.xx+field[0]).cpu().numpy()
                    values[key] = dict(clean_image=image_errors(warped, clean, mask),
                        field_vector_rms_px=float(torch.sqrt(field[:, mask].square().sum(0).mean())))
                report['frames'].append(dict(scene=name, sigma=sigma, noise=noise, seed=seed,
                                              variants=values, statistics=engine.stats[-1]))
            print(name, sigma, 'complete', flush=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':main()

"""Check that known fractional global translations do not invent local motion."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import torch

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.validated_registration import ValidatedRegistration
from tools.validate_local_warp_centring import texture_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference = np.load('out/saturn-controlled-alignment/reference.npy')
    yy, xx = np.indices(reference.shape, dtype=float)
    results = {'reference_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
               'model_sha256': hashlib.sha256(Path('tools/validated_registration.py').read_bytes()).hexdigest(),
               'seed': 4407, 'noise_sigma_adu': 2., 'device': args.device, 'scenes': {}}
    for name, scene in [('saturn', reference), ('texture', texture_scene(reference.shape))]:
        engine = ValidatedRegistration(scene, CircularMultiscaleRegistration(scene), device=args.device)
        rows = []
        for blur in (0., .75):
            rng = np.random.default_rng(4407)
            for dx, dy in ((.25, .5), (-.5, .25), (.75, -.25)):
                source = gaussian_filter(scene, blur) if blur else scene
                frame = map_coordinates(source, [yy-dy, xx-dx], order=3, mode='reflect')
                frame += rng.normal(0, 2., scene.shape)
                field = engine.displacement_raw(frame, (dx, dy), lambda: None).cpu().numpy()
                mask = scene > .08*scene.max()
                residual = field-np.array([dx, dy])[:, None, None]
                rows.append({'blur': blur, 'shift': [dx, dy], 'residual_vector_rms':
                    float(np.sqrt(np.mean(np.sum(residual[:, mask]**2, axis=0)))),
                    'statistics': engine.stats[-1]})
        results['scenes'][name] = rows
        print(name, [(r['blur'], r['residual_vector_rms'], r['statistics']['accepted_halves'])
                     for r in rows], flush=True)
    args.out.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()

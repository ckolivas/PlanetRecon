"""Run production stack_source on independent even/odd halves of the screened frames.

Each half gets the frames of one parity (in screened capture order) and its own
references; both anchor to adjacent frames so a rotating planet shares one
geometry. Output npz matches xstack.py (image, a, b) for snr.py.
"""
import argparse, json, time
from dataclasses import replace
import numpy as np
from planetrecon.io.source import open_source
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.pipeline.preprocess_cache import load_cache, selection_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('capture'); p.add_argument('--cache', required=True)
    p.add_argument('--config', required=True, help='JSON of ReconstructionConfig overrides')
    p.add_argument('--out', required=True)
    p.add_argument('--no-lq', action='store_true', help='experiment only: skip motion local quality')
    a = p.parse_args()
    over = json.loads(a.config)
    if a.no_lq:
        from planetrecon.pipeline import geometry_stack
        geometry_stack._motion_local_quality = lambda matcher, selection, *args, **kw: (None, selection)
    cfg = ReconstructionConfig(**{**dict(device='gpu', threads=32, frame_preselection=True, local_alignment=True), **over})
    with open_source(a.capture) as src:
        sel, rep = load_cache(src, cfg, path=a.cache)
        if sel is None:
            raise SystemExit(rep)
        screened = np.flatnonzero(sel.accepted)
        # Anchor: the best frame overall and its nearest screened neighbour (other parity).
        globally = np.flatnonzero(best_frame_mask(sel, cfg.stack_percent, cfg.frame_selection_mode))
        best = int(globally[np.argmax(sel.measurements[globally, 0])])
        k = int(np.searchsorted(screened, best))
        partner = int(screened[k+1] if k+1 < len(screened) else screened[k-1])
        images, weights, info = [], [], {}
        for h in (0, 1):
            mask = np.zeros_like(sel.accepted)
            mask[screened[h::2]] = True
            half = replace(sel, accepted=mask)
            half.digest = selection_digest(half)
            anchor = best if mask[best] else partner
            t = time.time()
            r = stack_source(src, replace(cfg, reference_index=anchor), preprocessing=half)
            info[f'half{h}'] = dict(seconds=time.time()-t, n_used=r.n_used, anchor=anchor, warnings=r.warnings)
            print(f'half {h}: {r.n_used} frames, {time.time()-t:.0f}s, anchor {anchor}', flush=True)
            images.append(r.image); weights.append(r.coverage)
    total = (images[0]*weights[0] + images[1]*weights[1]) / np.maximum(weights[0]+weights[1], 1e-30)
    info.update(config=over, n=sum(info[f'half{h}']['n_used'] for h in (0, 1)))
    np.savez_compressed(a.out, image=total, a=images[0], b=images[1], info=json.dumps(info, default=str))


if __name__ == '__main__':
    main()

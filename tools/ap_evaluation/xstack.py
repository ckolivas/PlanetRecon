"""Experimental stacker: PlanetRecon components, interleaved half stacks.

Baseline ('pr') reproduces the application's default local path: best-frame
reference -> 64-frame globally aligned template -> global phase correlation ->
circular multiscale local field (CUDA) -> bilinear raw backprojection, linear
quality weights. Halves A/B alternate in selected capture order.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from planetrecon.io.ser import SERSource
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.pipeline.preprocess_cache import load_cache
from planetrecon.detector import is_bayer, PreparedBilinearDemosaic


def open_capture(path, percent, mode, cache=None):
    source = SERSource(path)
    cfg = ReconstructionConfig(frame_preselection=True, device='gpu', threads=32)
    selection, report = load_cache(source, cfg, path=cache)
    if selection is None:
        raise SystemExit(f"cache not usable: {report.get('status')} {report.get('reason')}")
    mask = best_frame_mask(selection, percent, mode)
    return source, selection, mask


def planes(source):
    color = source.color_mode()
    h, w = source.frame_shape()[:2]
    if is_bayer(color):
        demosaic = PreparedBilinearDemosaic((h, w), color)
        return lambda raw: demosaic(raw) @ np.array([.25, .5, .25])
    return lambda raw: np.asarray(raw, float)


class Accumulator:
    def __init__(self, shape, rgb):
        s = (*shape, 3) if rgb else shape
        self.sum = [np.zeros(s), np.zeros(s)]
        self.wt = [np.zeros(s), np.zeros(s)]

    def add(self, half, score, add, wt):
        self.sum[half] += score*add
        self.wt[half] += score*wt

    def images(self):
        def norm(s, w):
            return np.divide(s, w, out=np.zeros_like(s), where=w > 0)
        total = norm(self.sum[0]+self.sum[1], self.wt[0]+self.wt[1])
        return total, norm(self.sum[0], self.wt[0]), norm(self.sum[1], self.wt[1]), self.wt[0]+self.wt[1]


def run_pr(source, selection, mask, args):
    """Two fully independent pipelines (own best frame, template and matcher)."""
    from planetrecon.backends.torch_accel import TorchBackend
    from planetrecon.pipeline.local_align import build_template
    from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
    plane_of = planes(source)
    color = source.color_mode()
    indices = np.flatnonzero(mask)
    scores = selection.measurements[:, 0]
    read_plane = lambda i: plane_of(source.read_raw(int(i)).astype(float))
    halves = []
    # Both halves anchor to adjacent selected frames so a rotating planet has
    # the same geometry; templates and matchers stay independent.
    k_best = int(np.argmax(scores[indices]))
    anchors = {k_best % 2: int(indices[k_best]),
               1 - k_best % 2: int(indices[k_best+1 if k_best+1 < len(indices) else k_best-1])}
    for h in (0, 1):
        own = indices[h::2]
        backend = TorchBackend()
        best = anchors[h]
        reference = read_plane(best)
        if args.local != 'none' or args.template:
            candidates = own[np.argsort(-scores[own], kind='stable')[:args.template_frames]]
            reference = build_template(reference, candidates, read_plane, backend.phase_correlation, 32.)
            for _ in range(args.template_passes - 1):
                # Refine: align the template frames locally to the previous template.
                from apalign import APRegistration
                reference = backend.prepare_reference(reference)
                m = APRegistration(reference, size=args.ap_size, search=args.ap_search, min_ncc=args.ap_ncc,
                                   tensor=not args.ap_scalar, prior=args.ap_prior)
                total = np.zeros(reference.shape); wsum = np.zeros(reference.shape)
                for i in candidates:
                    pl = read_plane(i)
                    field = m.displacement_tensor(pl, backend.phase_correlation(reference, pl))
                    a, wt, _, _ = backend.backproject(pl, field, 'mono')
                    total += a; wsum += wt
                reference = np.divide(total, wsum, out=reference.copy(), where=wsum > .5)
        reference = backend.prepare_reference(reference)
        matcher = None
        if args.local == 'circular':
            matcher = CircularMultiscaleRegistration(reference, sampling_multiplier=args.sampling, wavelength_nm=550.)
        elif args.local == 'ap':
            from apalign import APRegistration
            matcher = APRegistration(reference, size=args.ap_size, search=args.ap_search, min_ncc=args.ap_ncc, tensor=not args.ap_scalar, prior=args.ap_prior)
            print(f'  half {h}: {len(matcher.centres)} alignment points')
        halves.append((backend, reference, matcher, best))
    acc = Accumulator(source.frame_shape()[:2], is_bayer(color) or source.frame_shape()[-1:] == (3,))
    shifts = []
    t0 = time.time()
    for k, index in enumerate(indices):
        backend, reference, matcher, _ = halves[k % 2]
        raw = source.read_raw(int(index)).astype(float)
        plane = plane_of(raw)
        shift = backend.phase_correlation(reference, plane)
        shifts.append(shift)
        field = shift if matcher is None else matcher.displacement_tensor(plane, shift)
        add, wt, demo, support = backend.backproject(raw, field, color)
        score = scores[index] if args.weights == 'linear' else 1.
        acc.add(k % 2, score, add, wt)
        if k % 1000 == 0:
            print(f'  {k}/{len(indices)} {time.time()-t0:.1f}s', flush=True)
    stats = [getattr(h[2], 'stats', None) for h in halves]
    if stats[0]:
        st = np.array(stats[0] + stats[1])
        print(f'  AP acceptance {st[:, 0].mean():.2f}, median NCC {st[:, 1].mean():.3f}')
    return acc, dict(reference=[h[3] for h in halves], n=len(indices), shifts=np.array(shifts).tolist())


def make_reference(source, own, scores, best, backend, read_plane, args):
    from planetrecon.pipeline.local_align import build_template
    from apalign import APRegistration
    reference = read_plane(best)
    candidates = own[np.argsort(-scores[own], kind='stable')[:args.template_frames]]
    reference = build_template(reference, candidates, read_plane, backend.phase_correlation, 32.)
    for _ in range(args.template_passes - 1):
        reference = backend.prepare_reference(reference)
        m = APRegistration(reference, size=args.ap_size, search=args.ap_search, min_ncc=args.ap_ncc,
                           tensor=not args.ap_scalar, prior=args.ap_prior)
        total = np.zeros(reference.shape); wsum = np.zeros(reference.shape)
        for i in candidates:
            pl = read_plane(i)
            field = m.displacement_tensor(pl, backend.phase_correlation(reference, pl))
            a, wt, _, _ = backend.backproject(pl, field, 'mono')
            total += a; wsum += wt
        reference = np.divide(total, wsum, out=reference.copy(), where=wsum > .5)
    return backend.prepare_reference(reference)


def run_lq(source, selection, mask, args):
    """Local quality: each AP keeps its own best args.percent of screened frames."""
    import torch
    from planetrecon.backends.torch_accel import TorchBackend
    from apalign import APRegistration
    from localq import LocalQuality, blend_masks
    plane_of = planes(source)
    color = source.color_mode()
    screened = np.flatnonzero(selection.accepted)
    scores = selection.measurements[:, 0]
    read_plane = lambda i: plane_of(source.read_raw(int(i)).astype(float))
    global_sel = np.flatnonzero(mask)
    k_best = int(np.argmax(scores[global_sel]))
    pos = {int(i): n for n, i in enumerate(screened)}
    best = int(global_sel[k_best])
    # Anchor halves to adjacent screened frames (rotation), halves alternate in screened order.
    nb = screened[pos[best]+1] if pos[best]+1 < len(screened) else screened[pos[best]-1]
    anchors = {pos[best] % 2: best, 1 - pos[best] % 2: int(nb)}
    shape = source.frame_shape()[:2]
    halves = []
    t0 = time.time()
    for h in (0, 1):
        own = screened[h::2]
        own_sel = own[mask[own]]
        backend = TorchBackend()
        reference = make_reference(source, own_sel, scores, anchors[h], backend, read_plane, args)
        matcher = APRegistration(reference, size=args.ap_size, search=args.ap_search, min_ncc=args.ap_ncc,
                                 tensor=not args.ap_scalar, prior=args.ap_prior)
        lq = LocalQuality(matcher.centres, shape, ap_size=args.lq_size or args.ap_size)
        # Pass 1: local quality of every screened frame in this half.
        Q = np.zeros((len(own), len(matcher.centres)))
        for n, i in enumerate(own):
            pl = read_plane(i)
            Q[n] = lq.measure(pl, backend.phase_correlation(reference, pl))
        count = max(1, (len(own)*args.percent + 99)//100)
        chosen = np.zeros_like(Q, dtype=bool)
        order = np.argsort(-Q, axis=0, kind='stable')[:count]
        for k in range(Q.shape[1]):
            chosen[order[:, k], k] = True
        M, rest = blend_masks(matcher.centres, shape, matcher.spacing)
        halves.append(dict(own=own, backend=backend, reference=reference, matcher=matcher, chosen=chosen,
                           M=M, rest=rest, gsel=mask[own], Q=Q))
        print(f'  half {h}: {len(matcher.centres)} APs, local quality {time.time()-t0:.0f}s, '
              f'frames used by any AP {chosen.any(1).sum()}/{len(own)}', flush=True)
    rgb = is_bayer(color) or source.frame_shape()[-1:] == (3,)
    acc = Accumulator(shape, rgb)
    n_used = 0
    for h, H in enumerate(halves):
        backend, matcher = H['backend'], H['matcher']
        for n, i in enumerate(H['own']):
            sel_k = H['chosen'][n]
            g = bool(H['gsel'][n])
            if not sel_k.any() and not g:
                continue
            raw = source.read_raw(int(i)).astype(float)
            pl = plane_of(raw)
            shift = backend.phase_correlation(H['reference'], pl)
            field = matcher.displacement_tensor(pl, shift)
            add, wt, _, _ = backend.backproject(raw, field, color)
            sk = torch.as_tensor(sel_k, device=H['M'].device, dtype=torch.float64)
            W = (torch.einsum('k,kyx->yx', sk, H['M']) + H['rest']*float(g)).cpu().numpy()
            if args.weights == 'linear':
                W = W*scores[i]
            if rgb:
                W = W[..., None]
            acc.sum[h] += W*add
            acc.wt[h] += W*wt
            n_used += 1
            if n_used % 2000 == 0:
                print(f'  stacked {n_used} {time.time()-t0:.0f}s', flush=True)
    return acc, dict(reference=list(anchors.values()), n=n_used)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('capture')
    p.add_argument('--percent', type=int, required=True)
    p.add_argument('--mode', default='frame_count')
    p.add_argument('--method', default='pr')
    p.add_argument('--local', default='circular', choices=['circular', 'none', 'ap'])
    p.add_argument('--weights', default='linear', choices=['linear', 'equal'])
    p.add_argument('--sampling', type=float, default=5.)
    p.add_argument('--ap-size', type=int, default=48)
    p.add_argument('--ap-search', type=int, default=6)
    p.add_argument('--ap-ncc', type=float, default=0.5)
    p.add_argument('--ap-scalar', action='store_true')
    p.add_argument('--template-frames', type=int, default=64)
    p.add_argument('--lq-size', type=int, default=0)
    p.add_argument('--cache', default=None)
    p.add_argument('--template-passes', type=int, default=1)
    p.add_argument('--ap-prior', type=float, default=0.01)
    p.add_argument('--template', action='store_true', help='global-only runs still use a 64-frame template')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    source, selection, mask = open_capture(args.capture, args.percent, args.mode, args.cache)
    t = time.time()
    acc, info = {'pr': run_pr, 'lq': run_lq}[args.method](source, selection, mask, args)
    total, a, b, wt = acc.images()
    info.update(vars(args), seconds=time.time()-t)
    np.savez_compressed(args.out, image=total, a=a, b=b, weight=wt, info=json.dumps(info))
    print(f"saved {args.out}: {info['n']} frames, {info['seconds']:.1f}s")


if __name__ == '__main__':
    main()

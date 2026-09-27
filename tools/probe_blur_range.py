"""Check whether the fixed two-pixel blur bank truncates global-pose fits.

This score-only sensitivity check does not alter any displacement or stack.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
import torch
import torch.nn.functional as F

from planetrecon.backends.torch_circular import sample
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.local_align import LocalRegistration
from tools.blur_matched_registration import choose_blur
from tools.coherent_saturn_experiment import inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    _, indices, _, shifts, reference, hashes = inputs()
    h,w = reference.shape
    yy,xx = torch.meshgrid(torch.arange(h,dtype=torch.float64),
                          torch.arange(w,dtype=torch.float64), indexing='ij')
    sigmas = np.arange(0.,4.01,.25)
    bank = torch.tensor(np.stack([LocalRegistration.proxy(gaussian_filter(reference,s) if s else reference)
                                  for s in sigmas]))
    mask = binary_dilation(gaussian_filter(reference,3.)>reference.max()*.04, iterations=6)
    mask[:8] = mask[-8:] = False
    mask[:,:8] = mask[:,-8:] = False
    mask = torch.tensor(mask.astype(float))
    rows = []
    with SERSource('2024-09-27-1154_3-CK-R-Sat.ser') as source:
        for position in np.linspace(0,len(indices)-1,96,dtype=int):
            sx,sy = shifts[position]
            x,y = xx-sx, yy-sy
            valid = sample(mask,y,x)>.5
            grid = torch.stack((2*x/(w-1)-1,2*y/(h-1)-1),dim=-1)[None]
            predictions = F.grid_sample(bank[None],grid,mode='bicubic',padding_mode='border',align_corners=True)[0]
            observed = torch.tensor(LocalRegistration.proxy(source.read_raw(int(indices[position]))))
            narrow,_ = choose_blur(predictions[:9],observed,valid)
            wide,losses = choose_blur(predictions,observed,valid)
            rows.append(dict(position=int(position),sigma_2=float(sigmas[narrow]),sigma_4=float(sigmas[wide]),
                relative_loss_reduction=(losses[narrow]-losses[wide])/max(losses[narrow],1e-30)))
    report = dict(hashes=hashes,model_sha256=hashlib.sha256(Path('tools/blur_matched_registration.py').read_bytes()).hexdigest(),
                  score_only=True,frames=rows)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('96 frames;', sum(r['sigma_4']>2 for r in rows), 'prefer blur beyond two pixels;',
          sum(r['sigma_4']==4 for r in rows), 'hit four-pixel boundary')


if __name__ == '__main__':main()

"""Diagnostic native-size unit-square deposition through an inverted local warp.

The field maps output coordinates to detector coordinates. Drops are fixed,
axis-aligned unit squares in output space, not warped detector polygons.
"""
import torch
from planetrecon.backends.torch_circular import sample


def inverse_coordinates(field, tolerance=1e-7, iterations=40):
    if field.ndim != 3 or field.shape[0] != 2 or not bool(torch.isfinite(field).all()):
        raise ValueError('Expected a finite XY displacement field')
    h,w=field.shape[1:]
    y,x=torch.meshgrid(torch.arange(h,device=field.device,dtype=field.dtype),
                       torch.arange(w,device=field.device,dtype=field.dtype),indexing='ij')
    # Newton solves q + d(q) = detector_position. Nearest exterior extension
    # matches the constant global translation around the actual object support.
    qx,qy=x-field[0],y-field[1]
    fy,fx=torch.gradient(field,dim=(1,2))
    for step in range(iterations):
        displacement=sample(field,qy,qx,nearest=True)
        rx,ry=qx+displacement[0]-x,qy+displacement[1]-y
        error=float(torch.hypot(rx,ry).max())
        if error <= tolerance:
            return qy,qx,dict(iterations=step,max_inverse_error_px=error)
        dx,dy=sample(fx,qy,qx,nearest=True),sample(fy,qy,qx,nearest=True)
        # Clamped extension has zero derivative in the clamped coordinate.
        dx=dx*((qx>=0)&(qx<=w-1))
        dy=dy*((qy>=0)&(qy<=h-1))
        a,b,c,d=1+dx[0],dy[0],dx[1],1+dy[1]
        det=a*d-b*c
        if not bool(torch.isfinite(det).all() & (det > .05).all()):
            raise ValueError('Singular or reversed inverse warp')
        qx=qx-((d*rx-b*ry)/det).clamp(-2.,2.)
        qy=qy-((-c*rx+a*ry)/det).clamp(-2.,2.)
    raise ValueError(f'Inverse warp failed tolerance: {error}')


def splat(raw, qy, qx):
    """Unit-square overlap equals the triangular weight at deposited centres."""
    h,w=raw.shape
    iy,ix=torch.floor(qy).long(),torch.floor(qx).long()
    fy,fx=qy-iy,qx-ix
    signal,support=torch.zeros_like(raw),torch.zeros_like(raw)
    for dy,wy in ((0,1-fy),(1,fy)):
        for dx,wx in ((0,1-fx),(1,fx)):
            y,x=iy+dy,ix+dx
            valid=(y>=0)&(y<h)&(x>=0)&(x<w)
            weights=wy*wx*valid
            dest=(y.clamp(0,h-1)*w+x.clamp(0,w-1)).flatten()
            signal.flatten().scatter_add_(0,dest,(raw*weights).flatten())
            support.flatten().scatter_add_(0,dest,weights.flatten())
    return signal,support


def deposit(raw, field):
    qy,qx,stats=inverse_coordinates(field)
    signal,support=splat(raw,qy,qx)
    return signal,support,stats

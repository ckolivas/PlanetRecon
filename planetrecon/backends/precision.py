"""Floating-point precision of the grouped CUDA motion path.

Registration work (filtered and resampled planes, alignment-point and local
quality correlations, residual fields) runs in ``REGISTRATION`` precision.
Geometry coordinates, field solves, peak decisions and every accumulated sum
stay float64. Measured on real captures, float32 registration moves the
displacement fields by about 2e-4 px rms and leaves stacks far below their
noise floor (docs/alignment-points.md); tests set float64 to compare
the device path with the CPU reference exactly.
"""
import torch

REGISTRATION = torch.float32

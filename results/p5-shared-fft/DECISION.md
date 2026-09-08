# Shared-transform CPU/CUDA pilot

All four modes (reference, uncached shared CPU, cached shared CPU, cached CUDA)
pass full-optical-grid forward, adjoint, normal, objective, reconstructed-image
and independent CPU certificate checks. The scene has 1152×1152 cells and uses
three 128×128 observations with scalar observed-data variances. No truth enters
the reconstruction or variance estimate.

| Mode | Warm normal evaluation | Complete iterative solve |
|---|---:|---:|
| Reference | 0.6272 s | 20.3681 s |
| Shared CPU, no PSF cache | 0.3289 s | 10.4085 s |
| Shared CPU, bounded PSF cache | 0.3014 s | 10.0965 s |
| Shared CUDA, bounded PSF cache | 0.01666 s | 1.5091 s |

Kernel times are medians of three repeats. Solve times include solver iterations,
checks and its final objective; setup, independent reference checks and total
mode wall time are separately recorded in the report. Runs were sequential on
the same host, with two CPU threads and the local RTX 5070/venv CUDA runtime.
All source/runtime/input identities remained unchanged. This is a three-frame
pilot, not full-family, Q3, release, or general-device qualification.

The cache limit is 256 MiB of retained PSF spectra, not a total RAM/VRAM limit.
The implementation shares the scene forward transform and summed adjoint inverse
transform, and evicts cached spectra within the declared bound. Reported memory
separates process high-water RSS, retained spectra and Torch peak allocations.
Only zero-reference-translation operators with common native/kernel shapes take
this path. Other geometries must use the explicit reference operator; no motion
approximation or implicit fallback was added.

Proceed with the predeclared bounded prior/domain/sampling studies on CUDA while
retaining independent CPU certificates. Full-frame-count profiling and bounded
resource/cancellation/recovery acceptance remain P5 work.

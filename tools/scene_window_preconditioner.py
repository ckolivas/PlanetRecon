"""Window-averaged periodic inverse candidate; never a forward-model replacement."""
import numpy as np
from tools.scene_periodic_preconditioner import PeriodicScenePreconditioner


class WindowAveragedPreconditioner(PeriodicScenePreconditioner):
    def __init__(self, problem, *, workers=8):
        super().__init__(problem, workers=workers)
        native = [getattr(op, 'base', op) for op in problem.operators]
        signatures = {(op.bin_factor, tuple(op.detector_shape)) for op in native}
        if len(signatures) != 1: raise ValueError('common detector sampling required')
        b, detector_shape = signatures.pop()
        self.window_fraction = float(b*b*np.prod(detector_shape)/np.prod(self.shape[:2]))
        if not 0 < self.window_fraction <= 1: raise ValueError('detector window must fit native scene')
        fy=np.fft.fftfreq(self.shape[0]);fx=np.fft.rfftfreq(self.shape[1])
        laplacian=4*np.sin(np.pi*fy[:,None])**2+4*np.sin(np.pi*fx[None,:])**2
        if len(self.shape)==3: laplacian=laplacian[...,None]
        regularizer=problem.ridge+problem.smoothness*laplacian
        # The old data symbol uses sum(weights)/(b² * detector_pixels).
        # Multiplication by this fraction gives sum(weights)/native_pixels,
        # preserving the total observation weight under spatial averaging.
        self.symbol=self.window_fraction*self.symbol+(1-self.window_fraction)*regularizer

    def info(self):
        return super().info() | {'kind':'window_averaged_periodic_inverse_v1',
                'window_fraction':self.window_fraction,
                'scope':'Positive inverse candidate with conserved observation density; aliases and finite-boundary coupling remain in the exact Hessian.'}

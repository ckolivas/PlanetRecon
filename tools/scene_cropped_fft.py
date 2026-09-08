"""Experimental exact detector-crop FFT padding with bounded retained spectra.

Circular folding may affect discarded convolution samples, but cannot reach any
retained detector sample under the derived bounds. The adjoint uses the same
embedding and exact transpose. See docs/scene-cropped-fft.md.
"""
from scipy import fft
from tools.scene_retained_cache import RetainedSceneFFTBatch
from tools.scene_study import CellFFTBatch


def crop_fft_shape(scene_shape, kernel_shape, crops):
    """Sufficient common FFT dimensions for half-open full-convolution crops."""
    if len(scene_shape) not in (2,3) or len(kernel_shape)!=2 or not crops:
        raise ValueError('scene/kernel dimensions and at least one crop required')
    if any(int(v)!=v or v<1 for v in (*scene_shape,*kernel_shape)):
        raise ValueError('positive integer dimensions required')
    result=[]
    for axis,(n,k) in enumerate(zip(scene_shape[:2],kernel_shape)):
        size=max(n,k)
        for crop in crops:
            if len(crop)!=2 or len(crop[axis])!=2:
                raise ValueError('two half-open crop intervals required')
            start,stop=crop[axis]
            if int(start)!=start or int(stop)!=stop or not 0<=start<stop<=n+k-1:
                raise ValueError('crop outside full convolution')
            # N>=n,k embeds inputs without truncation; stop<=N embeds the
            # output crop; N>=n+k-1-start excludes every wrapped high sample.
            size=max(size,stop,n+k-1-start)
        result.append(fft.next_fast_len(int(size),real=True))
    return tuple(result)


class CroppedSceneFFTBatch(RetainedSceneFFTBatch):
    def __init__(self,operators,**kwargs):
        super().__init__(operators,**kwargs)
        self.full_fft_shape=self.fft_shape
        crops=[tuple((s.start,s.stop) for s in self._crop_slices(op)) for op in self.operators]
        self.fft_shape=crop_fft_shape(self.shape,self.kernel_shape,crops)

    def cache_info(self):
        return super().cache_info() | {'padding':'exact_detector_crop_v1',
            'fft_shape':list(self.fft_shape),'full_fft_shape':list(self.full_fft_shape),
            'padding_scope':'Only discarded convolution samples may fold; retained forward crop and its adjoint are unchanged in exact arithmetic.'}


class CroppedCellFFTBatch(CellFFTBatch):
    def __init__(self,operators,**kwargs):
        self.operators=tuple(operators)
        if not self.operators or len({op.factor for op in self.operators})!=1:
            raise ValueError('common cell factor required')
        self.factor=self.operators[0].factor
        self.native=CroppedSceneFFTBatch([op.base for op in self.operators],**kwargs)

import numpy as np
import pytest
from planetrecon.result import ReconstructionResult
from planetrecon.export import export_result, ExportConfig
from tools.compare_stack_reference import read_reference, compare


def test_reference_preserves_low_bits_and_perfect_match(tmp_path):
    pytest.importorskip('PySide6')
    y,x=np.indices((48,64))
    image=np.stack([10000+x*7+y*3,12000+x*9+y*4,14000+x*13+y*7],axis=2).astype(float)
    r=ReconstructionResult(image,np.ones_like(image),np.ones_like(image,dtype=bool),
                           'adu','RGB','cpu','float64','final',False)
    path=tmp_path/'reference.png'
    export_result(r,path,ExportConfig('png16',0,65535))
    ref=read_reference(path)
    np.testing.assert_array_equal(ref,image)
    metrics,_=compare(ref,r)
    assert metrics['relative_rmse']<1e-12
    assert metrics['highpass_correlation_sigma3']>1-1e-10

import json
import numpy as np
import pytest
from planetrecon.operators import SceneDetectorOperator
from tools.scene_quadratic import SceneQuadratic
from tools.scene_state_diagnostics import reference_stage,fourier_energy
from tools.scene_window_preconditioner import WindowAveragedPreconditioner
from tools.study_io import file_hash


def test_frozen_stage_requires_checksum_objective_and_feasible_image(tmp_path):
    original={'input':'frozen'};selection={'indices':[1,2]}
    (tmp_path/'identity.json').write_text(json.dumps({'protocol':original,'selection':selection}))
    info={'solver':'extended_scene_projected_acceleration_v4','maxiter':750,'n_iter':310,'converged':True,'scaling':'diagonal'}
    path=tmp_path/'budget-750.npz'
    np.savez(path,image=np.ones((2,2)),metadata=json.dumps(info))
    digest=file_hash(path)
    (tmp_path/'budget-750.json').write_text(json.dumps({'status':'completed','sha256':digest}))
    np.testing.assert_array_equal(reference_stage(tmp_path,original,selection,digest,shape=(2,2)),1.)
    with pytest.raises(ValueError,match='objective'): reference_stage(tmp_path,{},selection,digest,shape=(2,2))
    with pytest.raises(ValueError,match='scene'): reference_stage(tmp_path,original,selection,digest)
    with path.open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='checksum'): reference_stage(tmp_path,original,selection,digest,shape=(2,2))


@pytest.mark.parametrize('origin',[(0,0),(1,1)])
def test_forward_fourier_energy_matches_dense_hessian_at_boundary_and_interior(origin):
    op=SceneDetectorOperator((6,6),(np.ones((3,4))/12,),2,origin,(2,2))
    p=SceneQuadratic([op],[np.ones((2,2))],[np.array([[1.,2.],[3.,4.]])],ridge=.01,smoothness=.02)
    basis=np.eye(36).reshape((36,6,6));h=np.stack([p.normal(x).ravel() for x in basis],axis=1)
    yy,xx=np.indices((6,6))
    for ky,kx in [(0,0),(1,0),(0,1),(2,2)]:
        phi=np.exp(2j*np.pi*(ky*yy/6+kx*xx/6)).ravel()/6
        np.testing.assert_allclose(fourier_energy(p,ky,kx),(phi.conj()@h@phi).real,rtol=1e-12,atol=1e-12)


def test_boundary_truncation_does_not_claim_exact_periodic_symbol():
    op=SceneDetectorOperator((4,4),(np.ones((3,3))/9,),1,(0,0),(2,2))
    p=SceneQuadratic([op],[np.ones((2,2))],[1.],ridge=.01)
    inverse=WindowAveragedPreconditioner(p,workers=1)
    assert abs(fourier_energy(p,0,0)-inverse.symbol[0,0])>.01


def test_declared_larger_reference_cap_is_bound_to_payload_metadata(tmp_path):
    original={'input':'frozen'};selection={'indices':[1,2]}
    (tmp_path/'identity.json').write_text(json.dumps({'protocol':original,'selection':selection}))
    info={'solver':'extended_scene_projected_acceleration_v4','maxiter':1500,'n_iter':1410,'converged':True,'scaling':'diagonal'}
    path=tmp_path/'budget-1500.npz';np.savez(path,image=np.ones((2,2)),metadata=json.dumps(info))
    digest=file_hash(path)
    (tmp_path/'budget-1500.json').write_text(json.dumps({'status':'completed','sha256':digest}))
    np.testing.assert_array_equal(reference_stage(tmp_path,original,selection,digest,shape=(2,2),cap=1500,iterations=1410),1.)
    with pytest.raises(ValueError,match='unexpected reference'):
        reference_stage(tmp_path,original,selection,digest,shape=(2,2),cap=1500,iterations=1400)
    with pytest.raises(ValueError,match='cap'):
        reference_stage(tmp_path,original,selection,digest,shape=(2,2),cap=1500,iterations=1501)

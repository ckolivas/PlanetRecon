"""Check grid capacity and the unobserved motion modes of exact patch averages.

The spline basis is orthonormalized in detector-pixel L2 before SVD, so the
minimum-norm observation inverse minimizes field energy, not coefficient norm.
This is an oracle diagnostic, not a proposed noise-sensitive estimator.
"""
import json
from pathlib import Path

import numpy as np
import torch

from tools.ap_oracle_observations import source_hashes, patch_averages, CASES
from tools.regularized_ap_fit import RegularizedAPFit
from tools.joint_saturn_experiment import digest


def orthogonal_operator(fitter):
    qy, ry = np.linalg.qr(fitter.by, mode='reduced')
    qx, rx = np.linalg.qr(fitter.bx, mode='reduced')
    transform = np.kron(np.linalg.inv(ry), np.linalg.inv(rx))
    operator = np.asarray(fitter.design@transform)
    u, s, vt = np.linalg.svd(operator, full_matrices=False)
    tolerance = max(operator.shape)*np.finfo(float).eps*s[0]
    rank = int(np.count_nonzero(s > tolerance))
    return qy, qx, operator, u, s, vt, rank, tolerance


def main():
    torch.set_num_threads(4)
    root = Path('out/ap-oracle-observations')
    report = json.loads((root/'report.json').read_text())
    if report['hashes'] != source_hashes():
        raise ValueError('Experiment sources changed')
    with np.load(root/'geometry.npz') as data:
        reference, mask = data['reference'], data['mask']
    model = RegularizedAPFit(reference, device='cpu', policies=['baseline'])
    qy, qx, operator, u, singular, vt, rank, tolerance = orthogonal_operator(model.engine.fitter)
    result = dict(source_sha256=digest(__file__), experiment_report_sha256=digest(root/'report.json'),
        ap_count=operator.shape[0], coefficients=operator.shape[1], numerical_rank=rank,
        unobserved_dimensions=operator.shape[1]-rank, rank_tolerance=tolerance,
        singular_values=singular.tolist(), retained_condition_number=float(singular[0]/singular[rank-1]), cases={})
    for case in CASES:
        values = dict(dense_projection=[], minimum_field_energy_from_projected_averages=[],
                      inverse_from_exact_averages=[], unobserved_component=[], measured_fit=[], true_all_fit=[])
        residuals, observation_errors = [], []
        with np.load(root/f'{case}.npz') as data:
            if report['cases'][case]['raw_sha256'] != digest(root/f'{case}.npz'):
                raise ValueError('Control results changed')
            for truth, ideal, measured_fit, true_fit in zip(data['truth_fields'], data['ideal_observations'],
                    data['field_measured'], data['field_truth_all']):
                # Reconstruct the exact global origin used in the experiment from
                # the constant difference between truth averages and ideal residuals.
                origins = patch_averages(model.engine, truth)-ideal
                shift = origins.mean(0)
                np.testing.assert_allclose(origins, np.broadcast_to(shift, origins.shape), atol=1e-12, rtol=0)
                residual = truth-shift[:, None, None]
                z = np.stack([(qy.T@v@qx).ravel() for v in residual], axis=1)
                projected = np.stack([qy@z[:, i].reshape(qy.shape[1], qx.shape[1])@qx.T for i in range(2)])
                observations = operator@z
                # Equivalent to pinv(operator) @ observations, without dividing
                # by very small singular values in a round-trip of noiseless data.
                recoverable = vt[:rank].T@(vt[:rank]@z)
                recovered = np.stack([qy@recoverable[:, i].reshape(qy.shape[1], qx.shape[1])@qx.T for i in range(2)])
                exact_inverse = vt[:rank].T@((u[:, :rank].T@ideal)/singular[:rank, None])
                inverted = np.stack([qy@exact_inverse[:, i].reshape(qy.shape[1], qx.shape[1])@qx.T for i in range(2)])
                null = projected-recovered
                residuals.append(float(np.max(abs(operator@(z-recoverable)))))
                observation_errors.append(float(np.sqrt(np.mean(np.sum((observations-ideal)**2, axis=1)))))
                for name, delta in [('dense_projection', projected-residual),
                        ('minimum_field_energy_from_projected_averages', recovered-residual),
                        ('inverse_from_exact_averages', inverted-residual),
                        ('unobserved_component', null), ('measured_fit', measured_fit-truth),
                        ('true_all_fit', true_fit-truth)]:
                    values[name].append(float(np.mean(np.sum(delta[:, mask]**2, axis=0))))
        result['cases'][case] = dict(vector_rmse_px={k: float(np.sqrt(np.mean(v))) for k, v in values.items()},
            null_max_observation_residual_px=max(residuals),
            projected_versus_exact_observation_rms_px=float(np.sqrt(np.mean(np.square(observation_errors)))))
        print(case, result['cases'][case], flush=True)
    result['limits'] = [
        'Dense projection uses unavailable ground truth and tests basis capacity only.',
        'The null component is invisible to the fixed linear patch-average observation model, not necessarily to the full images.',
        'Minimum field energy is one reconstruction choice, not a lower bound on every estimator with a physical prior.',
        'The projected-average round trip removes basis approximation error; its small error does not establish stable recovery from exact continuous-field averages.',
        'The exact-average inverse deliberately retains every numerically nonzero singular value to expose amplification of tiny basis mismatch.',
        'All configured APs are used; frame-dependent gates and confidence are excluded from this algebraic diagnostic.',
        'Singular values refer to detector-pixel L2-normalized spline fields; rank uses floating-point precision, not a noise threshold.',
        'No deployment or real-capture improvement is demonstrated.']
    (root/'identifiability.json').write_text(json.dumps(result, indent=2)+'\n')
    Path('results/registration/ap-oracle-identifiability.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

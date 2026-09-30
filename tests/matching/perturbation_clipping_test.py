import numpy as np
import pytest
from pymatching import Matching
from perturbation_reference import probabilities


def test_clipped_native_draws_fresh_matching_and_state_law():
    h = np.array([[1,1,0],[0,1,1]], dtype=np.uint8)
    p = np.array([.5,.4,.1])
    m = Matching.from_check_matrix(h,weights=np.log((1-p)/p),error_probabilities=p,
        apply_perturbation=True,alpha=1,ensemble_size=12,seed=19,
        clip_perturbed_probabilities=True)
    shots = np.array([[0,0],[1,0],[1,1]],dtype=np.uint8)
    corrections,weights = m.decode_batch(shots,return_weights=True,shot_offset=7)
    for i,s in enumerate(shots):
        q = np.minimum(probabilities(p,1,12,19,0,7+i), .5)
        expected = np.log((1-q)/q)
        np.testing.assert_allclose(m._perturbation_weights_for_shot(7+i), expected,atol=2e-15)
        assert np.all(expected >= 0) and np.any(expected == 0)
        for j,w in enumerate(expected):
            corr,weight = Matching.from_check_matrix(h,weights=w).decode(s,return_weight=True)
            np.testing.assert_array_equal(corrections[i*12+j],corr)
            assert weights[i*12+j] == weight
    assert m.get_perturbation_state()['scheme_version'] == 2
    plain = Matching([[1]],weights=np.log(9),error_probabilities=.1,
        apply_perturbation=True,alpha=1,ensemble_size=12,seed=19)
    with pytest.raises(ValueError,match='configuration'):
        plain.set_perturbation_state(m.get_perturbation_state())


def test_all_zero_weights_clipped_ensemble_and_default_rejection():
    m = Matching([[1]],weights=0,error_probabilities=.5,apply_perturbation=True,
        alpha=1,ensemble_size=20,seed=19,clip_perturbed_probabilities=True)
    assert np.all(m._perturbation_weights_for_shot(0) >= 0)
    assert m.decode_batch([[1],[0]]).shape == (40,1)
    with pytest.raises(ValueError,match='negative weights'):
        Matching([[1]],weights=0,error_probabilities=.5,apply_perturbation=True,
            alpha=1,ensemble_size=20,seed=19)

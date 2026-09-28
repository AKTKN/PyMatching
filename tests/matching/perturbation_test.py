import numpy as np
import pytest
from scipy.sparse import csc_matrix
from pymatching import Matching
from perturbation_reference import MT64, splitmix64, probabilities


def problem(width=14, virtual=False, alpha=1, size=5, seed=17, stream=2):
    h = np.zeros((7, 14), dtype=np.uint8)
    for node in range(7):
        h[node, 2*node] = 1
        h[node, 2*node+1] = h[(node+1) % 7, 2*node+1] = 1
    p = np.linspace(.02, .2, h.shape[1])
    faults = np.zeros((width, 14), dtype=np.uint8)
    faults[:14] = np.eye(14, dtype=np.uint8)
    if width > 14:
        faults[-1, ::2] = 1
    weights = np.log((1-p)/p)
    options = dict(faults_matrix=csc_matrix(faults), use_virtual_boundary_node=virtual)
    matcher = Matching.from_check_matrix(h, weights=weights, error_probabilities=p,
        apply_perturbation=True, alpha=alpha, seed=seed, ensemble_size=size, stream_id=stream, **options)
    return h, p, options, matcher


def test_standard_rng_known_values_and_prior_formula():
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert MT64(5489).next() == 14514284786278117030
    _, p, _, matcher = problem(size=30)
    for shot in (0, 1, 123, 2**63):
        draws = probabilities(p, 1, 30, 17, 2, shot)
        expected = np.log((1-draws)/draws)
        np.testing.assert_allclose(matcher._perturbation_weights_for_shot(shot), expected, rtol=2e-15, atol=2e-15)
    assert matcher.get_perturbation_state()['shot_position'] == 0


@pytest.mark.parametrize('width', [14, 72])
@pytest.mark.parametrize('virtual', [False, True])
def test_fresh_factory_predictions_weights_and_original_graph(width, virtual):
    h, p, options, matcher = problem(width=width, virtual=virtual)
    shots = np.random.default_rng(1).integers(0, 2, (20, 7), dtype=np.uint8)
    graph_before = matcher.edges()
    builds = matcher._matching_graph.native_mwpm_build_count
    actual, actual_weights = matcher.decode_batch(shots, return_weights=True, shot_offset=12)
    assert actual.shape == (100, width)
    for shot, syndrome in enumerate(shots):
        values = matcher._perturbation_weights_for_shot(12+shot)
        for member, weights in enumerate(values):
            fresh = Matching.from_check_matrix(h, weights=weights, **options)
            correction, weight = fresh.decode(syndrome, return_weight=True)
            np.testing.assert_array_equal(actual[5*shot+member], correction)
            assert actual_weights[5*shot+member] == weight
    assert matcher.edges() == graph_before
    assert matcher._matching_graph.native_mwpm_build_count == builds == 2


def test_chunks_replay_empty_state_and_single_decode():
    shots = np.random.default_rng(2).integers(0, 2, (11, 7), dtype=np.uint8)
    whole = problem()[-1]
    expected = whole.decode_batch(shots, return_weights=True)
    chunked = problem()[-1]
    pieces = [chunked.decode_batch(shots[:3], return_weights=True),
              chunked.decode_batch(shots[3:3], return_weights=True),
              chunked.decode_batch(shots[3:], return_weights=True)]
    for index in (0, 1):
        np.testing.assert_array_equal(expected[index], np.concatenate([p[index] for p in pieces]))
    state = chunked.get_perturbation_state()
    assert state['shot_position'] == 11
    replay = chunked.decode_batch(shots, shot_offset=0)
    np.testing.assert_array_equal(replay, expected[0])
    assert chunked.get_perturbation_state() == state
    resumed = problem()[-1]
    resumed.set_perturbation_state(state)
    np.testing.assert_array_equal(chunked.decode(shots[0]), resumed.decode(shots[0]))
    assert resumed.decode_batch(shots[:0]).shape == (0, 14)


@pytest.mark.parametrize('size,alpha', [(1, 1), (4, 0)])
def test_exact_unperturbed_special_cases(size, alpha):
    h, p, options, matcher = problem(size=size, alpha=alpha)
    ordinary = Matching.from_check_matrix(h, weights=np.log((1-p)/p), **options)
    shots = np.random.default_rng(3).integers(0, 2, (10, 7), dtype=np.uint8)
    expected = ordinary.decode_batch(shots, return_weights=True)
    actual = matcher.decode_batch(shots, return_weights=True)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, np.repeat(e, size, axis=0))
    assert matcher._matching_graph.native_mwpm_build_count == 1


def test_zero_weight_ties_and_exception_recovery():
    m = Matching([[1]], weights=[0], error_probabilities=[.5], apply_perturbation=True, alpha=0, ensemble_size=3)
    np.testing.assert_array_equal(m.decode([1]), np.ones((3, 1)))
    h = [[1, 0], [1, 1], [0, 1]]
    m = Matching(h, weights=np.log(9), error_probabilities=.1, apply_perturbation=True, alpha=1, ensemble_size=3, seed=0)
    state = m.get_perturbation_state()
    with pytest.raises(ValueError, match='perfect matching'):
        m.decode_batch([[1, 0, 0]])
    assert m.get_perturbation_state() == state
    np.testing.assert_array_equal(m.decode([1, 1, 0]), [[1, 0]] * 3)


def test_unsupported_options_and_mutation():
    for h, p, alpha in [([[1, 1]], [.1, .2], 1), ([[1]], [.6], 0), ([[1]], [.3], 1)]:
        p = np.asarray(p)
        with pytest.raises(ValueError):
            Matching(h, weights=np.log((1-p)/p), error_probabilities=p, apply_perturbation=True, alpha=alpha, ensemble_size=2)
    m = problem()[-1]
    with pytest.raises(NotImplementedError):
        m.add_edge(0, 2, weight=1)
    with pytest.raises(NotImplementedError):
        m.load_from_check_matrix([[1]])
    with pytest.raises(NotImplementedError):
        m.decode_to_edges_array([0]*7)
    with pytest.raises(NotImplementedError):
        m.decode_batch([[0]*7], bit_packed_predictions=True)
    with pytest.raises(ValueError):
        m.decode_batch([[.5]*7])
    with pytest.raises(ValueError):
        m.decode_batch([[0]*7], shot_offset=2**64-1)
    with pytest.raises(ValueError):
        m.set_perturbation_state(m.get_perturbation_state() | {'scheme_version': 999})


def test_false_retains_parallel_negative_and_packed_features():
    ordinary = Matching([[1, 1]], weights=[-1, 2], apply_perturbation=False, alpha=99, ensemble_size=0)
    np.testing.assert_array_equal(ordinary.decode_batch([[1]], bit_packed_predictions=True), [[1]])
    with pytest.raises(ValueError, match='shot_offset'):
        ordinary.decode_batch([[1]], shot_offset=0)


def test_different_seeds_streams_shots_and_resolved_entropy_seed():
    original = problem()[-1]._perturbation_weights_for_shot(0)
    for variant in (problem(seed=18)[-1]._perturbation_weights_for_shot(0),
                    problem(stream=1)[-1]._perturbation_weights_for_shot(0),
                    problem()[-1]._perturbation_weights_for_shot(1)):
        np.testing.assert_array_equal(original[0], variant[0])
        assert np.any(original[1:] != variant[1:])
    m = problem(seed=None)[-1]
    resolved = problem(seed=m.get_perturbation_state()['seed'])[-1]
    np.testing.assert_array_equal(m._perturbation_weights_for_shot(0), resolved._perturbation_weights_for_shot(0))

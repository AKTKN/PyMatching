import numpy as np
import pytest
import pymatching
from pymatching.soft_output import SoftOutputConfig, metric_from_radii


def line():
    m = pymatching.Matching()
    m.add_boundary_edge(0, weight=1, fault_ids=0)
    m.add_edge(0, 1, weight=5, fault_ids=1)
    m.add_boundary_edge(1, weight=2, fault_ids=2)
    cfg = SoftOutputConfig((0, 1, -1, -1),
        ((10, 2, 0, 1.), (11, 0, 1, 5.), (12, 1, 3, 2.)), ((2, 3), (2, 1)))
    m.configure_soft_output(cfg)
    return m, cfg


def test_separate_outputs_and_shot_reset():
    m, cfg = line()
    shots = np.array([[0,0], [1,0], [0,1], [1,1], [0,0]], dtype=np.uint8)
    pred, weights = m.decode_batch(shots, return_weights=True)
    result = m.decode_batch_with_soft_output(shots, include_radii=True)
    np.testing.assert_array_equal(result.predictions, pred)
    np.testing.assert_array_equal(result.solution_weights, weights)
    assert result.soft_outputs.shape == (5, 2)
    np.testing.assert_array_equal(result.soft_outputs[0], [8, 6])
    for i, shot in enumerate(shots):
        single = m.decode_batch_with_soft_output(shot[None, :], include_radii=True)
        np.testing.assert_array_equal(single.predictions[0], pred[i])
        np.testing.assert_array_equal(single.soft_outputs[0], result.soft_outputs[i])
        np.testing.assert_array_equal(single.radii[0], result.radii[i])
        ordinary, weight = m.decode(shot, return_weight=True)
        np.testing.assert_array_equal(ordinary, pred[i])
        assert weight == weights[i]
    reverse = m.decode_batch_with_soft_output(shots[::-1])
    np.testing.assert_array_equal(reverse.soft_outputs[::-1], result.soft_outputs)
    assert reverse.radii is None
    assert m.decode_batch_with_soft_output(shots[:0]).soft_outputs.shape == (0, 2)


def test_configuration_invalidation():
    m, cfg = line()
    m.add_edge(0, 1, weight=6, merge_strategy='replace')
    with pytest.raises(ValueError, match='stale'):
        m.decode_batch_with_soft_output([[0,0]])
    m.decode_batch(np.zeros((1,2), dtype=np.uint8))
    with pytest.raises(ValueError, match='stale'):
        m.decode_batch_with_soft_output([[0,0]])
    cfg = SoftOutputConfig(cfg.node_map, ((10,2,0,1), (11,0,1,6), (12,1,3,2)), cfg.terminal_pairs)
    m.configure_soft_output(cfg)
    assert m.decode_batch_with_soft_output([[0,0]]).soft_outputs[0,0] == 9


@pytest.mark.parametrize('constructor', ['matrix', 'dem'])
def test_constructors_and_reconfiguration(constructor):
    if constructor == 'matrix':
        m = pymatching.Matching.from_check_matrix([[1,1,0], [0,1,1]], weights=[1,5,2])
    else:
        import stim
        m = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(
            'error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1'))
    with pytest.raises(ValueError, match='configuration'):
        m.decode_batch_with_soft_output([[0,0]])
    edges = tuple((i, u, 2 if v is None else v, a['weight']) for i,(u,v,a) in enumerate(m.edges()))
    cfg = SoftOutputConfig((0,1,-1), edges, ((0,2),))
    for _ in range(2):
        m.configure_soft_output(cfg)
        result = m.decode_batch_with_soft_output([[1,0]])
        pred, weight = m.decode_batch(np.array([[1,0]], dtype=np.uint8), return_weights=True)
        np.testing.assert_array_equal(result.predictions, pred)
        np.testing.assert_array_equal(result.solution_weights, weight)


def test_no_boundary_and_many_observables():
    m = pymatching.Matching()
    m.add_edge(0, 1, weight=3, fault_ids={0,65})
    m.add_edge(1, 2, weight=4, fault_ids=64)
    cfg = SoftOutputConfig((0,1,2), ((0,0,1,3), (1,1,2,4)), ((0,2),))
    m.configure_soft_output(cfg)
    shots = np.array([[1,0,1], [0,0,0]], dtype=np.uint8)
    result = m.decode_batch_with_soft_output(shots)
    pred, weight = m.decode_batch(shots, return_weights=True)
    np.testing.assert_array_equal(result.predictions, pred)
    np.testing.assert_array_equal(result.solution_weights, weight)
    with pytest.raises(ValueError, match='perfect matching'):
        m.decode_batch_with_soft_output([[1,0,0]])
    np.testing.assert_array_equal(m.decode_batch_with_soft_output(shots).soft_outputs, result.soft_outputs)


def test_parallel_analysis_edges_and_partial_coverage():
    cfg = SoftOutputConfig((-1,-1), ((100,0,1,5), (101,0,1,9)), ((0,1),))
    distances, residual = metric_from_radii(cfg, [1,2])
    assert distances == [2]
    assert residual == [2,6]


def test_reject_invalid_data():
    m, cfg = line()
    with pytest.raises(ValueError):
        m.decode_batch_with_soft_output([[2,0]])
    with pytest.raises(ValueError):
        m.configure_soft_output(SoftOutputConfig((99,), (), ((0,0),)))
    with pytest.raises(ValueError, match='duplicate'):
        m.configure_soft_output(SoftOutputConfig(cfg.node_map, cfg.edges + (cfg.edges[0],), cfg.terminal_pairs))
    with pytest.raises(ValueError, match='nonnegative'):
        metric_from_radii(SoftOutputConfig((-1,-1), ((0,0,1,-1),), ((0,1),)), [0,0])
    for value in [-1, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='nonnegative'):
            metric_from_radii(cfg, [value,0,0,0])
    m.add_edge(0,1,weight=-1,merge_strategy='replace')
    with pytest.raises(ValueError, match='nonnegative'):
        m.configure_soft_output(cfg)

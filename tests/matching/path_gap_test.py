"""Independent shortest-path, correction and signed-v1 regression oracles."""
import itertools
import networkx as nx
import numpy as np
import pymatching
import pytest
from pymatching.soft_output import SoftOutputConfig, PATH_GAP_VERSION


def fixture():
    m = pymatching.Matching()
    # Logical chain plus a disconnected syndrome component: subtraction must
    # include the latter even though no logical shortest path can overlap it.
    m.add_boundary_edge(0, weight=2)
    m.add_edge(0, 1, weight=3)
    m.add_boundary_edge(1, weight=4, fault_ids=0)
    m.add_boundary_edge(2, weight=20)
    cfg = SoftOutputConfig((0,1,2,-1,-1),
                          ((0,0,3,2),(1,0,1,3),(2,1,4,4)), ((3,4),))
    m.configure_soft_output(cfg)
    return m, cfg


def test_exhaustive_syndromes_against_simple_path_enumeration():
    m,cfg = fixture()
    shots = np.array(list(itertools.product((0,1), repeat=3)), dtype=np.uint8)
    original = shots.copy()
    edges_before = m.edges()
    ordinary,weights = m.decode_batch(shots,return_weights=True)
    swim_before = m.decode_batch_with_soft_output(shots).soft_outputs
    result = m.decode_batch_with_path_gap(shots)
    assert result.metric_version == PATH_GAP_VERSION
    np.testing.assert_array_equal(result.predictions, ordinary)
    np.testing.assert_array_equal(result.solution_weights, weights)
    hard = {(min(u,-1 if v is None else v),max(u,-1 if v is None else v)):(i,d)
            for i,(u,v,d) in enumerate(m.edges())}
    for i, shot in enumerate(shots):
        correction = m.decode_to_edges_array(shot)
        chosen = {hard[tuple(sorted((int(u),int(v))))][0] for u,v in correction}
        cost = sum(d['weight'] for j,(_,_,d) in enumerate(m.edges()) if j in chosen)
        graph = nx.Graph()
        for j,u,v,w in cfg.edges:
            graph.add_edge(u,v,weight=0 if j in chosen else w)
        distance = min(sum(graph[u][v]['weight'] for u,v in zip(path,path[1:]))
                       for path in nx.all_simple_paths(graph,3,4))
        assert result.correction_weights[i] == cost
        assert result.residual_distances[i,0] == distance
        assert result.path_gaps[i,0] == distance-cost
    assert result.path_gaps[1,0] == -11  # 9 - 20, not overlap-only 9
    np.testing.assert_array_equal(shots,original)
    assert m.edges() == edges_before
    np.testing.assert_array_equal(m.decode_batch_with_soft_output(shots).soft_outputs,swim_before)
    reverse = m.decode_batch_with_path_gap(shots[::-1])
    for field in ('predictions','solution_weights','correction_weights','residual_distances','path_gaps'):
        np.testing.assert_array_equal(getattr(result,field),getattr(reverse,field)[::-1])
    assert m.decode_batch_with_path_gap(shots[:0]).path_gaps.shape == (0,1)


def test_floating_weight_is_not_quantized_weight_and_tied_paths():
    m = pymatching.Matching()
    m.add_boundary_edge(0,weight=np.pi)
    m.add_edge(0,1,weight=np.pi)
    m.add_boundary_edge(1,weight=1.234567891,fault_ids=0)
    m.configure_soft_output(SoftOutputConfig((0,1,-1,-1),
        ((0,0,2,np.pi),(1,0,1,np.pi),(2,1,3,1.234567891)),((2,3),(3,2))))
    shots = np.array([[1,0],[0,1],[1,1],[0,0]],dtype=np.uint8)
    result = m.decode_batch_with_path_gap(shots)
    np.testing.assert_array_equal(result.path_gaps[:,0],result.path_gaps[:,1])
    assert result.correction_weights[1] == 1.234567891
    np.testing.assert_array_equal(result.predictions,m.decode_batch(shots))


@pytest.mark.parametrize('shots', [[[2,0,0]], [[0,0]], [0,0,0], [[np.nan,0,0]]])
def test_invalid_inputs(shots):
    m,_ = fixture()
    with pytest.raises(ValueError):
        m.decode_batch_with_path_gap(shots)


@pytest.mark.parametrize('edge', [(99,0,3,2),(0,0,3,5),(0,1,3,2)])
def test_invalid_original_edge_mapping(edge):
    m,_ = fixture()
    m.configure_soft_output(SoftOutputConfig((0,1,2,-1,-1),(edge,),((3,4),)))
    with pytest.raises(ValueError,match='edge|topology'):
        m.decode_batch_with_path_gap(np.zeros((1,3),dtype=np.uint8))


def test_missing_stale_negative_and_explicit_boundary():
    m,cfg = fixture()
    m.add_edge(0,1,weight=5,merge_strategy='replace')
    with pytest.raises(ValueError,match='stale'):
        m.decode_batch_with_path_gap([[0,0,0]])
    m = pymatching.Matching()
    m.add_edge(0,1,weight=-1)
    with pytest.raises(ValueError,match='nonnegative'):
        m.configure_soft_output(SoftOutputConfig((0,1),((0,0,1,1),),((0,1),)))
    m = pymatching.Matching()
    m.add_edge(0,1,weight=1)
    m.set_boundary_nodes({1})
    m.configure_soft_output(SoftOutputConfig((0,1),((0,0,1,1),),((0,1),)))
    with pytest.raises(ValueError,match='implicit'):
        m.decode_batch_with_path_gap([[0]])


def test_zero_edges_no_observables_unreachable_and_recovery():
    m = pymatching.Matching()
    m.add_edge(0,1,weight=0)
    m.add_edge(1,2,weight=2)
    m.configure_soft_output(SoftOutputConfig((0,1,2,-1),
        ((0,0,1,0),(1,1,2,2)),((0,2),(0,3))))
    result = m.decode_batch_with_path_gap([[1,0,1],[0,0,0]])
    assert result.predictions.shape == (2,0)
    np.testing.assert_array_equal(result.path_gaps[:,0],[-2,2])
    assert np.isinf(result.path_gaps[:,1]).all()
    with pytest.raises(ValueError,match='perfect matching'):
        m.decode_batch_with_path_gap([[1,0,0]])
    np.testing.assert_array_equal(m.decode_batch_with_path_gap([[1,0,1]]).path_gaps,result.path_gaps[:1])


def test_many_observables_and_merged_edge_identity():
    m = pymatching.Matching()
    m.add_boundary_edge(0,weight=4,fault_ids={0,65})
    m.add_edge(0,1,weight=8)
    m.add_edge(0,1,weight=2,merge_strategy='smallest-weight')
    m.add_boundary_edge(1,weight=3,fault_ids={64})
    m.configure_soft_output(SoftOutputConfig((0,1,-1,-1),
        ((0,0,2,4),(1,0,1,2),(2,1,3,3)),((2,3),)))
    shots = np.array([[1,0],[0,1],[1,1]],dtype=np.uint8)
    result = m.decode_batch_with_path_gap(shots)
    np.testing.assert_array_equal(result.predictions,m.decode_batch(shots))
    np.testing.assert_array_equal(result.correction_weights,[4,3,2])

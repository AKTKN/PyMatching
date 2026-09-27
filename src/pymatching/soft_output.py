"""Explicit generic metric topology and separate soft-output results."""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class SoftOutputConfig:
    """Labelled analysis graph, with no implicit physical boundary convention.

    ``node_map[u]`` is a hard-matcher node index, or -1 for an analysis-only
    vertex. ``edges`` contains (unique_id, u, v, nonnegative_weight), preserving
    parallel mechanisms. Values are in the same units as matching weights.
    ``terminal_pairs`` is ordered; the result has one column per pair.
    Supplying this configuration does not certify its physical interpretation.
    """
    node_map: tuple
    edges: tuple
    terminal_pairs: tuple

    def __post_init__(self):
        object.__setattr__(self, 'node_map', tuple(self.node_map))
        object.__setattr__(self, 'edges', tuple(tuple(e) for e in self.edges))
        object.__setattr__(self, 'terminal_pairs', tuple(tuple(p) for p in self.terminal_pairs))


@dataclass(frozen=True)
class SoftOutputResult:
    predictions: np.ndarray
    solution_weights: np.ndarray
    soft_outputs: np.ndarray
    radii: Optional[np.ndarray] = None


def metric_from_radii(config: SoftOutputConfig, radii):
    """Evaluate supplied metric balls in C++; return pair distances and edge costs.

    Validation utility independent of a decoding shot. No matching certificate
    is inferred from supplied radii. Edge costs follow configuration order.
    """
    from pymatching._cpp_pymatching import metric_from_radii as native
    return native(config.node_map, config.edges, config.terminal_pairs, radii)


PATH_GAP_VERSION = "global_subtraction_v1"


@dataclass(frozen=True)
class PathGapResult:
    """Signed heuristic with full original-weight correction subtraction.

    predictions and solution_weights are ordinary decoder products.
    correction_weights sums original floating weights on the returned XOR
    edge support (not the backend's quantized solution weight).
    residual_distances and path_gaps have shape (shots, terminal_pairs).
    No radius, clipping, absolute value or overlap-only subtraction is used.
    """
    predictions: np.ndarray
    solution_weights: np.ndarray
    correction_weights: np.ndarray
    residual_distances: np.ndarray
    path_gaps: np.ndarray
    metric_version: str = PATH_GAP_VERSION

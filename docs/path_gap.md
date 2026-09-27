# Global-subtraction path gap v1

`Matching.decode_batch_with_path_gap(shots)` returns a `PathGapResult`:
ordinary `predictions`, backend `solution_weights`, original floating
`correction_weights`, `residual_distances`, signed `path_gaps`, and
`metric_version="global_subtraction_v1"`.

Configure the terminal graph with `configure_soft_output(SoftOutputConfig(...))`.
For this API, every analysis edge ID must equal its index in `Matching.edges()`.
The analysis edge's mapped endpoints and weight must equal that original edge.
Map each resolved implicit boundary endpoint to -1. IDs, endpoints, weights,
nonnegative finite weights, binary shots and stale configuration are checked.
Explicit hard boundary nodes are currently unsupported. Edge identity refers
to the **merged ordinary graph**, not pre-merge DEM mechanisms.

For each shot, E is the XOR edge support returned by the ordinary
`decode_to_edges_array` convention. The implementation checks its syndrome and
observable against the ordinary decode. On the supplied uncontracted terminal
graph, set exactly the edges in E to zero and run nonnegative Dijkstra. Each
pair returns D(E) - W(E), with W(E) the sum of original weights over **all** E,
including correction edges omitted from the terminal graph. Do not subtract
only the path overlap. Negative results and zero residual edges are retained.
Unreachable pairs return infinity; the surface experiment rejects these.

The original floating sum is distinct from the backend quantized matching
weight, returned separately. Neither the matcher weights nor input shots are
mutated. No metric-ball radii are used. Existing SWIM APIs are unchanged.
The batch implementation currently performs ordinary prediction and correction
extraction as two deterministic solves, checking their hard-output agreement;
it is not an overhead-optimized decoder implementation.

The caller supplies the logical topology; PyMatching makes no physical logical
or gap certification. This quantity is a heuristic, not a posterior LLR or
an exact complementary gap.

The surface companion uses the same split X-boundary terminal topology as its
SWIM calculation. Z-check half-edges omitted there are omitted from the path
search too; their correction weights still enter W(E). The original merged
hard decoder graph and both CSS sectors of its correction remain unchanged.

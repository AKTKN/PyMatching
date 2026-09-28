# Native prior perturbation (version 1)

Opt-in ensembles for simple, fixed check-matrix graphs with nonnegative log-odds
weights. Ordinary matching, SWIM and path-gap APIs retain their existing
behavior when `apply_perturbation=False` (the default).

```python
import numpy as np
import pymatching

p = np.array([0.1, 0.12, 0.08])
m = pymatching.Matching.from_check_matrix(
    [[1, 1, 0], [0, 1, 1]],
    weights=np.log((1-p)/p), error_probabilities=p,
    apply_perturbation=True, alpha=1.0, seed=19,
    ensemble_size=3, stream_id=0,
)
corrections, weights = m.decode_batch([[1, 0], [0, 1]], return_weights=True)
assert corrections.shape == (2 * 3, 3)
assert weights.shape == (2 * 3,)
```

Member 0 uses the original solver and original weights. Each further member
independently samples each edge prior for that shot:
`p' = clip(p * (1 + alpha * Uniform(-1, 1)), 1e-14, 1-1e-14)`.
The returned weight is that candidate's MWPM weight, using the ordinary
PyMatching integer discretization; it is not a common-prior selection score.
Rows are shot-major: `shot * ensemble_size + member`. `decode` on one syndrome
returns `(ensemble_size, num_fault_ids)` instead of an ordinary single vector.
M=1 and alpha=0 use the original solver without clipping baseline priors.

The constructor, `from_check_matrix` and `load_from_check_matrix` accept
`apply_perturbation=False, alpha=0.0, seed=None, ensemble_size=1, stream_id=0`.
`error_probabilities` must specify the original priors and `weights` must agree
with their log odds. Alpha is in [0,1]; seed/stream IDs are unsigned 64-bit
integers. An omitted seed is resolved once and exposed by
`get_perturbation_state()`.

`decode` and `decode_batch` accept `shot_offset=None`. Without an offset, the
matcher advances its shot position across successful calls. An explicit
absolute offset replays those shots, independently of batch partition; the
cursor becomes the larger of its current position and the batch end. Empty
batches do not advance it. Failed batches do not commit a new position.
`set_perturbation_state` restores a cursor only when the stream configuration
and version match. Graph/matching objects are not serialized by this state.

Version 1 derives a shot seed as
`splitmix64(seed ^ splitmix64(shot) ^ splitmix64(stream_id + 0xD1B54A32D192ED03))`,
with unsigned 64-bit arithmetic and the standard SplitMix64 finalizer. A
standard `mt19937_64` seeded for that shot emits member-major, edge-major draws;
uniform values use `(rng() >> 11) * 2**-53`. Edge order is check-matrix column
order. Member 0 consumes no draws. This differs from the NumPy original-DEM
perturbation stream.

One work solver preserves topology and weight slots in both MatchingGraph and,
for more than 64 fault IDs, SearchGraph. Weights and the normalization constant
are updated per candidate. Queues/nodes are reset and arena blocks are recycled;
arena capacity can grow with a more complex syndrome. The pristine solver
remains available for member 0. No graph is rebuilt on the warm decode path.

Unsupported in this opt-in mode: parallel/empty check-matrix columns, self
loops, repetitions other than 1, negative/nonfinite weights, graph mutation,
packed decoding, special decoding/SO/path-gap APIs, and graph construction
from NetworkX/DEM inputs. Priors must lie in (0,0.5], and stochastic ensembles
must satisfy `p*(1+alpha) <= 0.5`. Unsupported inputs raise errors; they do not
silently fall back or clip a negative weight to zero. Ordinary mode retains
its existing support for these features.

Validation includes independent Python MT19937-64 draws, fresh graph
construction for every candidate, explicit/virtual boundaries, more than 64
fault IDs, zero weights, failed-syndrome recovery and stable graph storage.
`_perturbation_weights_for_shot` and the private backend's
`native_mwpm_build_count` are reference/test diagnostics, not simulation APIs.

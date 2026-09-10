# Gradland: code companion

Code accompanying "Gradland: On Phenomenal Experience, Differentiated Across Many Dimensions" and "Metabolic Fire: Minimal Models of Affect, Purpose and Norms".

```
jacobians.py                 input/parameter Jacobians of an nn.Module, any chosen neurons
jacobian_measures.py         transparency, cohesion, cohesive rank of a matrix
test_jacobians.py            pytest suite for jacobians.py
test_jacobian_measures.py    pytest suite for jacobian_measures.py
gradland.ipynb               worked examples + plots
```

Run `pytest` from this directory for the tests; open `gradland.ipynb` for the walkthrough.

## `jacobians.py`

A **system** is any subgraph of a computation graph with chosen input and output nodes. This module computes, for any `nn.Module`, the **input Jacobian** J_{y←x} = ∇ₓy and the **parameter Jacobian** J_{y←W} = ∇_W y of such a system.

**Core primitive:**

```python
jacobian(outputs, inputs) -> Tensor   # the n x d matrix d(outputs)/d(inputs)
```

`outputs` and `inputs` are each a tensor or list of tensors from one forward pass's autograd graph, flattened and concatenated in order. `inputs` need not be leaves — an intermediate activation or an `nn.Parameter` both work directly. One batched backward pass computes the whole matrix at once (with a per-row fallback for ops that don't support vmapped backward). Being agnostic to what "inputs"/"outputs" mean lets this one function cover input Jacobians, parameter Jacobians, layer-to-layer Jacobians, and Jacobians of *combined* systems (two nets' stacked outputs, a layer plus a loss, several timesteps concatenated).

**Picking out neurons**, via two helpers:

- `capture(model, names)` — hooks named submodules during a forward pass and stores their output tensors (still attached to the graph), so any intermediate layer's activation can be used as a Jacobian input or output.
- `select_params(model, names)` — selects `nn.Parameter`s by exact dotted name or by prefix (e.g. `"layer3"` for all of that submodule's parameters).

**Convenience wrappers**, for "the model's output w.r.t. its raw input / all parameters, optionally narrowed by name":

```python
input_jacobian(model, x, output_names=None, input_names=None)
parameter_jacobian(model, x, output_names=None, param_names=None)
```

See the module's own `__main__` block and `test_jacobians.py` for worked checks: agreement with finite differences and manual `backward()`, diagonal vs. lower-triangular time-Jacobians for an MLP vs. an RNN, and two nets sharing an input having a dense input Jacobian but a block-diagonal (zero-cohesion) parameter Jacobian.

## `jacobian_measures.py`

Two complementary measures of how a matrix's rows and columns interact (applied to Jacobians, but defined for any matrix):

| name | function | implementation |
|---|---|---|
| transparency | `transparency(A)` | alias for `effective_rank` |
| luminosity | `luminosity(A)` | `A.norm()` (Frobenius norm) |
| cohesion | `cohesion(A)` | `normalized_kirchhoff(A.abs())` |
| cohesive rank | `cohesive_rank(A)` | alias for `realized_rank` |
| cohesive luminosity | `cohesive_luminosity(A)` | `luminosity(A) * cohesion_normalized(A)` |

`cohesion`, `cohesive_rank` and `cohesive_luminosity` take `A` directly (any real entries, e.g. a signed Jacobian); `normalized_kirchhoff` itself expects an already-nonnegative matrix.

Effective rank here is the participation ratio (Σσᵢ²)²/Σσᵢ⁴. `effective_rank_trace`, `effective_rank_collision` and `effective_rank_cv` are equivalent alternate formulations, cross-checked against it in the smoke test.

Cohesion is exactly zero iff the matrix's bipartite graph is disconnected — either block-diagonal (up to permutation) or containing an all-zero row/column. The second case is the common one in practice: an inactive relu unit zeroes every one of its incoming weight-gradients, so most parameter Jacobians of relu nets have many exactly-zero columns and read as zero-cohesion by default. `drop_isolated(A)` restricts to the rows/columns that actually carry gradient, so `cohesive_rank(drop_isolated(J))` measures how the *live* circuit hangs together — see the module's docstrings for the details and `cohesion_naive` for a slow, literal reference implementation used to cross-check the fast one.

## `gradland.ipynb`

The first three sections follow "Gradland"; the fourth comes from "Metabolic Fire".

1. **Jacobians** — `input_jacobian`/`parameter_jacobian` on a small relu MLP, checked against finite differences and manual backward passes; picking out an intermediate layer as input or output.
2. **Transparency, cohesion, cohesive rank** — singular-value spectra and effective rank on rank-1/identity/random matrices; cohesion on block-diagonal vs. dense matrices; cohesive rank on identity/all-ones/random/Hadamard matrices.
3. **Worked examples** — two nets sharing an input; an MLP vs. an RNN vs. a tunable per-channel EMA cell over time; relu's dot-product identity and the KQ/VO paths of attention; gradient shattering with depth in a deep relu MLP; a block-diagonal Gram matrix from two non-interacting circuits, alongside a dense/cohesive matrix with an exactly diagonal Gram (cohesion and distinctness are independent).
4. **The metabolic channel** — an example from "Metabolic Fire".

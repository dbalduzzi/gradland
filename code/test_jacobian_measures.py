"""
Property tests for `normalized_kirchhoff` (a.k.a. `dense`).

    dense(A) = normalized_kirchhoff(A) = (tau(G) / tau(K_{n,d}))^(1/(n+d-1))

where G is the weighted bipartite graph with biadjacency matrix A and
tau(.) is the (weighted) number of spanning trees.

Claimed properties under test:

  1. dense(A) == 0  iff  there exist permutations P_n, P_d of the rows/
     columns such that P_n A P_d is block diagonal with more than one
     block (i.e. the bipartite graph of A is disconnected).
  2. |A_ij| = alpha for all i,j  =>  dense(A) = alpha, independent of n, d.
  3. |A_ij| <= 1 for all i,j     =>  dense(A) in [0, 1], independent of n, d.

`normalized_kirchhoff` requires A nonnegative with all row/col sums > 0
(see its docstring), so "|A_ij|" throughout is tested with A itself
nonnegative rather than via an absolute value.

`cohesion_naive` (jacobian_measures.py) is a separate, reference implementation
(a direct, non-log-space determinant of the *full* Laplacian) that does
*not* get this hardening -- see TestCohesionNaive below, which checks it 
agrees with `cohesion` on ordinary graphs without asserting it holds up
at the same scale.
"""
import numpy as np
import pytest
import torch
from torch import nn

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from jacobian_measures import (cohesion, cohesion_naive, cohesion_normalized,
                       cohesive_luminosity, cohesive_rank, luminosity,
                       transparency, unit_mean, drop_isolated, log_kirchhoff)
from jacobian_measures import normalized_kirchhoff as dense
from jacobians import parameter_jacobian

DTYPE = torch.float64
# Empirically calibrated separation threshold: disconnected graphs land far
# below this and connected graphs far above it, across the sizes tested here.
ZERO_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Helpers (independent of the Kirchhoff code, used as ground truth / builders)
# ---------------------------------------------------------------------------

def block_diagonal(blocks):
    """Stack nonneg blocks along the diagonal; off-diagonal blocks are zero.

    Every row/column of every block is required to have at least one
    nonzero entry so the resulting matrix has no isolated node (staying
    inside normalized_kirchhoff's documented domain) while still being
    disconnected overall whenever len(blocks) > 1.
    """
    n = sum(b.shape[0] for b in blocks)
    d = sum(b.shape[1] for b in blocks)
    A = torch.zeros(n, d, dtype=DTYPE)
    i = j = 0
    for b in blocks:
        bn, bd = b.shape
        A[i:i + bn, j:j + bd] = b
        i += bn
        j += bd
    return A


def random_block(n, d, generator, low=0.3, high=1.0):
    """A dense, strictly positive (n,d) block -> internally fully connected."""
    return torch.rand(n, d, generator=generator, dtype=DTYPE) * (high - low) + low


def random_connected(n, d, generator, p_extra=0.3, low=0.3, high=1.0):
    """Random nonneg (n,d) matrix whose bipartite graph is connected.

    Row 0 is wired to every column and column 0 to every row (a spanning
    "double star"), so the graph is connected regardless of what extra
    random edges are added on top.
    """
    A = torch.zeros(n, d, dtype=DTYPE)
    A[0, :] = torch.rand(d, generator=generator, dtype=DTYPE) * (high - low) + low
    A[:, 0] = torch.rand(n, generator=generator, dtype=DTYPE) * (high - low) + low
    extra_mask = torch.rand(n, d, generator=generator, dtype=DTYPE) < p_extra
    extra_vals = torch.rand(n, d, generator=generator, dtype=DTYPE) * (high - low) + low
    return torch.where(extra_mask, extra_vals, A)


def random_no_isolated_node(n, d, p, generator, low=0.3, high=1.0):
    """Random Bernoulli-masked nonneg matrix with no all-zero row/column.

    May or may not be connected; used to cross-check dense(A) against an
    independent connectivity test.
    """
    while True:
        mask = torch.rand(n, d, generator=generator, dtype=DTYPE) < p
        if bool((mask.sum(1) > 0).all()) and bool((mask.sum(0) > 0).all()):
            vals = torch.rand(n, d, generator=generator, dtype=DTYPE) * (high - low) + low
            return torch.where(mask, vals, torch.zeros(n, d, dtype=DTYPE))


def is_connected_bipartite(A: torch.Tensor) -> bool:
    """Ground-truth connectivity via scipy BFS/union-find, independent of
    jacobian_measures.py entirely (does not call log_kirchhoff / slogdet)."""
    Anp = A.numpy()
    n, d = Anp.shape
    edges = np.abs(Anp) > 0
    N = n + d
    adj = np.zeros((N, N), dtype=bool)
    adj[:n, n:] = edges
    adj[n:, :n] = edges.T
    n_components, _ = connected_components(csr_matrix(adj), directed=False)
    return n_components == 1


def permute(A, pn, pd):
    return A[pn][:, pd]


def gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Property 1: dense(A) == 0  iff  disconnected (up to permutation)
# ---------------------------------------------------------------------------

class TestProperty1DenseZeroIffDisconnected:

    @pytest.mark.parametrize("shapes", [
        [(2, 2), (2, 2)],
        [(2, 3), (3, 2)],
        [(1, 2), (2, 1), (2, 2)],
    ])
    def test_block_diagonal_is_near_zero(self, shapes):
        g = gen(0)
        blocks = [random_block(n, d, g) for n, d in shapes]
        A = block_diagonal(blocks)
        assert not is_connected_bipartite(A)  # sanity: our builder is disconnected
        v = dense(A).item()
        assert not np.isnan(v)
        assert v < ZERO_THRESHOLD, f"disconnected matrix should be ~0, got {v}"

    def test_block_diagonal_hidden_by_permutation_is_still_near_zero(self):
        """Directly exercises the 'there exist permutations P_n, P_d' clause:
        shuffle rows/columns so the block structure is no longer visually
        block-diagonal, and confirm dense(.) is unchanged (still ~0)."""
        g = gen(1)
        blocks = [random_block(3, 3, g), random_block(2, 3, g)]
        A = block_diagonal(blocks)
        n, d = A.shape
        pn, pd = torch.randperm(n, generator=g), torch.randperm(d, generator=g)
        Ap = permute(A, pn, pd)
        assert not is_connected_bipartite(Ap)
        # Not asserting v_orig == v_perm to tight precision: near a singular
        # determinant the condition number is huge, so reordering the
        # arithmetic changes which floating-point noise comes out, even
        # though both are noise around the true value 0.
        assert dense(A).item() < ZERO_THRESHOLD
        assert dense(Ap).item() < ZERO_THRESHOLD

    @pytest.mark.parametrize("n,d", [(4, 3), (5, 5)])
    def test_permutation_invariance_connected(self, n, d):
        """dense(.) only depends on A up to row/column relabeling."""
        g = gen(2)
        A = random_connected(n, d, g)
        base = dense(A).item()
        for _ in range(5):
            pn, pd = torch.randperm(n, generator=g), torch.randperm(d, generator=g)
            v = dense(permute(A, pn, pd)).item()
            assert v == pytest.approx(base, abs=1e-9)

    @pytest.mark.parametrize("n,d", [(3, 4), (5, 4), (6, 6)])
    def test_connected_is_bounded_away_from_zero(self, n, d):
        g = gen(3)
        for _ in range(5):
            A = random_connected(n, d, g, p_extra=torch.rand(1, generator=g).item() * 0.5)
            assert is_connected_bipartite(A)
            v = dense(A).item()
            assert v > ZERO_THRESHOLD, f"connected matrix should be bounded away from 0, got {v}"

    def test_matches_independent_connectivity_check(self):
        """Cross-check sign of (dense(A) > threshold) against an independent
        (scipy-BFS-based) connectivity check over many random small graphs,
        with no isolated nodes (staying in normalized_kirchhoff's domain)."""
        g = gen(4)
        rng = np.random.default_rng(4)
        n_checked = 0
        for _ in range(100):
            n, d = int(rng.integers(2, 7)), int(rng.integers(2, 7))
            p = float(rng.uniform(0.25, 0.7))
            A = random_no_isolated_node(n, d, p, g)
            v = dense(A).item()
            if np.isnan(v):
                continue  # only happens outside the documented domain; none expected here
            n_checked += 1
            truth = is_connected_bipartite(A)
            predicted = v > ZERO_THRESHOLD
            assert predicted == truth, (
                f"n={n} d={d} p={p:.2f} dense={v} truth_connected={truth}"
            )
        assert n_checked > 80  # make sure the test actually exercised many cases

    @pytest.mark.parametrize("k", [4, 20, 80])
    def test_large_disconnected_graphs_give_exact_zero(self, k):
        """`log_kirchhoff` returns an honest -inf for log|det| on a singular
        reduced Laplacian, so `dense` is exactly 0.0 for a disconnected
        graph -- not just small -- at every size tested here, well beyond
        the small/medium regime `ZERO_THRESHOLD`-based tests target."""
        g = gen(5)
        blocks = [random_block(k, k, g), random_block(k, k, g)]
        A = block_diagonal(blocks)
        assert not is_connected_bipartite(A)
        v = dense(A).item()
        assert v == 0.0, f"k={k}: expected exact 0, got {v}"


# ---------------------------------------------------------------------------
# Property 2: |A_ij| = alpha for all i,j  =>  dense(A) = alpha
# ---------------------------------------------------------------------------

class TestProperty2UniformMagnitude:

    @pytest.mark.parametrize("alpha", [0.001, 1.0, 1000.0])
    @pytest.mark.parametrize("n,d", [(2, 2), (3, 5), (6, 2), (7, 7)])
    def test_uniform_magnitude_equals_alpha_regardless_of_shape(self, alpha, n, d):
        A = alpha * torch.ones(n, d, dtype=DTYPE)
        v = dense(A).item()
        assert v == pytest.approx(alpha, rel=1e-5)


# ---------------------------------------------------------------------------
# Property 3: |A_ij| <= 1 for all i,j  =>  dense(A) in [0, 1]
# ---------------------------------------------------------------------------

class TestProperty3BoundedEntries:

    @pytest.mark.parametrize("n,d", [(3, 4), (5, 5), (7, 4)])
    def test_random_bounded_entries_stay_in_unit_interval(self, n, d):
        g = gen(6)
        for _ in range(10):
            A = torch.rand(n, d, generator=g, dtype=DTYPE)  # entries in [0, 1)
            v = dense(A).item()
            assert not np.isnan(v)
            assert 0.0 <= v <= 1.0 + 1e-6, f"n={n} d={d} dense={v} out of [0,1]"

    def test_upper_bound_is_tight_at_all_ones(self):
        for n, d in [(2, 2), (5, 3), (7, 7)]:
            A = torch.ones(n, d, dtype=DTYPE)
            assert dense(A).item() == pytest.approx(1.0, abs=1e-6)

    def test_lower_bound_is_approached_by_near_disconnected_graphs(self):
        """As a bridge edge linking two dense clusters shrinks to 0, dense(A)
        should decrease towards 0 -- exercising the lower end of [0, 1]."""
        g = gen(7)
        b1 = random_block(3, 3, g, low=0.5, high=1.0)
        b2 = random_block(3, 3, g, low=0.5, high=1.0)
        A = block_diagonal([b1, b2])
        prev = None
        for bridge in (0.5, 0.1, 0.01, 0.001):
            Ab = A.clone()
            Ab[0, 3] = bridge  # connect block 1's row 0 to block 2's first column
            v = dense(Ab).item()
            assert 0.0 <= v <= 1.0 + 1e-6
            if prev is not None:
                assert v <= prev + 1e-9, "dense(.) should shrink as the bridge weakens"
            prev = v
        assert prev < 0.3  # comfortably smaller than a well-connected graph


# ---------------------------------------------------------------------------
# cohesion_naive: the literal reference implementation.
# Checks that it agrees with `cohesion` on the graphs it's expected to
# handle -- small/ordinary ones -- not at the sizes
# test_large_disconnected_graphs_give_exact_zero above exercises.
# ---------------------------------------------------------------------------

class TestCohesionNaive:

    @pytest.mark.parametrize("n,d", [(3, 4), (5, 5), (7, 4)])
    def test_agrees_with_cohesion_on_connected_graphs(self, n, d):
        g = gen(20)
        for _ in range(10):
            A = torch.randn(n, d, generator=g, dtype=DTYPE)  # signed, like a Jacobian
            fast, ref = cohesion(A).item(), cohesion_naive(A).item()
            assert fast == pytest.approx(ref, rel=1e-6, abs=1e-9), (
                f"n={n} d={d}: cohesion={fast} cohesion_naive={ref}"
            )

    def test_block_diagonal_is_near_zero_at_small_scale(self):
        g = gen(21)
        blocks = [random_block(2, 3, g), random_block(3, 2, g)]
        A = block_diagonal(blocks)
        assert not is_connected_bipartite(A)
        v = cohesion_naive(A).item()
        assert not np.isnan(v)
        assert v < ZERO_THRESHOLD, f"disconnected matrix should be ~0, got {v}"


# ---------------------------------------------------------------------------
# Unit-mean normalization: A~ := A / mean|A_ij| (kirchhoff.unit_mean).
#
# The paper normalizes by the *mean* magnitude, not the max, before feeding
# cohesion into cohesive rank / cohesive luminosity. The mean is the tightest
# divisor for which dense(A~) <= 1 still holds, so these tests pin down both
# halves of that: the bound holds, and it is tight exactly at uniform
# magnitude. Everything downstream (dense_erk(A) <= rk(A), Hadamard attaining
# n) rests on it.
# ---------------------------------------------------------------------------

class TestUnitMeanNormalization:

    @pytest.mark.parametrize("n,d", [(3, 4), (5, 5), (8, 8)])
    def test_dense_of_A_is_at_most_mean_magnitude(self, n, d):
        """dense(A) <= mean|A_ij|, i.e. dense(A~) <= 1 -- the property that
        makes the mean a legitimate normalizer at all. Exercised across
        uniform, heavy-tailed, log-normal and spiked magnitude profiles."""
        g = gen(30)
        for _ in range(10):
            style = int(torch.randint(0, 4, (1,), generator=g))
            if style == 0:
                A = torch.rand(n, d, generator=g, dtype=DTYPE)
            elif style == 1:
                A = torch.rand(n, d, generator=g, dtype=DTYPE) ** 8   # heavy-tailed
            elif style == 2:
                A = torch.randn(n, d, generator=g, dtype=DTYPE).exp()  # log-normal
            else:
                A = torch.full((n, d), 1e-4, dtype=DTYPE)              # one huge entry
                A[0, 0] = 1e4
            assert cohesion_normalized(A).item() <= 1.0 + 1e-9

    @pytest.mark.parametrize("alpha", [0.001, 13.7])
    @pytest.mark.parametrize("n,d", [(2, 2), (6, 2), (9, 2)])
    def test_bound_is_tight_exactly_at_uniform_magnitude(self, alpha, n, d):
        """dense(A~) == 1 when every |A_ij| is alpha, for any alpha, n, d --
        and strictly below 1 as soon as the magnitudes differ."""
        A = alpha * torch.ones(n, d, dtype=DTYPE)
        assert cohesion_normalized(A).item() == pytest.approx(1.0, abs=1e-9)
        # signs don't matter, only magnitudes
        signs = torch.where(torch.rand(n, d, generator=gen(31), dtype=DTYPE) < 0.5,
                            torch.tensor(-1.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE))
        assert cohesion_normalized(alpha * signs).item() == pytest.approx(1.0, abs=1e-9)
        # perturbing one entry's magnitude strictly lowers it
        B = A.clone()
        B[0, 0] = alpha * 2.0
        assert cohesion_normalized(B).item() < 1.0 - 1e-6

    @pytest.mark.parametrize("n,d", [(4, 4), (6, 5)])
    def test_scale_invariance(self, n, d):
        """dense(A~) is scale *invariant* where dense(A) is scale equivariant."""
        g = gen(32)
        A = torch.randn(n, d, generator=g, dtype=DTYPE)
        base = cohesion_normalized(A).item()
        for c in (1e-6, 3.0, 1e6):
            assert cohesion_normalized(c * A).item() == pytest.approx(base, rel=1e-9)

    def test_dividing_before_or_after_agrees(self):
        """dense is 1-homogeneous, so dense(A/mean) == dense(A)/mean. Guards
        against the two being silently different implementations."""
        g = gen(33)
        A = torch.randn(4, 4, generator=g, dtype=DTYPE)
        assert cohesion_normalized(A).item() == pytest.approx(
            (cohesion(A) / A.abs().mean()).item(), rel=1e-9)

    def test_mean_is_strictly_tighter_than_max(self):
        """The normalizer actually changed something: on a non-uniform matrix
        the mean-normalized reading is strictly larger than the old
        max-normalized one, and they coincide only at uniform magnitude."""
        g = gen(34)
        A = torch.randn(5, 5, generator=g, dtype=DTYPE)
        by_mean = cohesion(A / A.abs().mean()).item()
        by_max = cohesion(A / A.abs().max()).item()
        assert by_mean > by_max + 1e-6
        U = torch.ones(5, 5, dtype=DTYPE)
        assert cohesion(U / U.mean()).item() == pytest.approx(cohesion(U / U.max()).item())

    def test_unit_mean_helper(self):
        g = gen(35)
        A = torch.randn(4, 6, generator=g, dtype=DTYPE)
        T = unit_mean(A)
        assert bool((T >= 0).all())
        assert T.mean().item() == pytest.approx(1.0, rel=1e-12)
        assert torch.allclose(unit_mean(torch.zeros(3, 3, dtype=DTYPE)),
                              torch.zeros(3, 3, dtype=DTYPE))


# ---------------------------------------------------------------------------
# Cohesive rank / cohesive luminosity, which is where the unit-mean
# normalization is actually consumed.
# ---------------------------------------------------------------------------

class TestCohesiveRank:

    def test_hadamard_attains_the_bound_n(self):
        """dense_erk(A) = n for an n x n Hadamard matrix: equal magnitudes
        (dense(A~)=1) and equal singular values (erk=n)."""
        H = torch.tensor([[1., 1, 1, 1], [1, -1, 1, -1],
                          [1, 1, -1, -1], [1, -1, -1, 1]], dtype=DTYPE)
        assert cohesive_rank(H).item() == pytest.approx(4.0, abs=1e-9)
        H8 = torch.tensor(
            [[1. if bin(i & j).count("1") % 2 == 0 else -1. for j in range(8)]
             for i in range(8)], dtype=DTYPE)
        assert cohesive_rank(H8).item() == pytest.approx(8.0, abs=1e-8)

    @pytest.mark.parametrize("n,d", [(4, 4), (6, 3)])
    def test_bounded_by_rank(self, n, d):
        g = gen(36)
        for _ in range(20):
            A = torch.randn(n, d, generator=g, dtype=DTYPE)
            assert 0.0 <= cohesive_rank(A).item() <= min(n, d) + 1e-9

    def test_no_random_matrix_beats_hadamard(self):
        g = gen(37)
        best = max(cohesive_rank(torch.randn(4, 4, generator=g, dtype=DTYPE)).item()
                   for _ in range(500))
        assert best < 4.0

    def test_zero_at_the_two_degenerate_cases(self):
        assert cohesive_rank(torch.zeros(4, 3, dtype=DTYPE)).item() == 0.0
        assert cohesive_rank(torch.eye(5, dtype=DTYPE)).item() == 0.0      # disconnected
        blocks = block_diagonal([random_block(3, 3, gen(38)), random_block(3, 3, gen(39))])
        assert cohesive_rank(blocks).item() == pytest.approx(0.0, abs=1e-9)

    def test_identity_and_all_ones_are_the_two_extremes(self):
        """Identity maximizes transparency with zero cohesion; all-ones
        maximizes cohesion with transparency 1. Neither gets a high
        cohesive rank."""
        n = 6
        I, J = torch.eye(n, dtype=DTYPE), torch.ones(n, n, dtype=DTYPE)
        assert transparency(I).item() == pytest.approx(float(n))
        assert cohesion_normalized(I).item() == pytest.approx(0.0, abs=1e-12)
        assert transparency(J).item() == pytest.approx(1.0)
        assert cohesion_normalized(J).item() == pytest.approx(1.0, abs=1e-9)
        assert cohesive_rank(I).item() == pytest.approx(0.0, abs=1e-12)
        assert cohesive_rank(J).item() == pytest.approx(1.0, abs=1e-9)

    def test_cohesive_rank_is_scale_invariant_luminosity_equivariant(self):
        g = gen(40)
        A = torch.randn(5, 5, generator=g, dtype=DTYPE)
        for c in (0.01, 7.0, 1e5):
            assert cohesive_rank(c * A).item() == pytest.approx(cohesive_rank(A).item(), rel=1e-9)
            assert cohesive_luminosity(c * A).item() == pytest.approx(
                c * cohesive_luminosity(A).item(), rel=1e-9)

    def test_the_two_share_one_normalization(self):
        """dense_erk and dense_lum must use the *same* dense(A~), so their
        ratio is exactly erk(A)/lum(A). This is the assertion that would have
        caught a half-applied max->mean edit."""
        g = gen(41)
        A = torch.randn(5, 5, generator=g, dtype=DTYPE)
        assert (cohesive_rank(A) / cohesive_luminosity(A)).item() == pytest.approx(
            (transparency(A) / luminosity(A)).item(), rel=1e-9)

    def test_zero_iff_cohesive_luminosity_zero(self):
        g = gen(42)
        for A in [torch.zeros(3, 3, dtype=DTYPE), torch.eye(4, dtype=DTYPE),
                  torch.randn(4, 4, generator=g, dtype=DTYPE)]:
            assert (cohesive_rank(A).item() == 0) == (cohesive_luminosity(A).item() == 0)

# ---------------------------------------------------------------------------
# Isolated nodes: an all-zero row or column disconnects the bipartite graph
# just as a block structure does. These are regression tests for a crash --
# the Schur complement in `log_kirchhoff` divides by the node degrees, so a
# zero degree produced inf/nan and made `eigvalsh` raise `_LinAlgError`
# instead of returning 0. The builders above deliberately exclude isolated
# nodes, so nothing else in this file covers it, and it is the *typical*
# case for a relu net's parameter Jacobian.
# ---------------------------------------------------------------------------

def _with_zero_col(n, d, j, seed=0):
    A = torch.randn(n, d, generator=torch.Generator().manual_seed(seed), dtype=DTYPE)
    A[:, j] = 0.0
    return A


def _with_zero_row(n, d, i, seed=0):
    A = torch.randn(n, d, generator=torch.Generator().manual_seed(seed), dtype=DTYPE)
    A[i, :] = 0.0
    return A


class TestIsolatedNodes:

    # Both orientations matter: log_kirchhoff eliminates the smaller
    # partition, so n<d and n>d take different code paths and a guard on one
    # of them alone would leave the other crashing.
    @pytest.mark.parametrize("n,d", [(4, 5), (5, 4), (2, 9), (1, 6)])
    def test_zero_column_gives_exact_zero(self, n, d):
        A = _with_zero_col(n, d, j=d // 2)
        assert cohesion(A).item() == 0.0
        assert cohesive_rank(A).item() == 0.0
        assert cohesive_luminosity(A).item() == 0.0

    @pytest.mark.parametrize("n,d", [(4, 5), (5, 4), (9, 2), (6, 1)])
    def test_zero_row_gives_exact_zero(self, n, d):
        A = _with_zero_row(n, d, i=n // 2)
        assert cohesion(A).item() == 0.0
        assert cohesive_rank(A).item() == 0.0
        assert cohesive_luminosity(A).item() == 0.0

    def test_all_zero_matrix(self):
        A = torch.zeros(4, 5, dtype=DTYPE)
        assert cohesion(A).item() == 0.0
        assert cohesive_rank(A).item() == 0.0
        assert transparency(A).item() == 0.0

    @pytest.mark.parametrize("n,d", [(4, 5), (5, 4), (3, 7)])
    def test_agrees_with_cohesion_naive(self, n, d):
        """`cohesion_naive` handled these correctly all along (it builds the
        full Laplacian and never divides by a degree). It is the reference."""
        for A in (_with_zero_col(n, d, j=1), _with_zero_row(n, d, i=1)):
            assert cohesion(A).item() == pytest.approx(cohesion_naive(A).item(), abs=1e-12)

    def test_log_kirchhoff_is_minus_inf_not_nan(self):
        A = _with_zero_col(4, 5, j=2).abs()
        lk = log_kirchhoff(A)
        assert torch.isinf(lk) and lk.item() < 0
        assert not torch.isnan(lk)

    def test_agrees_with_independent_connectivity_check(self):
        for n, d, mk in [(4, 5, _with_zero_col), (5, 4, _with_zero_row)]:
            A = mk(n, d, 1)
            assert not is_connected_bipartite(A)
            assert cohesion(A).item() == 0.0

    def test_empty_dimension(self):
        assert cohesion(torch.zeros(0, 4, dtype=DTYPE)).item() == 0.0
        assert cohesion(torch.zeros(4, 0, dtype=DTYPE)).item() == 0.0


class TestDropIsolated:

    def test_removes_dead_rows_and_columns(self):
        A = torch.randn(4, 6, generator=torch.Generator().manual_seed(3), dtype=DTYPE)
        A[:, [1, 4]] = 0.0
        A[2, :] = 0.0
        live = drop_isolated(A)
        assert live.shape == (3, 4)
        assert (live.abs().sum(0) > 0).all() and (live.abs().sum(1) > 0).all()

    def test_is_a_no_op_when_nothing_is_isolated(self):
        A = torch.randn(5, 5, generator=torch.Generator().manual_seed(4), dtype=DTYPE)
        assert torch.equal(drop_isolated(A), A)

    def test_recovers_a_nonzero_reading(self):
        A = torch.randn(4, 6, generator=torch.Generator().manual_seed(5), dtype=DTYPE)
        A[:, [1, 4]] = 0.0
        assert cohesive_rank(A).item() == 0.0
        assert cohesive_rank(drop_isolated(A)).item() > 0.0

    def test_all_zero_matrix_gives_empty(self):
        assert drop_isolated(torch.zeros(3, 3, dtype=DTYPE)).numel() == 0

    def test_relu_parameter_jacobian_is_the_motivating_case(self):
        """The whole point: on an ordinary relu MLP, most parameter-Jacobian
        columns are exactly zero, so the raw reading is 0 and only the live
        subgraph says anything."""
        torch.manual_seed(0)
        mlp = nn.Sequential(nn.Linear(8, 16), nn.ReLU(),
                            nn.Linear(16, 16), nn.ReLU(),
                            nn.Linear(16, 4)).double()
        x = torch.randn(8, dtype=DTYPE)
        Jw = parameter_jacobian(mlp, x)
        assert (Jw.abs().sum(0) == 0).any(), "expected some dead units"
        assert cohesive_rank(Jw).item() == 0.0
        assert cohesive_rank(drop_isolated(Jw)).item() > 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

"""
Kirchhoff complexity of a bipartite graph.

Given an n×d nonneg matrix A, constructs the (n+d)×(n+d) Laplacian
    L = [ diag(A·1)   -A  ]
        [   -Aᵀ    diag(Aᵀ·1) ]
and returns log|det(cofactor)| via Schur complement.

Complexity: O(n·d·min(n,d) + min(n,d)³)  vs naive O((n+d)³).

Two versions of cohesion are provided (see `cohesion` and `cohesion_naive`
below): a fast, numerically robust one for general use, and a simple,
literal one for cross-checking the fast version.
"""

import math

import torch
from torch import Tensor


def _reduced_laplacian_logdet(S: Tensor, scale: Tensor) -> Tensor:
    """log|det(S)| for the symmetric Schur-complement matrix S built below
    (SPD iff the underlying graph is connected), computed from S's
    eigenvalues rather than via LU-based `slogdet`.

    Eigenvalues of a symmetric matrix are backward stable, unlike a general
    determinant, whose relative error on a near-singular matrix can blow
    up. A near-disconnected graph makes S near-singular, so its smallest
    eigenvalue reliably flags disconnection: below tolerance means
    disconnected, giving an exact -inf (hence `dense(A)=0`) instead of
    leftover floating-point noise.

    `scale` (the diagonal S was built from) sets the tolerance instead of
    S's own eigenvalues, because S can be as small as 1x1 when min(n,d) is
    small -- and a 1x1 matrix's only eigenvalue *is* its own max, so a
    cancellation-corrupted eigenvalue would set its own tolerance and never
    get caught. When min(n,d)=1, S is 0x0: log|det| of an empty matrix is 0
    by convention (a single row or column is always connected).
    """
    if S.numel() == 0:
        return S.new_zeros(())
    eigs = torch.linalg.eigvalsh(S)
    tol = 10 * S.shape[-1] * torch.finfo(S.dtype).eps * scale.abs().max()
    if bool((eigs < tol).any()):
        return S.new_full((), float("-inf"))
    return eigs.clamp_min(tol).log().sum()


def log_kirchhoff(A: Tensor) -> Tensor:
    """Log Kirchhoff complexity (log number of spanning trees).

    Delete the last node in the *smaller* partition so the Schur complement
    S is (min(n,d)-1) × (min(n,d)-1) and symmetric (SPD iff the graph is
    connected); log|det(S)| is then read off S's eigenvalues rather than
    computed via a direct/LU-based determinant, so that near-disconnected
    graphs degrade gracefully instead of leaking floating-point noise into
    `normalized_kirchhoff`'s root-and-normalize step (see
    `_reduced_laplacian_logdet`). For a simpler, literal version -- which
    computes a direct determinant, and is correspondingly more fragile at
    scale -- see `cohesion_naive`.

    Args:
        A: (n, d) nonneg weight matrix. Any input is accepted: an all-zero
           row or column is an isolated node, hence a disconnected graph,
           and returns -inf rather than raising.

    Returns:
        Scalar tensor: log|det(cofactor of L)|, or -inf if A's bipartite
        graph is disconnected -- whether by block structure or by an
        isolated node.
    """
    n, d = A.shape
    if n == 0 or d == 0:
        # No nodes on one side: there is no spanning tree of K_{n,d}.
        return A.new_full((), float("-inf"))

    dr = A.sum(1)   # (n,)  row degrees
    dc = A.sum(0)   # (d,)  col degrees

    # An all-zero row or column is an *isolated node* in the bipartite graph,
    # which disconnects it just as surely as a block structure does -- so the
    # honest answer is tau(G) = 0, i.e. log tau = -inf, i.e. dense(A) = 0.
    #
    # This case is not exotic: an inactive relu contributes an exactly-zero
    # gradient for every one of its incoming weights and its bias, so
    # parameter Jacobians of relu nets routinely have hundreds of zero
    # columns. It must be caught *before* the Schur complement below, which
    # divides by the degrees and would otherwise produce inf/nan and make
    # `torch.linalg.eigvalsh` raise `_LinAlgError`.
    if bool((dr <= 0).any()) or bool((dc <= 0).any()):
        return A.new_full((), float("-inf"))

    if d <= n:
        # Delete last col-node.  Schur complement over D_r (n×n diagonal).
        B = A[:, :-1]                              # n × (d-1)
        # S = D_c[:-1] - Bᵀ D_r⁻¹ B
        S = torch.diag(dc[:-1]) - B.T @ (B * dr.reciprocal().unsqueeze(1))
        return dr.log().sum() + _reduced_laplacian_logdet(S, dc[:-1])
    else:
        # Delete first row-node.  Schur complement over D_c (d×d diagonal).
        B = A[1:, :]                               # (n-1) × d
        # S = D_r[1:] - B D_c⁻¹ Bᵀ
        S = torch.diag(dr[1:]) - (B * dc.reciprocal().unsqueeze(0)) @ B.T
        return dc.log().sum() + _reduced_laplacian_logdet(S, dr[1:])


def normalized_kirchhoff(A: Tensor) -> Tensor:
    """Scale-invariant Kirchhoff complexity normalized by τ(K_{n,d}).

    Equals (τ(G) / τ(K_{n,d}))^(1/(n+d-1)), which is:
      - 1 for the all-ones matrix
      - symmetric in n and d
      - positively homogeneous of degree 1: nk(c*A) = c*nk(A) for c > 0
      - bounded above by A's mean entry: nk(A) <= A.mean(), with equality
        iff every entry of A is the same (see `unit_mean` below)
    """
    n, d = A.shape
    if n == 0 or d == 0:
        # No spanning tree, and log tau(K_{n,d}) is undefined; short-circuit
        # before the math.log below.
        return A.new_zeros(())
    lk = log_kirchhoff(A)
    # log tau(K_{n,d}) = (d-1) log n + (n-1) log d. Computed with `math.log`
    # in Python's float (=float64) rather than `torch.tensor(float(n)).log()`,
    # which would build a *float32* tensor under the default dtype and cap the
    # precision of the result at ~1e-8 even for a float64 input matrix.
    lk = lk - ((d - 1) * math.log(n) + (n - 1) * math.log(d))
    return (lk / (n + d - 1)).exp()


def unit_mean(A: Tensor) -> Tensor:
    """|A| rescaled to unit mean magnitude: A~ := A / mean_{ij}|A_ij|.

    Returns a matrix of zeros when A is zero (nothing to normalize).
    """
    absA = A.abs()
    m = absA.mean()
    if m == 0:
        return absA
    return absA / m


def drop_isolated(A: Tensor) -> Tensor:
    """A restricted to its rows and columns that are not identically zero.

    Cohesion is exactly zero whenever the bipartite graph has an isolated
    node, and a single dead unit is enough to produce one. For a relu net's
    parameter Jacobian that is the normal case, not a corner case: every
    inactive unit zeroes the columns for all of its incoming weights and its
    bias, so `cohesion(J)` on the raw, maximalist parameter Jacobian is 0 for
    essentially every net at essentially every input.

    Reporting 0 there is correct but uninformative -- it says the *chosen*
    system is disconnected, not that the live part of it is. This helper
    restricts to the subgraph that is actually carrying gradient, so that
    `cohesion(drop_isolated(J))` measures how the active circuit hangs
    together. That is a modelling decision (a narrower choice of system),
    not a numerical fix, so it is deliberately opt-in: nothing in this
    module calls it for you.

    Returns an empty (0, 0) tensor if every entry of A is zero.
    """
    absA = A.abs()
    rows = absA.sum(1) > 0
    cols = absA.sum(0) > 0
    return A[rows][:, cols]


def cohesion_normalized(A: Tensor) -> Tensor:
    """dense(A~): cohesion of A after rescaling to unit mean magnitude.

    Lies in [0, 1]: zero iff A's bipartite graph is disconnected, one iff
    every |A_ij| is the same. Scale invariant, unlike `cohesion`.
    """
    if A.abs().max() == 0:
        return A.new_zeros(())
    return normalized_kirchhoff(unit_mean(A))


def _kirchhoff_naive(A: Tensor) -> Tensor:
    """det(L'(A)): the raw (unnormalized) Kirchhoff complexity, computed by
    building the full (n+d)x(n+d) Laplacian directly and taking the
    determinant of a cofactor (delete row 0, col 0) -- Kirchhoff's matrix
    tree theorem applied literally, with no numerical safeguards. O((n+d)^3).
    """
    n, d = A.shape
    N = n + d
    L = torch.zeros(N, N, dtype=A.dtype, device=A.device)
    L[:n, n:] = -A
    L[n:, :n] = -A.T
    L[:n, :n] = torch.diag(A.sum(1))
    L[n:, n:] = torch.diag(A.sum(0))
    M = L[1:, 1:]
    return torch.linalg.det(M)


def cohesion_naive(A: Tensor) -> Tensor:
    """Reference cohesion, computed as literally as possible:

        dense(A) := ( d^{1-n} n^{1-d} * det(L'(|A|)) )^{1/(n+d-1)}

    where det(L'(|A|)) is, by Kirchhoff's matrix tree theorem, the sum over
    all spanning trees of the bipartite graph of the product of edge
    weights |A_ij| in each tree.

    No other numerical safeguards: builds the full (n+d)x(n+d) Laplacian and
    calls `torch.linalg.det` directly (not in log-space, not via a reduced
    Schur complement), so it is O((n+d)^3) rather than `cohesion`'s
    O(min(n,d)^3) and can overflow, underflow, or return spurious noise on
    larger or near-disconnected graphs where `cohesion` remains reliable.
    Use it as a small, easily-audited cross-check of `cohesion` -- not for
    general use.
    """
    absA = A.abs()
    n, d = absA.shape
    tau = _kirchhoff_naive(absA)
    normalizer = float(n) ** (d - 1) * float(d) ** (n - 1)
    return (tau.clamp_min(0) / normalizer) ** (1.0 / (n + d - 1))


def _nonzero_singular_values(A: Tensor) -> Tensor:
    """Singular values of A above a rank-revealing numerical tolerance."""
    s = torch.linalg.svdvals(A)
    eps = torch.finfo(s.dtype).eps
    tol = max(A.shape) * eps * s.max() if s.numel() > 0 else 0.0
    return s[s > tol]


def effective_rank(A: Tensor) -> Tensor:
    """Effective rank erk(A) = (Σσᵢ²)² / Σσᵢ⁴, or 0 when rank(A) = 0.

    Equals 1 for a rank-1 matrix (all weight on one singular value) and
    rank(A) when all nonzero singular values are equal. Lies in
    [1, rank(A)] whenever rank(A) > 0.
    """
    s = _nonzero_singular_values(A)
    if s.numel() == 0:
        return A.new_zeros(())
    ss = s.square()
    return ss.sum().square() / ss.square().sum()


def effective_rank_trace(A: Tensor) -> Tensor:
    """Effective rank as a ratio of traces: tr(AAᵀ)² / tr((AAᵀ)²).

    Equivalent to effective_rank(A). AAᵀ can be replaced with AᵀA without
    changing the result; the smaller of the two Gram matrices is used here.
    """
    n, d = A.shape
    G = A @ A.T if n <= d else A.T @ A
    tr = torch.trace(G)
    if tr == 0:
        return A.new_zeros(())
    tr_sq = torch.trace(G @ G)
    return tr.square() / tr_sq


def effective_rank_collision(A: Tensor) -> Tensor:
    """Effective rank as inverse collision probability of p_i = σᵢ²/Σⱼσⱼ².

    Equivalent to effective_rank(A).
    """
    s = _nonzero_singular_values(A)
    if s.numel() == 0:
        return A.new_zeros(())
    p = s.square()
    p = p / p.sum()
    collision_prob = p.square().sum()
    return 1.0 / collision_prob


def effective_rank_cv(A: Tensor) -> Tensor:
    """Effective rank via the coefficient of variation of squared singular
    values: erk(A) = r / (1 + (s/μ)²), where μ, s² are the mean and
    variance of σ₁²,...,σᵣ². Equivalent to effective_rank(A).
    """
    s = _nonzero_singular_values(A)
    r = s.numel()
    if r == 0:
        return A.new_zeros(())
    x = s.square()
    mu = x.mean()
    var = x.var(unbiased=False)
    cv_sq = var / mu.square()
    return r / (1 + cv_sq)


def realized_rank(A: Tensor, eps: float = 0.0) -> Tensor:
    """Realized rank ν = ρ · κ.

      ρ = effective_rank(A)        -- spectral spread, in [1, rank(A)]
      κ = cohesion_normalized(A)   -- graph connectivity, in [0, 1]
      ν = ρ · κ                    -- in [0, rank(A)]

    A may have arbitrary (including negative) entries; Kirchhoff is computed
    on |A| rescaled to unit mean magnitude (`unit_mean`). Maximized by
    orthogonal designs (equal-magnitude entries, uniform singular values),
    e.g. Hadamard matrices, which attain ν = n exactly for n x n.
    Returns 0 for the zero matrix, for any block-diagonal (up to
    permutation) matrix, and for any matrix with an all-zero row or column
    (an isolated node also disconnects the graph -- see `drop_isolated`).

    `eps` adds a floor to every |A_ij| before the Kirchhoff step, which
    connects an otherwise-disconnected graph; use it only to smooth a
    hard zero, not by default. It is a poor substitute for choosing the
    system properly: on a relu parameter Jacobian with many dead columns,
    eps=1e-12 returns ~1e-6 rather than anything interpretable, because
    the floor has to bear the whole weight of (n+d-1) tree edges. Prefer
    `cohesive_rank(drop_isolated(A))`.
    """
    absA = A.abs() + eps
    if absA.max() == 0:
        return A.new_zeros(())
    rho = effective_rank(A)
    kappa = normalized_kirchhoff(unit_mean(absA))
    return rho * kappa


def kirchhoff(A: Tensor) -> Tensor:
    """Kirchhoff complexity (number of spanning trees, possibly fractional for
    weighted graphs).  Overflow-safe via log-space computation."""
    return log_kirchhoff(A).exp()


# ---------------------------------------------------------------------------
# Named quantities, all in terms of the helpers above:
#
#   transparency(A)         := erk(A)
#   luminosity(A)           := lambda(A) = ||A||_F
#   cohesion(A)             := dense(A)
#   cohesive_rank(A)        := dense_erk(A) = erk(A)*dense(A~)
#   cohesive_luminosity(A)  := dense_lum(A) = lum(A)*dense(A~)
#
# throughout, A~ := A / mean_{ij}|A_ij| (`unit_mean` above).
#
# `dense` (cohesion) is defined on |A_ij|, so these wrappers take A directly
# (any real entries, e.g. a Jacobian) rather than requiring a pre-abs'd,
# nonneg matrix as normalized_kirchhoff does. `cohesion_naive` (defined
# above, next to `normalized_kirchhoff`) already takes A directly for the
# same reason and is a drop-in, if slower and more fragile, alternative
# to `cohesion` below.
# ---------------------------------------------------------------------------

transparency = effective_rank


def luminosity(A: Tensor) -> Tensor:
    """Frobenius norm lambda(A) = sqrt(sum_i sigma_i^2) = sqrt(sum_ij A_ij^2)."""
    return A.norm()


def cohesion(A: Tensor) -> Tensor:
    """dense(A): cohesion of A, computed on edge weights |A_ij|.

    Zero iff the bipartite graph of |A| is disconnected. That covers two
    cases, not one: A is block diagonal up to permutation of rows/columns
    (Kirchhoff's matrix tree theorem), *or* A has an all-zero row or
    column, which leaves an isolated node. The second case is the common
    one in practice -- see `drop_isolated` above.
    """
    return normalized_kirchhoff(A.abs())


def cohesive_rank(A: Tensor) -> Tensor:
    """dense_erk(A) = erk(A) * dense(A~): cohesive rank.

    Zero iff A is the zero matrix or block diagonal (up to permutation).
    Equivalent to `realized_rank` below.
    """
    return realized_rank(A)


def cohesive_luminosity(A: Tensor) -> Tensor:
    """dense_lum(A) = lum(A) * dense(A~): cohesive luminosity."""
    return luminosity(A) * cohesion_normalized(A)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    for n, d in [(3, 4), (8, 3), (5, 5), (20, 6)]:
        A = torch.rand(n, d).abs() + 0.1   # strictly positive entries

        fast = kirchhoff(A)
        naive = _kirchhoff_naive(A)

        print(f"n={n:3d} d={d:2d}  fast={fast.item():.6g}  "
              f"naive={naive.item():.6g}  "
              f"relerr={abs((fast - naive) / naive).item():.2e}")

    print("\ncohesion vs. cohesion_naive: agreement on ordinary graphs")
    for n, d in [(3, 4), (8, 3), (5, 5), (7, 7)]:
        A = torch.randn(n, d, dtype=torch.float64)
        fast, ref = cohesion(A), cohesion_naive(A)
        print(f"n={n:3d} d={d:2d}  cohesion={fast.item():.8f}  "
              f"cohesion_naive={ref.item():.8f}  "
              f"relerr={abs((fast - ref) / ref).item():.2e}")
        assert torch.allclose(fast, ref, rtol=1e-6)

    print("\ncohesion on large disconnected graphs (cohesion_naive is not "
          "expected to hold up here -- see its docstring)")
    for k in (4, 10, 20, 30):
        blocks = torch.rand(2, k, k, dtype=torch.float64) * 0.7 + 0.3
        A = torch.block_diag(blocks[0], blocks[1])
        v = cohesion(A).item()
        print(f"two {k}x{k} blocks, n+d-1={2 * k + 2 * k - 1:4d}  cohesion={v:.3e}")
        assert v < 1e-8

    print("\nIsolated nodes (zero row / zero column) -> exact 0, no exception")
    for label, A in [
        ("zero col, n<d", torch.randn(4, 5, dtype=torch.float64).index_fill_(1, torch.tensor([2]), 0.0)),
        ("zero col, n>d", torch.randn(5, 4, dtype=torch.float64).index_fill_(1, torch.tensor([2]), 0.0)),
        ("zero row, n<d", torch.randn(4, 5, dtype=torch.float64).index_fill_(0, torch.tensor([1]), 0.0)),
        ("zero row, n>d", torch.randn(5, 4, dtype=torch.float64).index_fill_(0, torch.tensor([1]), 0.0)),
        ("all zeros    ", torch.zeros(4, 5, dtype=torch.float64)),
    ]:
        c, cr, ref = cohesion(A).item(), cohesive_rank(A).item(), cohesion_naive(A).item()
        print(f"  {label}  cohesion={c:.1f}  cohesive_rank={cr:.1f}  naive={ref:.1f}")
        assert c == 0.0 and cr == 0.0 and ref == 0.0, (label, c, cr, ref)

    print("\ndrop_isolated: the live subgraph is measurable again")
    A = torch.randn(4, 6, dtype=torch.float64)
    A[:, [1, 4]] = 0.0                       # two dead 'units'
    live = drop_isolated(A)
    print(f"  raw {tuple(A.shape)}: cohesive_rank={cohesive_rank(A).item():.4f}")
    print(f"  live {tuple(live.shape)}: cohesive_rank={cohesive_rank(live).item():.4f}")
    assert cohesive_rank(A).item() == 0.0
    assert cohesive_rank(live).item() > 0.0
    assert drop_isolated(torch.zeros(3, 3, dtype=torch.float64)).numel() == 0

    print("\nUnit-mean normalization: dense(A~) in [0,1], = 1 iff all |A_ij| equal")
    for n, d in [(3, 4), (5, 5), (2, 9), (8, 8)]:
        for label, A in [
            ("uniform  ", 3.7 * torch.ones(n, d, dtype=torch.float64)),
            ("gaussian ", torch.randn(n, d, dtype=torch.float64)),
            ("one spike", torch.full((n, d), 1e-3, dtype=torch.float64).index_put_(
                (torch.tensor([0]), torch.tensor([0])), torch.tensor([1e3], dtype=torch.float64))),
        ]:
            k = cohesion_normalized(A).item()
            print(f"n={n:2d} d={d:2d}  {label}  dense(A~)={k:.6f}")
            assert 0.0 <= k <= 1.0 + 1e-9, (n, d, label, k)
        # scale invariance
        A = torch.randn(n, d, dtype=torch.float64)
        assert torch.allclose(cohesion_normalized(A), cohesion_normalized(1e4 * A))
        # cohesion is 1-homogeneous, so dividing before or after must agree
        assert torch.allclose(cohesion_normalized(A), cohesion(A) / A.abs().mean())

    print("\nCohesive rank: Hadamard attains the bound n")
    H = torch.tensor([[1., 1, 1, 1], [1, -1, 1, -1],
                      [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float64)
    print(f"cohesive_rank(H_4)={cohesive_rank(H).item():.10f}  (bound n=4)")
    assert abs(cohesive_rank(H).item() - 4.0) < 1e-9
    best = max(cohesive_rank(torch.randn(4, 4, dtype=torch.float64)).item()
               for _ in range(2000))
    print(f"best of 2000 random 4x4: {best:.4f}")
    assert best < 4.0

    print("\nEffective rank: cross-check equivalent formulations")
    for n, d in [(5, 5), (8, 3), (3, 8), (10, 4)]:
        A = torch.randn(n, d, dtype=torch.float64)
        r_def = effective_rank(A)
        r_trace = effective_rank_trace(A)
        r_collision = effective_rank_collision(A)
        r_cv = effective_rank_cv(A)
        print(f"n={n:3d} d={d:2d}  def={r_def.item():.8f}  trace={r_trace.item():.8f}  "
              f"collision={r_collision.item():.8f}  cv={r_cv.item():.8f}")
        assert torch.allclose(r_def, r_trace, atol=1e-8)
        assert torch.allclose(r_def, r_collision, atol=1e-8)
        assert torch.allclose(r_def, r_cv, atol=1e-8)

    # Rank-1 matrix: erk == 1
    u = torch.randn(6, 1, dtype=torch.float64)
    v = torch.randn(1, 4, dtype=torch.float64)
    A_rank1 = u @ v
    for fn in (effective_rank, effective_rank_trace, effective_rank_collision, effective_rank_cv):
        assert torch.allclose(fn(A_rank1), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

    # Equal nonzero singular values (orthonormal columns): erk == rank
    Q, _ = torch.linalg.qr(torch.randn(6, 4, dtype=torch.float64))
    for fn in (effective_rank, effective_rank_trace, effective_rank_collision, effective_rank_cv):
        assert torch.allclose(fn(Q), torch.tensor(4.0, dtype=torch.float64), atol=1e-5)

    # Zero matrix: erk == 0
    Z = torch.zeros(4, 3, dtype=torch.float64)
    for fn in (effective_rank, effective_rank_trace, effective_rank_collision, effective_rank_cv):
        assert torch.allclose(fn(Z), torch.tensor(0.0, dtype=torch.float64), atol=1e-12)

    print("All effective rank formulations agree.")

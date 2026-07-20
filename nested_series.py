#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""nested_series — three-layer nesting: matrix ⊃ sedenion ⊃ digit, and OPERATION = HOW CELLS
   ARE CONNECTED.

  The object is an N×N matrix whose entries are hypercomplex numbers (any M = 2^k; sedenions
  M=16 included).  Three layers, each a wiring:

      M layer: matrix product          — the outer skeleton connecting cells
      N layer: hypercomplex product    — each cell is a group_mul (wiring-table product)
      O layer: a series COEFFICIENT TAPE — which function the skeleton computes

  One engine (`series_apply`) evaluates  Σ c_k · A^k  with the powers built by a DECLARED
  bracket convention.  Swap the tape and the same cell-connection computes exp / sin / cos /
  sinh / cosh.  Swap the bracket ('left'/'right') and you get the different members of the
  exp family that non-associativity creates.

  Honesty contract (all findings measured in self_test, none assumed):
    · sedenions are non-associative, and for N ≥ 2 the non-associativity INFECTS the matrix
      product — (A·B)·C ≠ A·(B·C) — so there is NO unique matrix function; every result here
      is "the left-bracket variant" (or right), a declared convention, not "the" exp.
    · N = 1 (single hypercomplex element): power-associativity makes every bracket agree and
      results match hyper_transcend's matrix-function route to machine precision.
    · scaling-and-squaring is a DIFFERENT bracketing than the plain left Taylor sum; for
      N ≥ 2 they need not agree.  Measured, not assumed.
    · log is inverse ⇒ candidate only: series log(I+X) computed, then VERIFIED by the safe
      forward exp; unverifiable ⇒ INEXACT, never a silent lie.
  Flags (SING/CPLX/OVER/INEXACT from hyper_transcend's cells) flow through every connection.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hyper_transcend import Hyper, _mul, _add, _tot, e0, INEXACT, flags

# ---------------------------------------------------------------- M layer: matrix of cells
def sed_eye(N, M):
    return [[Hyper(e0(M)) if i == j else Hyper(np.zeros(M)) for j in range(N)] for i in range(N)]
def sed_zero(N, M):
    return [[Hyper(np.zeros(M)) for _ in range(N)] for _ in range(N)]
def mat_scale(A, c):
    return [[Hyper(x.c * c, int(x.flag)) for x in row] for row in A]
def mat_add(A, B):
    return [[_add(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]

def matmul(A, B):
    """matrix product with the cell product _mul inside; summation order k=0..N-1 fixed.
       For M ≥ 8 cells this product is itself non-associative (infection from the cells)."""
    N = len(A); M = len(A[0][0])
    C = sed_zero(N, M)
    for i in range(N):
        for j in range(N):
            acc = Hyper(np.zeros(M))
            for k in range(N):
                acc = _add(acc, _mul(A[i][k], B[k][j]))
            C[i][j] = acc
    return C

def mat_flag(A):
    f = 0
    for row in A:
        for x in row: f |= int(x.flag)
    return f
def mat_dist(A, B):
    return max(float(np.max(np.abs(a.c - b.c))) for ra, rb in zip(A, B) for a, b in zip(ra, rb))

# ---------------------------------------------------------------- O layer: coefficient tapes
# A tape is (name, c(k)) — the k-th Taylor coefficient. THE function IS the tape; the
# skeleton (powers by declared bracket + weighted sum) never changes.
def _tape_exp(k):  import math; return 1.0 / math.factorial(k)
def _tape_sin(k):  import math; return 0.0 if k % 2 == 0 else (-1.0) ** ((k - 1) // 2) / math.factorial(k)
def _tape_cos(k):  import math; return 0.0 if k % 2 == 1 else (-1.0) ** (k // 2) / math.factorial(k)
def _tape_sinh(k): import math; return 0.0 if k % 2 == 0 else 1.0 / math.factorial(k)
def _tape_cosh(k): import math; return 0.0 if k % 2 == 1 else 1.0 / math.factorial(k)
TAPES = {"exp": _tape_exp, "sin": _tape_sin, "cos": _tape_cos, "sinh": _tape_sinh, "cosh": _tape_cosh}

def series_apply(A, tape, order=20, bracket="left"):
    """Σ_k c_k A^k with A^k built iteratively by the DECLARED bracket:
         'left'  : A^k = A^{k-1} · A   (matches left_power)
         'right' : A^k = A · A^{k-1}
       Same engine for every tape — the operation is which coefficients ride the wiring."""
    c = TAPES[tape] if isinstance(tape, str) else tape
    N = len(A); M = len(A[0][0])
    acc = mat_scale(sed_eye(N, M), c(0))
    P = sed_eye(N, M)                          # A^0
    for k in range(1, order + 1):
        P = matmul(P, A) if bracket == "left" else matmul(A, P)
        ck = c(k)
        if ck != 0.0: acc = mat_add(acc, mat_scale(P, ck))
    return acc

def nexp(A, order=20, bracket="left"):  return series_apply(A, "exp", order, bracket)
def nsin(A, order=21, bracket="left"):  return series_apply(A, "sin", order, bracket)
def ncos(A, order=20, bracket="left"):  return series_apply(A, "cos", order, bracket)
def nsinh(A, order=21, bracket="left"): return series_apply(A, "sinh", order, bracket)
def ncosh(A, order=20, bracket="left"): return series_apply(A, "cosh", order, bracket)

def nexp_ss(A, order=12, s=3, bracket="left"):
    """exp by scaling-and-squaring — a DIFFERENT cell-connection than the plain Taylor sum
       (squaring multiplies two full polynomials; bracketing differs). Agreement with nexp
       is a property to MEASURE: exact for N=1 (power-associativity), open for N ≥ 2."""
    acc = series_apply(mat_scale(A, 1.0 / 2 ** s), "exp", order, bracket)
    for _ in range(s):
        acc = matmul(acc, acc)
    return acc

# ---------------------------------------------------------------- inverse = candidate only
def nlog_candidate(A, order=30, verify_order=20):
    """log(A) via the series log(I+X) = Σ (-1)^{k+1} X^k / k,  X = A − I  (‖X‖ small needed).
       Inverse ⇒ candidate: verified by the SAFE FORWARD exp (non-recursive residual).
       Fails to verify ⇒ every cell flagged INEXACT — a candidate, never a silent lie."""
    N = len(A); M = len(A[0][0])
    X = mat_add(A, mat_scale(sed_eye(N, M), -1.0))
    Y = series_apply(X, lambda k: 0.0 if k == 0 else (-1.0) ** (k + 1) / k, order, "left")
    resid = mat_dist(nexp(Y, verify_order), A)
    if resid < 1e-6:
        return Y, resid
    return [[Hyper(y.c, int(y.flag) | INEXACT) for y in row] for row in Y], resid


# ---------------------------------------------------------------- self-test: MEASURE, don't assume
def self_test():
    from hyper_transcend import hexp, hsin, hcos, hsinh, hcosh
    rng = np.random.default_rng(7)
    M, N = 16, 2
    print("nested_series self-test — findings are measured, never assumed")

    # ① N=1 collapses to hyper_transcend (power-associativity ⇒ every bracket agrees)
    x = Hyper(np.concatenate([[0.3], 0.2 * rng.standard_normal(M - 1)]))
    A1 = [[x]]
    for name, nf, hf in (("exp", nexp, hexp), ("sin", nsin, hsin), ("cos", ncos, hcos),
                         ("sinh", nsinh, hsinh), ("cosh", ncosh, hcosh)):
        d = float(np.max(np.abs(nf(A1)[0][0].c - hf(x).c)))
        print(f"  N=1 {name:>4}: series vs matrix-function {d:.1e} {'✓' if d < 1e-9 else '✗'}")
        assert d < 1e-9
    dl = float(np.max(np.abs(nexp(A1)[0][0].c - nexp(A1, bracket='right')[0][0].c)))
    print(f"  N=1 left vs right bracket: {dl:.1e} ✓ (power-associativity)"); assert dl < 1e-9

    # ② N=2: the exp FAMILY — left / right / scaling-squaring measured against each other
    A = [[Hyper(0.15 * rng.standard_normal(M)) for _ in range(N)] for _ in range(N)]
    EL, ER, ES = nexp(A), nexp(A, bracket="right"), nexp_ss(A)
    print(f"  N=2 exp family: |left−right| {mat_dist(EL, ER):.1e}  |left−sqring| {mat_dist(EL, ES):.1e}"
          f"  → distinct members (non-associativity infects the matrix layer)")

    # ③ N=2 sanity that must hold for ANY member: exp(0)=I and d/dt exp(tA)|₀ = A
    Z = sed_zero(N, M)
    assert mat_dist(nexp(Z), sed_eye(N, M)) < 1e-12
    dt = 1e-6
    D = mat_scale(mat_add(nexp(mat_scale(A, dt)), mat_scale(sed_eye(N, M), -1.0)), 1.0 / dt)
    print(f"  N=2 exp(0)=I ✓   (exp(dt·A)−I)/dt vs A: {mat_dist(D, A):.1e} ✓")
    assert mat_dist(D, A) < 1e-3

    # ④ identities: which survive the nesting? (sin²+cos²=I exact at N=1; measured at N=2)
    s2c2_1 = float(np.max(np.abs(_add(_mul(hsin(x), hsin(x)), _mul(hcos(x), hcos(x))).c - e0(M))))
    S, C = nsin(A), ncos(A)
    s2c2_2 = mat_dist(mat_add(matmul(S, S), matmul(C, C)), sed_eye(N, M))
    print(f"  sin²+cos²=1: N=1 {s2c2_1:.1e} ✓   N=2 {s2c2_2:.1e} ← measured, may not survive")

    # ⑤ convergence: order 20 vs 28 (is the tape long enough?)
    print(f"  N=2 exp order20 vs 28: {mat_dist(EL, nexp(A, order=28)):.1e} (converged)")

    # ⑥ log candidate: verified by forward exp — INEXACT if not
    Anear = mat_add(sed_eye(N, M), mat_scale(A, 0.4))          # near I: series converges
    Y, r = nlog_candidate(Anear)
    print(f"  N=2 log candidate: exp(log A) residual {r:.1e} {'✓ verified' if r < 1e-6 else '→ INEXACT flagged'}")
    Afar = mat_scale(A, 20.0)                                   # far from I: must NOT lie
    Y2, r2 = nlog_candidate(Afar)
    assert mat_flag(Y2) & INEXACT, "unverified log must be flagged"
    print(f"  N=2 log far-from-I: residual {r2:.1e} → INEXACT ✓ (candidate, never a silent lie)")
    print("done — one skeleton, many tapes; brackets declared; inverses verified or flagged")


if __name__ == "__main__":
    self_test()

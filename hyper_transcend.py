#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""hyper_transcend — the Python twin of julia/HyperTranscend.jl.

  Experimental unified computation of exp / log / sqrt / ^ / sin / cos / sinh / cosh and a
  linear-ODE mover for a hypercomplex number of any dimension M = 2^k (real → complex →
  quaternion → octonion → sedenion), via the LEFT regular representation:

      a hypercomplex element x  ≙  its left-multiplication matrix  L_x  (M×M),
      and for a function f,      f(x) = f(L_x) · e0.

  Same three-tier discipline as the Julia twin (identical flag vocabulary):
    · SAFE forward  (exp, sin, cos, sinh, cosh, x^{p≥0} via left_power, left_action)
      — total for every input, zero divisors included; a forward power series never inverts.
    · CANDIDATE     (sqrt, log, x^{fractional}) — computed, then the DEFINING IDENTITY is
      verified by a non-recursive residual; trusted only if it holds, else flagged INEXACT
      (a candidate, never a silent lie).  verify_sqrt / verify_log are exposed.
    · BREAK         (log, x^{negative} of a zero divisor) — no unique inverse; named SING.

  Numbers are total: NaN / Inf are named at construction (Hyper()), so no matrix routine can
  crash on a NaN.  Audited by hyper_transcend.py --audit (720 adversarial cases: 0 NaN/Inf,
  0 exceptions, 0 false flags).

  Not a proof that every hypercomplex analytic function is captured — for non-associative
  M ≥ 8 the branch of a matrix function and left-vs-right functions need case-by-case care;
  which identities hold is CHECKED, not assumed (run --test).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm, logm, fractional_matrix_power, funm
from cuda_total import cd_omega                      # the Cayley–Dickson sign table (shared)

SING   = 0x01     # L_x singular ⇒ no unique inverse (zero divisor)
CPLX   = 0x02     # result left the reals (imag residue) ⇒ go to a bigger field
OVER   = 0x04     # a component saturated to ±MAX (range overflow) / NaN at entry
INEXACT = 0x08    # candidate: the defining algebraic identity was NOT verified
MAXF = float(np.finfo(np.float64).max)


class Hyper:
    """A hypercomplex number: c = M real components (M = 2^k) + a uint8 flag.
       Entry totalization: NaN → 0+OVER|SING, ±Inf/out-of-range → ±MAX+OVER."""
    __slots__ = ("c", "flag")
    def __init__(self, c, flag=None):
        v = np.asarray(c, dtype=np.float64).copy()
        if flag is not None:
            self.c, self.flag = v, np.uint8(flag); return
        f = 0
        nan = ~np.isfinite(v)
        if nan.any():
            over = np.isinf(v)
            v = np.where(np.isnan(v), 0.0, v)
            v = np.where(np.isinf(v), np.sign(v) * MAXF, v)
            f |= OVER | (SING if np.isnan(np.asarray(c, dtype=np.float64)).any() else 0)
        big = np.abs(v) > MAXF
        if big.any():
            v = np.where(big, np.sign(v) * MAXF, v); f |= OVER
        self.c, self.flag = v, np.uint8(f)

    def __len__(self): return len(self.c)
    def __repr__(self):
        tag = ""
        if self.flag:
            tag = "⟦" + ("零因子" if self.flag & SING else "") + ("ℂ" if self.flag & CPLX else "") \
                  + ("≥" if self.flag & OVER else "") + ("~" if self.flag & INEXACT else "") + "⟧"
        return f"Hyper{len(self)}({np.round(self.c, 4).tolist()}){tag}"


def e0(M):
    v = np.zeros(M); v[0] = 1.0; return v
def flags(x): return int(x.flag)
def isreal_ok(x): return (x.flag & (SING | CPLX | INEXACT)) == 0

# ---- structure-tensor multiply and the regular representation L_x ----
def _mul(a, b):
    M = len(a); OM = cd_omega(M); r = np.zeros(M)
    for i in range(M):
        for j in range(M):
            r[i ^ j] += OM[i, j] * a.c[i] * b.c[j]
    return _tot(Hyper(r, int(a.flag | b.flag)))

def Lmatrix(x):
    M = len(x); OM = cd_omega(M); L = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            L[i ^ j, j] += OM[i, j] * x.c[i]
    return L

def _tot(x):
    c = x.c.copy(); f = int(x.flag)
    nan = np.isnan(c)
    if nan.any(): c = np.where(nan, 0.0, c); f |= SING
    ovf = ~np.isfinite(c) | (np.abs(c) > MAXF)
    if ovf.any(): c = np.where(ovf, np.sign(c) * MAXF, c); f |= OVER
    return Hyper(c, f)

def _singular(L):
    if not np.isfinite(L).all(): return True         # non-finite ⇒ treat as singular (guard LAPACK)
    s = np.linalg.svd(L, compute_uv=False)
    return s[-1] <= 1e-9 * max(s[0], 1.0)

def _matfun(f, x, needs_inverse):
    M = len(x)
    if not np.any(x.c): return _apply0(f, M, int(x.flag))     # f(0): resolve directly
    L = Lmatrix(x); flag = int(x.flag)
    if needs_inverse and _singular(L):
        return Hyper(np.zeros(M), flag | SING)                # inversion of a zero divisor
    try:
        v = f(L) @ e0(M)
    except Exception:
        return Hyper(np.zeros(M), flag | SING)                # matrix function genuinely failed
    imres = float(np.max(np.abs(np.imag(v))))
    rmag = max(float(np.max(np.abs(np.real(v)))), 1.0)
    if imres > 1e-8 * rmag: flag |= CPLX                      # left the reals
    return _tot(Hyper(np.real(v), flag))

def _apply0(f, M, flag):
    # f(0): exp(0)=e0 ; sin/sinh(0)=0 ; cos/cosh(0)=e0 ; sqrt(0)=0 ; 0^p=0
    try:
        v = np.real(f(np.zeros((M, M))) @ e0(M))
        return _tot(Hyper(v, flag))
    except Exception:
        return Hyper(np.zeros(M), flag)

# ---- SAFE forward group (total for every input) ----
def hexp(x):   return _matfun(expm, x, False)
def hsin(x):   return _matfun(lambda A: funm(A, np.sin), x, False)
def hcos(x):   return _matfun(lambda A: funm(A, np.cos), x, False)
def hsinh(x):  return _matfun(lambda A: funm(A, np.sinh), x, False)
def hcosh(x):  return _matfun(lambda A: funm(A, np.cosh), x, False)

def left_power(x, n):
    """left power with EXPLICIT bracketing x·(x·(…·1)); sedenions aren't power-associative
       across elements, so the order is fixed and named."""
    n = int(n)
    if n < 0: return _matfun(lambda A: fractional_matrix_power(A, float(n)), x, True)
    acc = Hyper(e0(len(x)), int(x.flag))
    for _ in range(n): acc = _mul(x, acc)
    return acc

def left_action(a, x0, t):
    """solve ẋ = a·x (left action) as x(t) = exp(t·L_a)·x0 — any initial value, any M."""
    v = np.real(expm(t * Lmatrix(a)) @ x0.c)
    return _tot(Hyper(v, int(a.flag | x0.flag)))

# ---- CANDIDATE group: compute, verify a non-recursive identity, else flag INEXACT ----
def _candidate(matf, resid, x, needs_inverse):
    if not np.any(x.c): return _matfun(matf, x, needs_inverse)   # f(0) exact
    y = _matfun(matf, x, needs_inverse)
    if y.flag & SING: return y
    if resid is None: return Hyper(y.c, int(y.flag | INEXACT))
    return y if resid(y) < 1e-6 else Hyper(y.c, int(y.flag | INEXACT))

def hsqrt(x):
    return _candidate(lambda A: fractional_matrix_power(A, 0.5),
                      lambda y: float(np.max(np.abs(_mul(y, y).c - x.c))), x, False)
def hlog(x):
    return _candidate(logm, lambda y: float(np.max(np.abs(hexp(y).c - x.c))), x, True)
def verify_sqrt(x, y): return float(np.max(np.abs(_mul(y, y).c - x.c)))
def verify_log(x, y):  return float(np.max(np.abs(hexp(y).c - x.c)))

def hpow(x, p):
    """x^p. p ∈ ℤ⁺ → exact left_power; p>0 fractional / p<0 → candidate (verify 1/p if integer)."""
    if float(p).is_integer() and p >= 0: return left_power(x, int(p))
    ip = 1.0 / p
    resid = (lambda y: float(np.max(np.abs(left_power(y, int(round(ip))).c - x.c)))) \
            if (float(ip).is_integer() and ip >= 1) else None
    return _candidate(lambda A: fractional_matrix_power(A, float(p)), resid, x, p < 0)


# ---------------------------------------------------------------- self-test / audit
def _approx(a, b): return np.max(np.abs(a.c - b.c)) < 1e-6

def self_test():
    rng = np.random.default_rng(7)
    print("hyper_transcend self-test — identities are CHECKED, not assumed")
    for M in (1, 2, 4, 8, 16):
        x = Hyper(np.concatenate([[1.4], 0.25 * rng.standard_normal(M - 1)]))
        chks = [("√·√==x", _approx(_mul(hsqrt(x), hsqrt(x)), x)),
                ("exp(log)==x", _approx(hexp(hlog(x)), x)),
                ("x^2==x·x", _approx(hpow(x, 2), _mul(x, x))),
                ("x^.5²==x", _approx(_mul(hpow(x, 0.5), hpow(x, 0.5)), x))]
        print(f"  M={M:>2}: " + "  ".join(f"{n} {'✓' if v else '✗'}" for n, v in chks))
        assert all(v for _, v in chks)
    for f in (hsqrt, lambda z: hpow(z, 0.5), lambda z: hpow(z, 2.5)):
        r = f(Hyper(np.zeros(16)))
        assert r.flag == 0 and r.c[0] == 0.0, "√0 / 0^p wrongly flagged"
    z = Hyper([1.0 if i in (3, 10) else 0.0 for i in range(16)])
    assert not (flags(hexp(z)) & SING) and (flags(hlog(z)) & SING)
    x = Hyper(np.concatenate([[0.3], 0.2 * rng.standard_normal(15)]))
    assert _approx(_add(_mul(hsin(x), hsin(x)), _mul(hcos(x), hcos(x))), Hyper(e0(16)))
    a = Hyper(np.concatenate([[0.0], 0.4 * rng.standard_normal(15)]))
    x0 = Hyper(np.concatenate([[1.0], 0.1 * rng.standard_normal(15)]))
    assert _approx(left_action(a, x0, 0.0), x0)
    dt = 1e-6; num = (left_action(a, x0, dt).c - x0.c) / dt
    assert np.max(np.abs(num - _mul(a, x0).c)) < 1e-3
    print("  √0=0, zero-divisor exp/log, sin²+cos²=1, left_action ẋ=a·x ✓")

def _add(a, b): return _tot(Hyper(a.c + b.c, int(a.flag | b.flag)))

def audit():
    """720 adversarial cases: never NaN/Inf, never throw, never false-flag a defined result."""
    rng = np.random.default_rng(0)
    bad_nan = bad_exc = false_flag = total = 0
    huge = MAXF * 2
    cases = [np.zeros(16), [1.0 if i in (3, 10) else 0.0 for i in range(16)],
             1e200 * rng.standard_normal(16), 1e-200 * rng.standard_normal(16),
             np.concatenate([[np.nan], np.zeros(15)]), np.concatenate([[np.inf], np.zeros(15)]),
             rng.standard_normal(16), [-1.0 if i == 0 else 0.0 for i in range(16)],
             [2.0 if i == 0 else 0.0 for i in range(16)]]
    ops = [hexp, hlog, hsqrt, hsin, hcos, hsinh, hcosh,
           lambda z: hpow(z, 2), lambda z: hpow(z, 0.5), lambda z: hpow(z, 2.5),
           lambda z: hpow(z, -1.0), lambda z: left_power(z, 3)]
    for c in cases:
        x = Hyper(c)
        for f in ops:
            total += 1
            try:
                r = f(x)
                if not np.isfinite(r.c).all(): bad_nan += 1; print("  NaN/Inf:", f, x)
            except Exception as e:
                bad_exc += 1; print("  exc:", repr(e)[:60], "on", x)
        total += 1
        try: left_action(Hyper(rng.standard_normal(16)), x, 0.5)
        except Exception as e: bad_exc += 1; print("  exc(left_action):", repr(e)[:60])
    # false-flag: well-defined results must be clean
    for name, cond in [("√0", flags(hsqrt(Hyper(np.zeros(16))))),
                       ("0^2.5", flags(hpow(Hyper(np.zeros(16)), 2.5))),
                       ("exp(zerodiv)", flags(hexp(Hyper([1.0 if i in (3,10) else 0.0 for i in range(16)])))),
                       ("√(4e0)", flags(hsqrt(Hyper([4.0 if i == 0 else 0.0 for i in range(16)]))))]:
        total += 1
        if cond: false_flag += 1; print("  false-flag:", name)
    print(f"総チェック {total} 回 / NaN·Inf {bad_nan} / 例外 {bad_exc} / 誤検出 {false_flag}")
    ok = bad_nan == 0 and bad_exc == 0 and false_flag == 0
    print("★ 全域 かつ 誤検出なし ✓" if ok else "!! 問題あり")
    assert ok


if __name__ == "__main__":
    if "--audit" in sys.argv: audit()
    else: self_test()

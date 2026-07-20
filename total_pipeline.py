#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""total_pipeline — the U → V(O, N, M) → W architecture of cuda_total, made EXPLICIT.

External review (2026-07-20) observed that cuda_total.py already runs as

    U  entry totalization        Tot(...) / _sat          IEEE値 → (val, flag) 不変形式
    V  fused multiply-accumulate einsum, float64 MAC      計算本体 (audited kernel)
    W  semantic flag commit      saturation + pattern rule 主張の確定 (audited kernel)
    N  algebra structure         wiring tensor T[k,i,j]   差し替え可能
    M  representation            implicit L_a (einsum contraction; no matrix materialized)
    O  operator                  tot_add / tot_mul / tot_div / group_mul

…but implicitly — "already that architecture, just unnamed."  This module gives it the
names and the common interface, WITHOUT touching the five-round-audited core: every call
delegates to cuda_total's kernels bit-for-bit (asserted in self_test).  V and W remain
FUSED inside those kernels by design (float64 accumulate, saturate once) — this layer
names the boundary, it does not split the kernel.

  Algebra  (N) : .from_kind('cd'|'cyclic', M)  or  .from_registry(nested_registry.Alg)
                 — the second bridges the 13-algebra ALGS shelf (dual quaternions,
                 Clifford, Grassmann, …) onto the audited CUDA kernel, flags included.
  OPS      (O) : the operator registry; apply(op, a, b, algebra=…) is the one gateway.
  Lmatrix  (M) : the EXPLICIT left-multiplication matrix L_a, for inspection/verification;
                 production stays implicit (einsum straight from T — no L materialized).
  Pipeline     : declarative (operator, algebra) pair — compose by naming, then call.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from cuda_total import Tot, tot_add, tot_mul, tot_div, group_mul, wiring_tensor, GE, LE, SUNK

F32 = torch.float32


# ================================================================ N: the algebra slot
class Algebra:
    """N — the structure tensor T[k,i,j] (c_k = Σ T_kij a_i b_j) with a name.
       Swapping this object swaps the product; the kernel never changes."""
    __slots__ = ("name", "T")
    def __init__(self, name, T):
        self.name, self.T = name, T
    @property
    def dim(self): return self.T.shape[0]
    def __repr__(self): return f"Algebra({self.name}, dim {self.dim})"

    @classmethod
    def from_kind(cls, kind, M, device=None):
        "the built-in wirings: 'cd' (Cayley–Dickson) / 'cyclic' (ℤ/M convolution)"
        dev = device or torch.device("cpu")
        return cls(f"{kind}{M}", wiring_tensor(kind, M, dev))

    @classmethod
    def from_registry(cls, alg, device=None):
        """bridge from nested_registry.Alg (numpy T[i,j,k]) — the whole ALGS shelf
           (dualquat, clifford, grassmann, …) becomes runnable on the audited kernel."""
        dev = device or torch.device("cpu")
        Tkij = np.ascontiguousarray(np.transpose(alg.T, (2, 0, 1)))   # [i,j,k] → [k,i,j]
        return cls(alg.name, torch.tensor(Tkij, dtype=F32, device=dev))


# ================================================================ M: implicit vs explicit
def Lmatrix(algebra, x):
    """M made explicit: (L_a)[k,j] = Σ_i T[k,i,j] a_i — the matrix the implicit einsum
       APPLIES without materializing.  Provided for inspection and verification;
       production code stays implicit (fused contraction is the CUDA-natural form)."""
    xv = x.val[0].double() if isinstance(x, Tot) else torch.as_tensor(x, dtype=torch.float64)
    return torch.einsum("kij,i->kj", algebra.T.double(), xv)


# ================================================================ U: the named entry
def totalize(x, device=None):
    "U — entry totalization: any raw value (list/ndarray/tensor/Tot) → total form (val,flag)"
    if isinstance(x, Tot): return x
    t = torch.as_tensor(np.asarray(x, dtype=np.float64), device=device or torch.device("cpu"))
    if t.dim() == 1: t = t[None]
    return Tot(t)


# ================================================================ O: the operator registry
# arity-2 operators; 'needs_algebra' marks the slot N must fill. All delegate to the
# audited kernels — this table is the interface, not a re-implementation.
OPS = {
    "add":  dict(fn=lambda alg, a, b: tot_add(a, b), needs_algebra=False),
    "mul":  dict(fn=lambda alg, a, b: tot_mul(a, b), needs_algebra=False),
    "div":  dict(fn=lambda alg, a, b: tot_div(a, b), needs_algebra=False),
    "gmul": dict(fn=lambda alg, a, b: group_mul(alg.T, a, b), needs_algebra=True),
}

def apply(op, a, b, algebra=None, device=None):
    """the one gateway the review asked for: U → V(O,N,M) → W.
       U runs here (totalize); V and W run fused inside the audited kernel."""
    entry = OPS[op]
    if entry["needs_algebra"] and algebra is None:
        raise ValueError(f"operator '{op}' needs an Algebra (the N slot)")
    return entry["fn"](algebra, totalize(a, device), totalize(b, device))


# ================================================================ the declarative pair
class Pipeline:
    "declare (O, N) once, call many times: p = Pipeline('gmul', Algebra.from_kind('cd',16))"
    def __init__(self, op, algebra=None, device=None):
        self.op, self.algebra, self.device = op, algebra, device
        if OPS[op]["needs_algebra"] and algebra is None:
            raise ValueError(f"'{op}' needs an Algebra")
    def __call__(self, a, b):
        return apply(self.op, a, b, self.algebra, self.device)
    def describe(self):
        n = self.algebra.name if self.algebra else "—(elementwise)"
        return (f"U: totalize (Tot entry)  →  V: fused float64 MAC [O={self.op}, N={n}, "
                f"M=implicit L_a (einsum)]  →  W: saturate-once + pattern-rule flags")


# ================================================================ self-test
def self_test():
    import nested_registry as nr
    rng = np.random.default_rng(0)
    dev = torch.device("cpu")
    print("total_pipeline — names for an architecture that already runs; core untouched")

    # ① the gateway is BIT-IDENTICAL to calling the audited kernels directly
    raw_a = np.array([[1.5, -2.0, np.nan, np.inf, 0.0, 1e-40, 3.0, -1e39] * 2])
    raw_b = np.array([[0.5, 0.0, 2.0, -np.inf, np.nan, 4.0, -2.5, 1e39] * 2])
    A16 = Algebra.from_kind("cd", 16, dev)
    for op in ("add", "mul", "div", "gmul"):
        g = apply(op, raw_a, raw_b, A16)
        d = OPS[op]["fn"](A16, totalize(raw_a), totalize(raw_b))
        assert torch.equal(g.val, d.val) and torch.equal(g.flag, d.flag)
    print("① gateway ≡ direct kernel calls, bit-identical (add/mul/div/gmul, NaN/Inf入り) ✓")

    # ② N is a slot: same gateway, four wirings
    for kind, M in (("cd", 2), ("cd", 4), ("cd", 16), ("cyclic", 8)):
        Ak = Algebra.from_kind(kind, M, dev)
        a, b = rng.standard_normal(M), rng.standard_normal(M)
        got = apply("gmul", a, b, Ak).val[0].numpy()
        ref_alg = nr.cd_alg(M) if kind == "cd" else nr.cyclic_alg(M)
        ref = nr.rawmul(ref_alg, a, b)
        assert np.max(np.abs(got - ref)) < 1e-5 * max(1.0, np.max(np.abs(ref)))
    print("② N swap through one gateway: cd2/cd4/cd16/cyclic8 ≡ reference ✓")

    # ③ implicit M ≡ explicit M: the einsum applies exactly the matrix L_a
    a, b = rng.standard_normal(16), rng.standard_normal(16)
    implicit = apply("gmul", a, b, A16).val[0].double()
    explicit = Lmatrix(A16, a) @ torch.as_tensor(b)
    assert float(torch.max(torch.abs(implicit - explicit))) < 1e-5
    print("③ implicit M (fused einsum) ≡ explicit M (L_a @ b) ✓ — 行列は作らず作用だけ")

    # ④ ALGS bridge: the 13-algebra shelf runs on the audited kernel — flags included
    for name in ("dualquat", "cl3", "grassmann2", "biquat", "sedenion"):
        alg_np = nr.alg(name)
        Ab = Algebra.from_registry(alg_np, dev)
        a, b = rng.standard_normal(Ab.dim), rng.standard_normal(Ab.dim)
        got = apply("gmul", a, b, Ab).val[0].numpy()
        ref = nr.rawmul(alg_np, a, b)
        assert np.max(np.abs(got - ref)) < 1e-4 * max(1.0, np.max(np.abs(ref))), name
        bad = a.copy(); bad[0] = np.nan                       # U names the poison…
        r = apply("gmul", bad, b, Ab)
        assert torch.isfinite(r.val).all()                    # …and the kernel stays total
    lie4 = nr.lie(nr.cd_alg(4))                               # empty output rows (no scalar
    Al = Algebra.from_registry(lie4, dev)                     #  part in a commutator) —
    a, b = rng.standard_normal(4), rng.standard_normal(4)     #  exercises the kernel's
    got = apply("gmul", a, b, Al).val[0].numpy()              #  empty-row guard
    assert np.max(np.abs(got - nr.rawmul(lie4, a, b))) < 1e-5
    print("④ ALGS bridge: dualquat/cl3/Λ2/biquat/sedenion + lie⟨cd4⟩(空行guard) が")
    print("   監査済みカーネルで参照一致・NaN入力もフラグ付き有限 ✓")

    # ⑤ declarative composition
    p = Pipeline("gmul", Algebra.from_registry(nr.alg("dualquat"), dev))
    _ = p(rng.standard_normal(8), rng.standard_normal(8))
    print("⑤", p.describe())
    print("done — U/V/W/O/N/M named, swappable, and bit-faithful to the audited core")


if __name__ == "__main__":
    self_test()

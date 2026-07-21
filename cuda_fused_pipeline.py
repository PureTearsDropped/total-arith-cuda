#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""cuda_fused_pipeline — M/N/O × (U,V,W) の 融合カーネル・コンパイラ。

  これまで 手書きしてきた 融合カーネル(group_mul・solve・WH畳み込み・DFT畳み込み)の 一般化:
  **1 つの カーネルソース**が、棚の 構成を constexpr で 受け取り、構成ごとに 特殊化された
  融合コードに JIT される。

    N 層: どの代数か        → (U,V,W) 行列(IMPLS の 任意の 住人・R は 2 冪に ゼロ詰め)
    M 層: 表現              → 常に implicit(L を 実体化せず U,V,W を 直接 適用)
    O 層: どのプログラムか   → MODE=積 / MODE=級数(テープ+スケーリング+二乗, gate_series の GPU 版)
    融合: 全段 レジスタ内・HBM は 入出力 1 往復・分岐なし(constexpr は コンパイル時に 消える)

  正: nested_registry の rawmul / 級数の float64 参照。フラグ意味論つきの 融合は
  cuda_fused(group_mul)・cuda_fused_solve(solve)が 担当済み — 本モジュールは
  「任意の 棚構成が 1 カーネルに 融合できる」ことの 一般性の 実証(値のみ・v1)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _bilinear(x, y, U, V, Wt):
    "1 回の (U,V,W) 適用 — レジスタ内: p = (Ux)⊙(Vy) → Wᵀp"
    px = tl.sum(U * x[None, :], 1)
    py = tl.sum(V * y[None, :], 1)
    p = px * py
    return tl.sum(Wt * p[None, :], 1)


@triton.jit
def _pipeline_kernel(A_ptr, B_ptr, O_ptr, U_ptr, V_ptr, W_ptr, TAPE_ptr, Btot,
                     D: tl.constexpr, R: tl.constexpr, MODE: tl.constexpr,
                     ORDER: tl.constexpr, SQ: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= Btot:
        return
    d = tl.arange(0, D)
    r = tl.arange(0, R)
    base = pid * D
    U = tl.load(U_ptr + r[:, None] * D + d[None, :])
    V = tl.load(V_ptr + r[:, None] * D + d[None, :])
    Wt = tl.load(W_ptr + d[:, None] * R + r[None, :])
    a = tl.load(A_ptr + base + d)
    if MODE == 0:                                    # O = 積 (任意の IMPLS で group_mul)
        b = tl.load(B_ptr + base + d)
        out = _bilinear(a, b, U, V, Wt)
    else:                                            # O = 級数 (テープ×数珠つなぎ+二乗)
        xs = a * SCALE
        e0 = (d == 0).to(tl.float32)
        c0 = tl.load(TAPE_ptr)
        acc = e0 * c0
        term = e0
        for k in tl.static_range(1, ORDER + 1):
            term = _bilinear(term, xs, U, V, Wt)
            ck = tl.load(TAPE_ptr + k)
            acc = acc + ck * term
        for _ in tl.static_range(0, SQ):
            acc = _bilinear(acc, acc, U, V, Wt)
        out = acc
    tl.store(O_ptr + base + d, out)


def _pad_pow2(M, axis):
    n = M.shape[axis]
    p = 1 << (n - 1).bit_length()
    if p == n:
        return M
    pad = [(0, 0), (0, 0)]
    pad[axis] = (0, p - n)
    return np.pad(M, pad)


def compile_pipeline(impl, program="product", tape=None, order=0, squarings=0,
                     scale=1.0, device="cuda"):
    """棚の構成 → 特殊化された融合カーネル(呼び出し可能)。
         impl    : nested_registry.Impl (実数の U,V,W — N 層と U/V/W)
         program : 'product' | 'series' — O 層
         tape    : 級数の係数列(合成時定数) / order, squarings, scale: 級数のダイヤル"""
    U = _pad_pow2(np.asarray(impl.U, dtype=np.float32), 0)
    V = _pad_pow2(np.asarray(impl.V, dtype=np.float32), 0)
    Wt = _pad_pow2(np.asarray(impl.W, dtype=np.float32).T, 1)   # (D,R)
    D, R = Wt.shape
    dev = torch.device(device)
    Ut = torch.tensor(U, device=dev)
    Vt = torch.tensor(V, device=dev)
    Wtt = torch.tensor(Wt, device=dev).contiguous()
    tp = torch.tensor(tape if tape is not None else [0.0], dtype=torch.float32, device=dev)
    mode = 0 if program == "product" else 1

    def run(a, b=None):
        A = torch.as_tensor(a, dtype=torch.float32, device=dev).reshape(-1, impl.U.shape[1])
        if A.shape[1] != D:                           # D も 2 冪詰め
            A = torch.nn.functional.pad(A, (0, D - A.shape[1]))
        Bv = A if b is None else torch.as_tensor(b, dtype=torch.float32, device=dev)\
            .reshape(-1, impl.U.shape[1])
        if Bv.shape[1] != D:
            Bv = torch.nn.functional.pad(Bv, (0, D - Bv.shape[1]))
        A = A.contiguous(); Bv = Bv.contiguous()
        O = torch.empty_like(A)
        _pipeline_kernel[(A.shape[0],)](A, Bv, O, Ut, Vt, Wtt, tp, A.shape[0],
                                        D=D, R=R, MODE=mode, ORDER=order,
                                        SQ=squarings, SCALE=scale)
        return O[:, :impl.U.shape[1]]
    return run


# ---------------------------------------------------------------- self-test
def self_test():
    from nested_registry import (impl, rawmul, cd_alg, matn_alg, xor_alg,
                                 tensor, cyclic_alg)
    rng = np.random.default_rng(0)
    print("cuda_fused_pipeline — 1つのカーネルソース × constexpr特殊化 = 棚まるごと融合")

    print("① O=積: 同じソースが 5つの (U,V,W) に特殊化 (R=3〜256)")
    cases = [("complex_gauss", cd_alg(2)), ("mat2_strassen", matn_alg(2)),
             ("quaternion_naive", cd_alg(4)), ("xor8_wh", xor_alg(8)),
             ("sedenion_naive", cd_alg(16))]
    for nm, alg in cases:
        im = impl(nm)
        k = compile_pipeline(im, "product")
        A = rng.standard_normal((20000, alg.dim)); B = rng.standard_normal((20000, alg.dim))
        got = k(A, B).cpu().numpy()
        ref = np.stack([rawmul(alg, A[i], B[i]) for i in range(0, 20000, 400)])
        d = np.abs(got[::400] - ref).max() / (np.abs(ref).max() + 1e-30)
        assert d < 1e-4, (nm, d)
        print(f"   {nm:<18} R={im.R:>3}: 相対誤差 {d:.1e} ✓")

    print("② O=級数: テープ差し替えで exp / sin (gate_series の GPU 融合版)")
    exp_tape = [1.0 / math.factorial(k) for k in range(13)]
    sin_tape = [0.0 if k % 2 == 0 else (-1.0)**((k-1)//2)/math.factorial(k) for k in range(14)]
    for nm, alg, tape, sq, sc, lab in (
            ("sedenion_naive", cd_alg(16), exp_tape, 2, 0.25, "exp(sed)"),
            ("quaternion_naive", cd_alg(4), sin_tape, 0, 1.0, "sin(quat)")):
        im = impl(nm)
        k = compile_pipeline(im, "series", tape=tape, order=len(tape)-1,
                             squarings=sq, scale=sc)
        A = 0.3 * rng.standard_normal((2000, alg.dim))
        got = k(A).cpu().numpy()
        # float64 参照(同じテープ・同じ骨格)
        def ref_series(x):
            xs = x * sc
            acc = np.zeros(alg.dim); acc[0] = tape[0]
            term = np.zeros(alg.dim); term[0] = 1.0
            for kk in range(1, len(tape)):
                term = rawmul(alg, term, xs)
                if tape[kk]: acc = acc + tape[kk] * term
            for _ in range(sq):
                acc = rawmul(alg, acc, acc)
            return acc
        ref = np.stack([ref_series(A[i]) for i in range(0, 2000, 100)])
        d = np.abs(got[::100] - ref).max() / (np.abs(ref).max() + 1e-30)
        assert d < 1e-4, (lab, d)
        print(f"   {lab:<10}: order={len(tape)-1} sq={sq} 相対誤差 {d:.1e} ✓")

    print("③ 融合の配当: 級数 exp(sed) 融合1カーネル vs 非融合(einsum×14回)")
    import time
    dev = torch.device('cuda')
    B = 200_000
    A = 0.3 * torch.randn(B, 16, device=dev)
    T16 = torch.zeros(16, 16, 16, device=dev)
    OMalg = cd_alg(16)
    T16 = torch.tensor(np.transpose(OMalg.T, (2, 0, 1)).copy(), dtype=torch.float32,
                       device=dev)
    def unfused(Amat):
        xs = Amat * 0.25
        acc = torch.zeros_like(Amat); acc[:, 0] = exp_tape[0]
        term = torch.zeros_like(Amat); term[:, 0] = 1.0
        for kk in range(1, 13):
            term = torch.einsum('kij,bi,bj->bk', T16, term, xs)
            acc = acc + exp_tape[kk] * term
        for _ in range(2):
            acc = torch.einsum('kij,bi,bj->bk', T16, acc, acc)
        return acc
    kf = compile_pipeline(impl("sedenion_naive"), "series", tape=exp_tape,
                          order=12, squarings=2, scale=0.25)
    def gb(f, n=10):
        f(); torch.cuda.synchronize(); ts = []
        for _ in range(n):
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); f(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return float(np.median(ts))
    t_un = gb(lambda: unfused(A))
    t_fu = gb(lambda: kf(A))
    print(f"   sed(R=256): 非融合 {t_un:.2f}ms → 融合 {t_fu:.2f}ms = {t_un/t_fu:.1f}倍"
          f" ← 計算律速(R大)では融合の配当なし=正直")
    # ★相乗効果の本命: 低ランクIMPLS(R=8のWH) × 融合Oプログラム
    A8 = 0.3 * torch.randn(B, 8, device=dev)
    X8 = xor_alg(8)
    T8 = torch.tensor(np.transpose(X8.T, (2, 0, 1)).copy(), dtype=torch.float32, device=dev)
    def unfused8(Amat):
        xs = Amat * 0.25
        acc = torch.zeros_like(Amat); acc[:, 0] = exp_tape[0]
        term = torch.zeros_like(Amat); term[:, 0] = 1.0
        for kk in range(1, 13):
            term = torch.einsum('kij,bi,bj->bk', T8, term, xs)
            acc = acc + exp_tape[kk] * term
        for _ in range(2):
            acc = torch.einsum('kij,bi,bj->bk', T8, acc, acc)
        return acc
    kf8 = compile_pipeline(impl("xor8_wh"), "series", tape=exp_tape,
                           order=12, squarings=2, scale=0.25)
    ref8 = unfused8(A8[:200])
    got8 = kf8(A8[:200])
    assert float((got8 - ref8).abs().max()) < 1e-3
    t_un8 = gb(lambda: unfused8(A8))
    t_fu8 = gb(lambda: kf8(A8))
    print(f"   ★xor8×WH(R=8): 非融合 {t_un8:.2f}ms → 融合 {t_fu8:.2f}ms = {t_un8/t_fu8:.1f}倍"
          f" ({B/t_fu8*1000/1e6:.0f}M exp/s) ← 低ランクIMPLS×融合の相乗効果")
    print("done — M(implicit)×N(任意UVW)×O(積/級数テープ) が 1 ソースから 融合コンパイルされた")


if __name__ == "__main__":
    self_test()

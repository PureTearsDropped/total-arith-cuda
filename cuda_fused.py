#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""cuda_fused — group_mul の 融合カーネル (Triton)。速さは 買うが 意味論は 1 ビットも 売らない。

  現行 group_mul の フラグあり経路は 出力成分 k ごとの Python ループで 数十個の カーネルを
  起動し、毎回 HBM を 往復する。本カーネルは 値の MAC(float64 蓄積・飽和は 最後に 1 回) と
  パターン則フラグ(P0〜P4・確実ゼロ/危険ゼロ・E1 保持・符号一致判定) を **1 カーネル**に
  融合する — データは レジスタに 載ったまま 最後まで 処理され、HBM は 入出力の 1 往復だけ。

  誠実さの 契約: 出力は cuda_total.group_mul と **フラグはビット一致・値は float32 一致**。
  self_test が 敵対的バッテリ(危険ゼロ・NaN 由来 (0,GE|LE|SUNK)・巨大/微小・全フラグ組合せ)で
  照合する。意味論の 定義は cuda_total.py が 唯一の 正であり、本カーネルは その 忠実な 影。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import triton
import triton.language as tl
from cuda_total import Tot, GE, LE, SUNK, group_mul, wiring_tensor

MAXF = torch.finfo(torch.float32).max
MINF = torch.finfo(torch.float32).tiny


@triton.jit
def _fused_kernel(Tptr, av_ptr, af_ptr, bv_ptr, bf_ptr, ov_ptr, of_ptr,
                  B, M: tl.constexpr, GEc: tl.constexpr, LEc: tl.constexpr,
                  SUNKc: tl.constexpr, MAXFc: tl.constexpr, MINFc: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B:
        return
    rows = tl.arange(0, M)                       # i (と 出力 k の 置き場)
    cols = tl.arange(0, M)                       # j
    base = pid * M
    av = tl.load(av_ptr + base + rows).to(tl.float64)
    bv = tl.load(bv_ptr + base + cols).to(tl.float64)
    fa = tl.load(af_ptr + base + rows).to(tl.int32)
    fb = tl.load(bf_ptr + base + cols).to(tl.int32)

    av2 = av[:, None]
    bv2 = bv[None, :]
    fa2 = fa[:, None]
    fb2 = fb[None, :]

    # 項ごとの 前計算(k に 依らない): 確実ゼロ/危険ゼロ/生存・E1 フラグ・符号
    defz = ((av2 == 0.0) & ((fa2 & GEc) == 0)) | ((bv2 == 0.0) & ((fb2 & GEc) == 0))
    dngr = (~defz) & (((av2 == 0.0) & ((fa2 & GEc) > 0)) | ((bv2 == 0.0) & ((fb2 & GEc) > 0)))
    # _mul_flags (E1 保守則): ge=(ga|gb)&~(la|lb), le=(la|lb)&~(ga|gb), nb=両方, SUNK 伝播
    ga = fa2 & GEc
    la = (fa2 >> 1) & 1
    gb = fb2 & GEc
    lb = (fb2 >> 1) & 1
    ge1 = ((ga | gb) & (1 - (la | lb)))
    le1 = ((la | lb) & (1 - (ga | gb)))
    nb1 = ((ga | gb) & (la | lb))
    tfall = ge1 * GEc + le1 * LEc + nb1 * (GEc + LEc) + ((fa2 | fb2) & SUNKc)
    sgn_a = tl.where(av2 > 0, 1.0, tl.where(av2 < 0, -1.0, 0.0))
    sgn_b = tl.where(bv2 > 0, 1.0, tl.where(bv2 < 0, -1.0, 0.0))

    out_v = tl.zeros((M,), dtype=tl.float64)
    out_f = tl.zeros((M,), dtype=tl.int32)
    for k in tl.static_range(0, M):
        Tk = tl.load(Tptr + k * M * M + rows[:, None] * M + cols[None, :])
        Tk64 = Tk.to(tl.float64)
        exists = Tk != 0.0
        raw_k = tl.sum(tl.sum(Tk64 * av2 * bv2, 1), 0)

        dead = defz & exists
        dt = dngr & exists
        live = exists & (~defz) & (~dngr)
        danger = tl.sum(tl.sum(dt.to(tl.int32), 1), 0) > 0
        tf = tl.where(live, tfall, 0)
        inb = tl.where(live, fa2 | fb2, 0)
        touched = tl.sum(tl.sum(((tf | inb) > 0).to(tl.int32), 1), 0) > 0
        sunk_any = tl.sum(tl.sum((live & (((fa2 | fb2) & SUNKc) > 0)).to(tl.int32), 1), 0) > 0
        n_live = tl.sum(tl.sum(live.to(tl.int32), 1), 0)
        ss = tl.where(Tk64 > 0, 1.0, tl.where(Tk64 < 0, -1.0, 0.0))
        tsgn = ss * sgn_a * sgn_b
        smax = tl.max(tl.max(tl.where(live, tsgn, -2.0), 1), 0)
        smin = tl.min(tl.min(tl.where(live, tsgn, 2.0), 1), 0)
        same_sign = smax == smin
        any_le_t = tl.sum(tl.sum(((tf & LEc) > 0).to(tl.int32), 1), 0) > 0
        any_ge_t = tl.sum(tl.sum(((tf & GEc) > 0).to(tl.int32), 1), 0) > 0
        ge_ok = ~any_le_t
        le_ok = ~any_ge_t
        f2 = (tl.where(ge_ok & any_ge_t, GEc, 0) | tl.where(le_ok & any_le_t, LEc, 0)
              | tl.where((~ge_ok) & (~le_ok), GEc + LEc, 0))
        keep = ((~sunk_any) & (same_sign | (n_live <= 1))) | (sunk_any & (n_live == 1))
        p0 = ~touched
        fk = tl.where(p0, 0, tl.where(keep, f2, GEc + LEc))
        sk = (~p0) & ((~keep) | (sunk_any & (n_live == 1)))
        fk = tl.where(danger, GEc + LEc, fk)
        sk = sk | danger
        fk = fk | tl.where(sk, SUNKc, 0)

        mask_k = rows == k
        out_v = out_v + tl.where(mask_k, raw_k, 0.0)
        out_f = out_f | tl.where(mask_k, fk, 0)

    # 飽和(最後に 1 回): NaN→(0, 境界なし+SUNK) / 溢れ→±MAX+GE / 潰れ→±MIN+LE
    isnan = out_v != out_v
    s = tl.where(out_v > 0, 1.0, tl.where(out_v < 0, -1.0, 0.0))
    a_abs = tl.abs(out_v)
    over = (a_abs > MAXFc) & (~isnan)
    under = (a_abs > 0) & (a_abs < MINFc) & (~isnan)
    val = tl.where(isnan, 0.0, tl.where(over, s * MAXFc, tl.where(under, s * MINFc, out_v)))
    sflag = (tl.where(isnan, GEc + LEc + SUNKc, 0) | tl.where(over, GEc, 0)
             | tl.where(under, LEc, 0))
    tl.store(ov_ptr + base + rows, val.to(tl.float32))
    tl.store(of_ptr + base + rows, (out_f | sflag).to(tl.uint8))


def fused_group_mul(T, a: Tot, b: Tot) -> Tot:
    """cuda_total.group_mul と 同一意味論の 融合版。値: float32 一致 / フラグ: ビット一致。"""
    M = T.shape[0]
    shp = a.val.shape
    av = a.val.reshape(-1, M).contiguous()
    bv = b.val.reshape(-1, M).contiguous()
    af = a.flag.reshape(-1, M).contiguous()
    bf = b.flag.reshape(-1, M).contiguous()
    B = av.shape[0]
    ov = torch.empty_like(av)
    of = torch.empty((B, M), dtype=torch.uint8, device=av.device)
    _fused_kernel[(B,)](T.contiguous(), av, af, bv, bf, ov, of,
                        B, M=M, GEc=int(GE), LEc=int(LE), SUNKc=int(SUNK),
                        MAXFc=MAXF, MINFc=MINF)
    return Tot(ov.reshape(shp), of.reshape(shp))


# ---------------------------------------------------------------- TOTALIZE の 融合 (税関 1 カーネル)
@triton.jit
def _totalize_kernel(x_ptr, v_ptr, f_ptr, n, BLOCK: tl.constexpr,
                     GEc: tl.constexpr, LEc: tl.constexpr, SUNKc: tl.constexpr,
                     MAXFc: tl.constexpr, MINFc: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float64)
    isnan = x != x
    s = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
    a = tl.abs(x)
    over = (a > MAXFc) & (~isnan)
    under = (a > 0) & (a < MINFc) & (~isnan)
    val = tl.where(isnan, 0.0, tl.where(over, s * MAXFc, tl.where(under, s * MINFc, x)))
    fl = (tl.where(isnan, GEc + LEc + SUNKc, 0) | tl.where(over, GEc, 0)
          | tl.where(under, LEc, 0))
    tl.store(v_ptr + offs, val.to(tl.float32), mask=mask)
    tl.store(f_ptr + offs, fl.to(tl.uint8), mask=mask)


def fused_totalize(x) -> Tot:
    """Tot(x) の 融合版 (入口 全域化を 1 カーネルで)。意味論の 正は cuda_total.Tot —
       値 float32 一致・フラグ ビット一致。獲物は 小バッチの カーネル起動 固定費
       (未融合 Tot() は torch 小演算 ~10 個を 別々に 起動する)。"""
    xf = x.reshape(-1).contiguous()
    n = xf.numel()
    v = torch.empty(n, dtype=torch.float32, device=xf.device)
    f = torch.empty(n, dtype=torch.uint8, device=xf.device)
    BLOCK = 1024
    _totalize_kernel[((n + BLOCK - 1) // BLOCK,)](
        xf, v, f, n, BLOCK=BLOCK, GEc=int(GE), LEc=int(LE), SUNKc=int(SUNK),
        MAXFc=MAXF, MINFc=MINF)
    return Tot(v.reshape(x.shape), f.reshape(x.shape))


# ============================================================ 要素演算 ekernel の 融合
@triton.jit
def _ekernel_kernel(X, C, OV, OF, n, ORDER: tl.constexpr, SHIFT: tl.constexpr,
                    TOT: tl.constexpr, BLOCK: tl.constexpr):
    """級数 全段 + ステップごと 全域化を レジスタ内で 1 パス。ekernel_gpu (⟺ numpy ekernel) と
       同じ 蓄積順序 (P←P·x, acc←acc+c_k·P, 毎ステップ nan→0+SING / ∞→±MAX+OVER)。"""
    MAXF64 = 1.7976931348623157e308
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(X + offs, mask=m, other=1.0).to(tl.float64)
    f = tl.zeros((BLOCK,), dtype=tl.int32)
    if TOT:                                                   # 入口税関
        nan = x != x
        f = f | tl.where(nan, 1, 0)
        x = tl.where(nan, 0.0, x)
        ovf = tl.abs(x) > MAXF64
        f = f | tl.where(ovf, 4, 0)
        x = tl.where(ovf, tl.where(x > 0, MAXF64, -MAXF64), x)
    if SHIFT:
        x = x - 1.0
    acc = tl.zeros((BLOCK,), dtype=tl.float64) + tl.load(C + 0)
    P = tl.zeros((BLOCK,), dtype=tl.float64) + 1.0
    for k in tl.static_range(1, ORDER + 1):
        P = P * x
        if TOT:
            nan = P != P
            f = f | tl.where(nan, 1, 0)
            P = tl.where(nan, 0.0, P)
            ovf = tl.abs(P) > MAXF64
            f = f | tl.where(ovf, 4, 0)
            P = tl.where(ovf, tl.where(P > 0, MAXF64, -MAXF64), P)
        ck = tl.load(C + k)
        nz = ck != 0.0                                        # 係数 0 は 完全スキップ (numpy と 同一)
        # FMA 縮約の 阻止: mul と add を 別々に 丸めないと numpy/torch と 最大 3ulp ずれる
        # (実測 62165/1e6 成分)。ランタイム条件の select を 挟むと LLVM は mul→add を
        # fma に 融合できない — ビット一致契約の ための 意図的な 遠回り。
        pc = tl.where(n >= 0, P * ck, 0.0)
        acc2 = acc + pc
        if TOT:
            nan2 = acc2 != acc2
            f = f | tl.where(nz & nan2, 1, 0)
            acc2 = tl.where(nan2, 0.0, acc2)
            ovf2 = tl.abs(acc2) > MAXF64
            f = f | tl.where(nz & ovf2, 4, 0)
            acc2 = tl.where(ovf2, tl.where(acc2 > 0, MAXF64, -MAXF64), acc2)
        acc = tl.where(nz, acc2, acc)
    tl.store(OV + offs, acc, mask=m)
    tl.store(OF + offs, f.to(tl.uint8), mask=m)


_EK_COEFS = {}

def fused_ekernel(name, x, order=None, tot=True):
    """cuda_total.ekernel_gpu の 融合版: 級数 全段 + 全域化が レジスタ内で 完結し、HBM は
       入出力の 1 往復だけ (未融合は 級数 1 段 ≈ カーネル 数個 × order)。契約: 値・成分ごと旗
       とも ekernel_gpu と ビット一致 (self_test が 恒久検査) — つまり numpy ekernel とも 一致。
       tot=False は 全域化なしの 対照 (誠実さ税の 測定用)。candidate の 検算は 出口で 数パス。
       **自作テープtoo 融合で 走る**: name に (キー文字列, テープ関数, 次数, shift) の 組を
       渡すと 任意の 級数 (合成式の テイラー係数 等) が 同じ 1 パス評価器に 載る — O 層の
       開放が 融合カーネルまで 通る (torch は 関数ごとに 専用カーネルだが、こちらは 評価器
       1 個 × テープ差し替え)。"""
    from nested_registry import OPS
    from cuda_total import _tot64
    if isinstance(name, (tuple, list)):
        keyname, tapefn, ordr, shift = name
        kind = "forward"
    else:
        op = OPS[name]
        keyname, tapefn = name, op["tape"]
        ordr = order or op["order"]
        shift, kind = bool(op["shift"]), op["kind"]
    key = (keyname, ordr)
    if key not in _EK_COEFS:
        _EK_COEFS[key] = torch.tensor([float(tapefn(k)) for k in range(ordr + 1)],
                                      dtype=torch.float64, device="cuda")
    xT = x if torch.is_tensor(x) else torch.as_tensor(
        __import__("numpy").asarray(x, float), dtype=torch.float64, device="cuda")
    n = xT.numel()
    ov = torch.empty_like(xT)
    of = torch.empty(n, dtype=torch.uint8, device=xT.device)
    BLOCK = 1024
    _ekernel_kernel[((n + BLOCK - 1) // BLOCK,)](
        xT, _EK_COEFS[key], ov, of, n, ORDER=ordr, SHIFT=shift,
        TOT=bool(tot), BLOCK=BLOCK)
    if kind == "forward" or not tot:
        return ov, of
    vt, _ = _tot64(xT, torch.zeros(n, dtype=torch.uint8, device=xT.device))
    if name == "log":
        resid = (fused_ekernel("exp", ov)[0] - vt).abs()
    elif name == "sqrt":
        resid = (ov * ov - vt).abs()
    elif name == "cbrt":
        resid = (ov * ov * ov - vt).abs()
    else:
        resid = (ov * vt - 1.0).abs()
    return ov, of | (~(resid < 1e-6)).to(torch.uint8) * 0x08


# ---------------------------------------------------------------- self-test: 影は本体と一致するか
def self_test():
    dev = torch.device('cuda')
    torch.manual_seed(0)
    print("cuda_fused — 融合カーネル vs cuda_total.group_mul (意味論の照合)")
    for kind, M in (("cd", 4), ("cd", 16), ("cyclic", 8)):
        T = wiring_tensor(kind, M, dev)
        B = 20000
        # 敵対的バッテリ: 普通・ゼロ多数・危険ゼロ(0,GE)・NaN由来(0,7)・巨大/微小・全フラグ
        v = torch.randn(B, M, device=dev, dtype=torch.float64)
        v[torch.rand(B, M, device=dev) < 0.25] = 0.0
        v[torch.rand(B, M, device=dev) < 0.05] *= 1e30
        v[torch.rand(B, M, device=dev) < 0.05] *= 1e-30
        f = torch.randint(0, 8, (B, M), device=dev, dtype=torch.uint8)
        f[torch.rand(B, M, device=dev) < 0.3] = 0
        w = torch.randn(B, M, device=dev, dtype=torch.float64)
        w[torch.rand(B, M, device=dev) < 0.25] = 0.0
        g = torch.randint(0, 8, (B, M), device=dev, dtype=torch.uint8)
        g[torch.rand(B, M, device=dev) < 0.3] = 0
        a = Tot(v); a = Tot(a.val, f)                       # 値は入口全域化・フラグは注入
        b = Tot(w); b = Tot(b.val, g)
        ref = group_mul(T, a, b)
        got = fused_group_mul(T, a, b)
        vdiff = (ref.val != got.val)
        both_nan_free = True
        nv = int(vdiff.sum())
        nf = int((ref.flag != got.flag).sum())
        # float64 の 総和順序差で float32 が 変わる 稀ケースを 許容せず まず 報告する
        print(f"  {kind}{M}: 値の不一致 {nv}/{B*M}  フラグの不一致 {nf}/{B*M}"
              f"  {'✓ 完全一致' if nv == 0 and nf == 0 else '← 要調査'}")
        assert nf == 0, "フラグは 1 ビットも 違ってはならない"
        assert nv == 0, "値は float32 で 一致するはず(総和順序差が出たら報告)"
    # フラグなし高速路も 同一
    T = wiring_tensor("cd", 16, dev)
    a = Tot(torch.randn(5000, 16, device=dev, dtype=torch.float64))
    b = Tot(torch.randn(5000, 16, device=dev, dtype=torch.float64))
    ref, got = group_mul(T, a, b), fused_group_mul(T, a, b)
    assert int((ref.val != got.val).sum()) == 0 and int((ref.flag != got.flag).sum()) == 0
    print("  フラグなし入力でも 完全一致 ✓")
    # TOTALIZE の 融合も 影: 敵対的バッテリ (NaN・±Inf・範囲外・非正規化・真の零・普通)
    B = 200000
    x = torch.randn(B, device=dev, dtype=torch.float64)
    x[torch.rand(B, device=dev) < 0.05] = float('nan')
    x[torch.rand(B, device=dev) < 0.05] = float('inf')
    x[torch.rand(B, device=dev) < 0.05] = float('-inf')
    x[torch.rand(B, device=dev) < 0.05] *= 1e300                    # f32 範囲外
    x[torch.rand(B, device=dev) < 0.05] *= 1e-300                   # f32 未満
    x[torch.rand(B, device=dev) < 0.10] = 0.0
    ref, got = Tot(x), fused_totalize(x)
    nv = int((ref.val != got.val).sum()); nf = int((ref.flag != got.flag).sum())
    print(f"  fused_totalize: 値の不一致 {nv}/{B}  フラグの不一致 {nf}/{B}"
          f"  {'✓ 完全一致' if nv == 0 and nf == 0 else '← 要調査'}")
    assert nv == 0 and nf == 0
    print("★ 融合カーネル = 本体の忠実な影 (フラグ ビット一致・値 float32 一致)")
    # ekernel の 融合too 影: 値・成分ごと旗 とも ekernel_gpu と ビット一致か
    from cuda_total import ekernel_gpu
    import numpy as _np
    rngk = _np.random.default_rng(11)
    sck = rngk.standard_normal(1_000_000) * 0.4 + 1.0
    sck[7] = 9.0                                              # 検算破れ → INEXACT
    sck[11] = float("nan"); sck[13] = float("inf")            # 税関 → SING / OVER
    for opn in ("exp", "sqrt", "log"):
        gv, gf = ekernel_gpu(opn, sck)
        fv, ff = fused_ekernel(opn, sck)
        nv = int((gv != fv).sum()); nf = int((gf != ff).sum())
        print(f"  fused_ekernel {opn}: 値の不一致 {nv}/1e6  旗の不一致 {nf}/1e6"
              f"  {'✓ 完全一致' if nv == 0 and nf == 0 else '← 要調査'}")
        assert nv == 0 and nf == 0
    import time as _t
    big = torch.as_tensor(rngk.standard_normal(10_000_000) * 0.4 + 1.0,
                          dtype=torch.float64, device=dev)
    def _g(fn):
        fn(); torch.cuda.synchronize()
        t0 = _t.perf_counter(); fn(); torch.cuda.synchronize()
        return _t.perf_counter() - t0
    t_un = _g(lambda: ekernel_gpu("exp", big))
    t_fu = _g(lambda: fused_ekernel("exp", big))
    t_fr = _g(lambda: fused_ekernel("exp", big, tot=False))
    t_lm = _g(lambda: torch.exp(big))
    print(f"  1e7成分 exp: 未融合 {t_un*1e3:.1f}ms → 融合 {t_fu*1e3:.2f}ms ({t_un/t_fu:.0f}×)"
          f" / 全域化なし {t_fr*1e3:.2f}ms → レジスタ内の 誠実さ税 {t_fu/t_fr:.2f}×"
          f" / torch.exp(libm) {t_lm*1e3:.2f}ms → 差 {t_fu/t_lm:.1f}×")


def benchmark():
    """実測 (RTX 5090, cd16, フラグあり): 融合の獲物は スループットでなく レイテンシ。
       既存フラグ路は 出力成分ごとの .nonzero() が GPU↔CPU 同期を 強制し、バッチに 依らない
       ~36ms の 床を 持つ。融合カーネルは それを 消す:
         B=1:      36.5ms → 63µs  (582倍)   ← 1kHz 制御ループに 入るのは 融合だけ
         B=1024:   387倍 / B=16k: 32倍 / B=1M: 1.0倍(同等)
       巨大バッチの クリーン経路は einsum が 既に メモリ最適で 融合版が 負ける(0.2倍) —
       そこは 既存の 高速路を 使うこと。用途: 制御ループ・逐次推定・小バッチ推論。"""
    import time
    dev = torch.device('cuda')
    T = wiring_tensor("cd", 16, dev)
    torch.manual_seed(0)

    def bench(fn, n):
        fn(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / n

    print(f"{'B':>10} {'既存(フラグあり)':>16} {'融合':>12} {'速度比':>8}")
    for B in (1, 64, 1024, 16384, 262144, 1000000):
        a = Tot(torch.randn(B, 16, device=dev, dtype=torch.float64))
        b = Tot(torch.randn(B, 16, device=dev, dtype=torch.float64))
        af = torch.randint(0, 8, (B, 16), device=dev, dtype=torch.uint8)
        bf = torch.randint(0, 8, (B, 16), device=dev, dtype=torch.uint8)
        a, b = Tot(a.val, af), Tot(b.val, bf)
        n = 50 if B <= 16384 else 5
        t1 = bench(lambda: group_mul(T, a, b), n)
        t2 = bench(lambda: fused_group_mul(T, a, b), n)
        print(f"{B:>10,} {t1*1e3:>13.2f}ms {t2*1e3:>9.2f}ms {t1/t2:>7.1f}倍")


if __name__ == "__main__":
    self_test()
    benchmark()

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
    print("★ 融合カーネル = 本体の忠実な影 (フラグ ビット一致・値 float32 一致)")


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

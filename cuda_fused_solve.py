#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""cuda_fused_solve — nsolve (方程式を解く除算 L_a⁺x) の 融合カーネル (Triton)。

  1 プログラム = 1 要素の 完全な solve が レジスタ内で 完結する:
    ① L_a の構築 (配線表 T から・積なしの 重み和)
    ② Ben-Israel 反復 X ← X(2I − L X) × K 回 (乗算のみ・判断/ピボット/除算なし)
    ③ y = X·x
    ④ 二層検算: 前向き残差 ‖L y − x‖ → 厳密解 / 正規方程式 ‖Lᵀ(Ly−x)‖ → 最小二乗(SING)
       / どちらも 不成立 → SING|INEXACT — 「解けたフリ」を カーネル内で 構造的に 禁止
    ⑤ 入力フラグの OR 伝播

  素朴実装は 反復ごとに カーネル起動+HBM 往復(K=25 → 50 回の bmm)。融合版は 1 起動・
  HBM は 入出力 1 往復。tl.dot は input_precision="ieee" (tf32 の 黙った精度落ちを 拒否)。

  意味論の 正: julia/NestedSeries.jl の nsolve_left (フラグ規約も 同じ)。
  検証: torch.linalg.pinv(float64) との 値照合 + 零因子バッテリでの フラグ照合。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import triton
import triton.language as tl
from cuda_total import Tot, GE, LE, SUNK, wiring_tensor

SING_F = GE | LE | SUNK          # 「一意の厳密解なし(最小二乗)」を 既存語彙で: 境界なし+SUNK
INEXACT_F = 0x08                 # 未収束/解なし検算不成立(HyperTranscend と 同じビット)


@triton.jit
def _solve_kernel(Tptr, av_ptr, af_ptr, xv_ptr, xf_ptr, yv_ptr, yf_ptr,
                  B, M: tl.constexpr, K: tl.constexpr,
                  TOL: tl.constexpr, SINGc: tl.constexpr, INEXc: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B:
        return
    r = tl.arange(0, M)
    base = pid * M
    a = tl.load(av_ptr + base + r)
    x = tl.load(xv_ptr + base + r)
    fa = tl.load(af_ptr + base + r).to(tl.int32)
    fx = tl.load(xf_ptr + base + r).to(tl.int32)
    inflag = tl.max(fa | fx, 0)                          # 入力フラグの OR (要素単位)

    # ① L[k,j] = Σ_i T[k,i,j]·a_i — 配線の 重み和 (積の 融合なし)
    L = tl.zeros((M, M), dtype=tl.float32)
    for i in tl.static_range(0, M):
        Ti = tl.load(Tptr + r[:, None] * M * M + i * M + r[None, :])
        ai = tl.load(av_ptr + base + i)
        L += ai * Ti

    # ② Ben-Israel: X₀ = Lᵀ/(‖L‖₁‖L‖∞) → X(2I−LX) ×K (すべて レジスタ内)
    n1 = tl.max(tl.sum(tl.abs(L), 0), 0)
    ninf = tl.max(tl.sum(tl.abs(L), 1), 0)
    d = n1 * ninf
    X = tl.where(d > 0, tl.trans(L) / (d + 1e-38), tl.zeros((M, M), dtype=tl.float32))
    I2 = tl.where(r[:, None] == r[None, :], 2.0, 0.0)
    for _ in tl.static_range(0, K):
        LX = tl.dot(L, X, input_precision="ieee")
        X = tl.dot(X, I2 - LX, input_precision="ieee")

    # ③ y = X·x
    y = tl.sum(X * x[None, :], 1)

    # ④ 二層検算 (相対残差・fp32 の 正直な 閾値)
    Ly = tl.sum(L * y[None, :], 1)
    res = Ly - x
    r1 = tl.max(tl.abs(res), 0)
    r2 = tl.max(tl.abs(tl.sum(L * res[:, None], 0)), 0)   # Lᵀ·res
    sx = tl.max(tl.abs(x), 0) + 1e-30
    sL = tl.max(tl.abs(L)) + 1e-30
    exact = (r1 / sx) < TOL
    lsq = (r2 / (sL * sx)) < TOL
    f = tl.where(exact, 0, tl.where(lsq, SINGc, SINGc | INEXc)) | inflag

    tl.store(yv_ptr + base + r, y)
    tl.store(yf_ptr + base + r, tl.full((M,), 1, tl.int32) * f)


def fused_solve(T, a, x, K=25, tol=1e-3):
    """solve a·y = x (バッチ)。a, x: Tot か 生テンソル (..., M)。
       返り値 Tot: フラグ = 0(厳密解) / SING(最小二乗のみ) / SING|INEXACT(未収束) | 入力OR"""
    M = T.shape[0]
    at = a if isinstance(a, Tot) else Tot(torch.as_tensor(a, dtype=torch.float64))
    xt = x if isinstance(x, Tot) else Tot(torch.as_tensor(x, dtype=torch.float64))
    shp = at.val.shape
    av = at.val.reshape(-1, M).contiguous()
    xv = xt.val.reshape(-1, M).contiguous()
    af = at.flag.reshape(-1, M).contiguous()
    xf = xt.flag.reshape(-1, M).contiguous()
    B = av.shape[0]
    yv = torch.empty_like(av)
    yf = torch.empty((B, M), dtype=torch.uint8, device=av.device)
    _solve_kernel[(B,)](T.contiguous(), av, af, xv, xf, yv, yf,
                        B, M=M, K=K, TOL=tol, SINGc=int(SING_F), INEXc=int(INEXACT_F))
    return Tot(yv.reshape(shp), yf.reshape(shp))


# ---------------------------------------------------------------- 参照実装(非融合・意味論の影)
def unfused_solve(T, av, K=25):
    "バッチ torch 実装 (毎反復 カーネル起動+HBM 往復) — ベンチの 比較対象・値の 二証人"
    M = T.shape[0]
    L = torch.einsum('kij,bi->bkj', T, av)
    n1 = L.abs().sum(1).amax(-1)
    ninf = L.abs().sum(2).amax(-1)
    d = (n1 * ninf).clamp(min=1e-38)
    X = L.transpose(1, 2) / d[:, None, None]
    I2 = 2 * torch.eye(M, device=av.device)
    for _ in range(K):
        X = torch.bmm(X, I2 - torch.bmm(L, X))
    return L, X


def self_test():
    dev = torch.device('cuda')
    torch.manual_seed(0)
    T = wiring_tensor('cd', 16, dev)
    print("cuda_fused_solve — 意味論の照合 (正: nsolve/pinv・フラグ: 二層検算)")

    B = 4096
    av = torch.randn(B, 16, device=dev, dtype=torch.float64)
    xv = torch.randn(B, 16, device=dev, dtype=torch.float64)
    # 零因子を 混ぜる: e3+e10 のスカラー倍 (rank 12/16)
    zi = torch.arange(0, B, 8)
    av[zi] = 0.0
    av[zi, 3] = 1.0; av[zi, 10] = 1.0
    y = fused_solve(T, av, xv)

    # ① 正則要素: 前向き残差ゼロ級・pinv(float64) と 一致・フラグ clean
    L64 = torch.einsum('kij,bi->bkj', T.double(), av)
    y_ref = torch.linalg.pinv(L64) @ xv[..., None]
    reg = torch.ones(B, dtype=torch.bool, device=dev); reg[zi] = False
    rel = (y.val.double()[reg] - y_ref[reg, :, 0]).abs().amax() / y_ref[reg].abs().amax()
    assert rel < 2e-3, f"正則の値照合 {rel}"
    assert int(y.flag[reg].amax()) == 0, "正則要素に 余計なフラグ"
    print(f"  ① 正則{int(reg.sum())}件: pinv(f64)と 相対 {rel:.1e} 一致・フラグ clean ✓")

    # ② 零因子 × 一般の x: 最小二乗 → SING (INEXACT なし)・pinv と 一致
    relz = (y.val.double()[zi] - y_ref[zi, :, 0]).abs().amax() / y_ref[zi].abs().amax()
    fz = y.flag[zi]
    assert relz < 2e-3, f"零因子の値照合 {relz}"
    assert bool(((fz & SING_F) == SING_F).all()) and int((fz & INEXACT_F).max()) == 0
    print(f"  ② 零因子{len(zi)}件: 最小二乗解が pinv と {relz:.1e} 一致・SING 正直 ✓")

    # ③ 零因子 × range 内の x: 厳密解 → フラグ clean
    y0 = torch.randn(len(zi), 16, device=dev, dtype=torch.float64)
    x_in = (L64[zi] @ y0[..., None])[..., 0]
    y_in = fused_solve(T, av[zi], x_in)
    assert int(y_in.flag.amax()) == 0
    print(f"  ③ 零因子×range内x: 厳密解・フラグ clean ✓")

    # ④ NaN 毒入力: 入口 Tot が 名指し → 出力へ OR 伝播・値は 有限
    bad = av[:4].clone(); bad[0, 0] = float('nan')
    yb = fused_solve(T, bad, xv[:4])
    assert torch.isfinite(yb.val).all() and int(yb.flag[0].min()) > 0
    print(f"  ④ NaN毒: 出力有限・フラグ伝播 ✓")
    print("★ 融合solve = nsolve の忠実な影 (値: pinv二証人・フラグ: 二層検算+入力OR)")


def benchmark():
    """厳密計測 (外部レビュー 2026-07-21 の 指摘を 実装):
       CUDA イベント計測・ウォームアップ 5 回・中央値 + [p10, p90]・同期は 計測区間の 外。
       計測対象は solve 本体のみ (入力生成・Tot 化・フラグ初期化は 区間外 — 併記)。
       環境: GPU 機種は 実行時に 印字・fp32 (tl.dot ieee)・K=25 固定。コンパイルは
       ウォームアップで 除外。"""
    import numpy as np
    dev = torch.device('cuda')
    torch.manual_seed(0)
    T = wiring_tensor('cd', 16, dev)
    print(f"\nGPU: {torch.cuda.get_device_name(0)} / fp32(ieee) / K=25 / 計測=solve本体のみ")

    def bench(fn, n_rep):
        for _ in range(5): fn()                            # ウォームアップ(コンパイル込み)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n_rep):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))                 # ms
        a = np.array(ts)
        return np.median(a), np.quantile(a, 0.1), np.quantile(a, 0.9)

    hdr = f"{'B':>9} {'pinv(torch)':>16} {'非融合BI':>16} {'融合':>16} {'vs pinv':>8} {'vs 非融合':>9}"
    print(hdr)
    for B in (64, 1024, 16384, 262144, 1000000):
        av = torch.randn(B, 16, device=dev, dtype=torch.float64)
        xv = torch.randn(B, 16, device=dev, dtype=torch.float64)
        at, xt = Tot(av), Tot(xv)                          # 入口は 計測区間の 外
        a32 = av.float(); x32 = xv.float()
        L32 = torch.einsum('kij,bi->bkj', T, a32)
        n = 30 if B <= 16384 else 10
        p_m, p_l, p_h = bench(lambda: torch.bmm(torch.linalg.pinv(L32), x32[..., None]), n)
        u_m, u_l, u_h = bench(lambda: unfused_solve(T, a32), n)
        f_m, f_l, f_h = bench(lambda: fused_solve(T, at, xt), n)
        print(f"{B:>9,} {p_m:>8.2f}[{p_l:.2f},{p_h:.2f}] {u_m:>8.2f}[{u_l:.2f},{u_h:.2f}]"
              f" {f_m:>8.2f}[{f_l:.2f},{f_h:.2f}] {p_m/f_m:>7.1f}倍 {u_m/f_m:>8.1f}倍")
    print("(表示: 中央値[p10,p90] ms)")


def quality_report():
    """二層検算の 品質分布 (外部レビューの 指摘②): 相対残差
         r1 = ‖Ay−x‖/(‖x‖+ε)           … 前向き(厳密解の 主張)
         r2 = ‖Aᵀ(Ay−x)‖/(‖A‖‖Ay−x‖+ε) … 正規方程式(最小二乗の 主張)
       を フラグ階級別に 最大値・中央値・p99.9 で。フラグの 主張が 分布として 裏づくかの 検査。"""
    import numpy as np
    dev = torch.device('cuda')
    torch.manual_seed(1)
    T = wiring_tensor('cd', 16, dev)
    B = 200_000
    av = torch.randn(B, 16, device=dev, dtype=torch.float64)
    xv = torch.randn(B, 16, device=dev, dtype=torch.float64)
    zi = torch.arange(0, B, 4)                             # 25% を 零因子に
    av[zi] = 0.0; av[zi, 3] = 1.0; av[zi, 10] = 1.0
    y = fused_solve(T, av, xv)
    L = torch.einsum('kij,bi->bkj', T.double(), av)
    res = (L @ y.val.double()[..., None])[..., 0] - xv
    r1 = res.norm(dim=1) / (xv.norm(dim=1) + 1e-30)
    nr = (L.transpose(1, 2) @ res[..., None])[..., 0]
    r2 = nr.norm(dim=1) / (L.flatten(1).norm(dim=1) * res.norm(dim=1) + 1e-30)
    clean = (y.flag.amax(dim=1) == 0)
    sing = ~clean
    def stats(v):
        a = v.cpu().numpy()
        return f"max {a.max():.1e} / 中央値 {np.median(a):.1e} / p99.9 {np.quantile(a, 0.999):.1e}"
    print(f"\n品質分布 (B={B:,}, 零因子25%):")
    print(f"  clean({int(clean.sum()):,}件) の r1(前向き):   ", stats(r1[clean]))
    print(f"  SING ({int(sing.sum()):,}件) の r2(正規方程式): ", stats(r2[sing]))
    print(f"  SING の r1 (解なしの 正直な 大きさ):        ", stats(r1[sing]))
    assert float(r1[clean].max()) < 1e-3 and float(r2[sing].max()) < 1e-3
    print("  ★フラグの主張は 分布の 裾(p99.9・max)まで 裏づけられた")


if __name__ == "__main__":
    self_test()
    benchmark()
    quality_report()

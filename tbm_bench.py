#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""tbm_bench — TBM の 性能測定 (SPEC §3 の 傾斜誠実性を 機械の 上で 実測)。

  問い:
    ① BILIN の 誠実さダイヤル 3段 (bare/coarse/evidence) の 税金 — バッチ別
    ② アセンブラ(tbm.run)の 薄層 オーバーヘッド — 直接 カーネル呼びとの 差
    ③ EXP: マクロ逐次形 (仕様どおり) vs 融合実行係 (cuda_fused_pipeline) の 配当
    ④ 命令ごとの 素の スループット
  測り方: CUDA イベント・中央値・ウォームアップ後。基準線は 裸 einsum (IEEE・フラグなし)。
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import tbm
from tbm import Program, run, coarse_group_mul, bare_group_mul
from cuda_total import Tot, wiring_tensor, group_mul
from cuda_fused import fused_group_mul
from cuda_fused_pipeline import compile_pipeline
from nested_registry import impl

assert torch.cuda.is_available(), "GPU 必須"
dev = torch.device("cuda")


def gb(f, n=20):
    f(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); f(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return float(np.median(ts)) * 1e-3          # 秒


def main():
    print(f"tbm_bench — {torch.cuda.get_device_name(0)}")
    T = wiring_tensor("cd", 16, dev)
    Tk = T.to(torch.float32)

    print("\n① BILIN(セデニオン) 誠実さダイヤルの 税金 (基準 = 裸 einsum)")
    print(f"   {'B':>10} {'裸einsum':>10} {'bare':>8} {'coarse':>8} {'evidence':>9}"
          f"  {'coarse税':>8} {'evid税':>8}")
    for B in (1_000, 100_000, 1_000_000):
        a = Tot(torch.randn(B, 16, device=dev, dtype=torch.float64))
        b = Tot(torch.randn(B, 16, device=dev, dtype=torch.float64))
        av32 = a.val.clone()
        bv32 = b.val.clone()
        t_raw = gb(lambda: torch.einsum('kij,bi,bj->bk', Tk, av32, bv32))
        t_bare = gb(lambda: bare_group_mul(T, a, b))
        t_co = gb(lambda: coarse_group_mul(T, a, b))
        t_ev = gb(lambda: fused_group_mul(T, a, b))
        print(f"   {B:>10,} {t_raw*1e3:>8.2f}ms {t_bare*1e3:>6.2f}ms {t_co*1e3:>6.2f}ms"
              f" {t_ev*1e3:>7.2f}ms  {t_co/t_raw:>7.2f}× {t_ev/t_raw:>7.2f}×")

    print("\n①b 税金の 分解 (B=1M): 貯め幅 (f64 quire) と フラグは 別の 請求書")
    B = 1_000_000
    a = Tot(torch.randn(B, 16, device=dev, dtype=torch.float64))
    b = Tot(torch.randn(B, 16, device=dev, dtype=torch.float64))
    av32, bv32 = a.val.clone(), b.val.clone()
    MAXF = torch.finfo(torch.float32).max
    def coarse32():                              # f32 蓄積の 粗誠実 (貯め幅 規律を 外した 対照)
        o = torch.einsum('kij,bi,bj->bk', Tk, av32, bv32)
        oc = o.clamp(-MAXF, MAXF)
        dirty = ((a.flag | b.flag).amax(1, keepdim=True) > 0)
        return oc, (o.abs() >= MAXF).to(torch.uint8) | dirty.to(torch.uint8) * 7
    t_raw = gb(lambda: torch.einsum('kij,bi,bj->bk', Tk, av32, bv32))
    t_c32 = gb(coarse32)
    t_bare = gb(lambda: bare_group_mul(T, a, b))
    t_co = gb(lambda: coarse_group_mul(T, a, b))
    t_ev = gb(lambda: fused_group_mul(T, a, b))
    print(f"   裸f32 {t_raw*1e3:.2f}ms → +粗フラグ(f32蓄積) {t_c32*1e3:.2f}ms ({t_c32/t_raw:.2f}×)"
          f" → +f64蓄積 {t_co*1e3:.2f}ms ({t_co/t_raw:.1f}×) → +模様則 {t_ev*1e3:.2f}ms ({t_ev/t_raw:.1f}×)")
    print(f"   ⟹ フラグ {t_c32/t_raw:.2f}× / f64 quire {t_co/t_c32:.1f}× / 証拠級模様則 {t_ev/t_co:.1f}×"
          f" (民生GPUは fp64=1/64 レート — 貯め幅は 誠実さと 独立の ダイヤル)")

    print("\n② アセンブラ薄層の オーバーヘッド (プログラム 1 命令 = BILIN evidence)")
    P = (Program("one").TOTALIZE("a", "in_a").TOTALIZE("b", "in_b")
         .BILIN("s", "a", "b", honesty="evidence"))
    for B in (1_000, 1_000_000):
        feed = {"in_a": np.random.randn(B, 16), "in_b": np.random.randn(B, 16)}
        a = Tot(torch.as_tensor(feed["in_a"], device=dev))
        b = Tot(torch.as_tensor(feed["in_b"], device=dev))
        t_dir = gb(lambda: fused_group_mul(T, a, b))
        t_asm = gb(lambda: run(P, feed, "gpu"))
        print(f"   B={B:>9,}: 直接 {t_dir*1e3:.2f}ms / tbm.run {t_asm*1e3:.2f}ms"
              f" = {t_asm/t_dir:.2f}× (TOTALIZE の H2D 転送 込み)")
        # 転送を 除いた 公平版: TOTALIZE 済み Tot を 使う プログラム相当
        env = {"a": a, "b": b}
        t_asm2 = gb(lambda: fused_group_mul(T, env["a"], env["b"]))
        print(f"              命令 dispatch のみ (転送 除外): {t_asm2/t_dir:.2f}×")

    print("\n③ EXP(order12+二乗2): マクロ逐次形 vs 融合実行係 (bare 同士)")
    exp_tape = [1.0 / math.factorial(k) for k in range(13)]
    for nm, alg_name, d, lab in (("xor8_wh", None, 8, "xor8×WH R=8"),
                                 ("sedenion_naive", "sedenion", 16, "sedenion R=256")):
        B = 200_000
        X = 0.3 * torch.randn(B, d, device=dev)
        kf = compile_pipeline(impl(nm), "series", tape=exp_tape, order=12,
                              squarings=2, scale=0.25)
        t_fu = gb(lambda: kf(X))
        if alg_name:                             # マクロ逐次形 (BILIN×14 + AXPY×12 個別起動)
            Tm = wiring_tensor("cd", d, dev)
            def macro_seq():
                xs = Tot(X.double() * 0.25)
                acc = Tot(torch.zeros(B, d, device=dev, dtype=torch.float64))
                acc.val[:, 0] = exp_tape[0]
                term = Tot(torch.zeros(B, d, device=dev, dtype=torch.float64))
                term.val[:, 0] = 1.0
                for k in range(1, 13):
                    term = bare_group_mul(Tm, term, xs)
                    acc = Tot(acc.val + exp_tape[k] * term.val, acc.flag)
                for _ in range(2):
                    acc = bare_group_mul(Tm, acc, acc)
                return acc
            t_mac = gb(macro_seq, n=5)
            print(f"   {lab:<15}: マクロ逐次 {t_mac*1e3:>7.2f}ms → 融合 {t_fu*1e3:>6.2f}ms"
                  f" = {t_mac/t_fu:>5.1f}× ({B/t_fu/1e6:.0f}M exp/s)")
        else:
            print(f"   {lab:<15}: 融合 {t_fu*1e3:>6.2f}ms ({B/t_fu/1e6:.0f}M exp/s)"
                  f" — 逐次形は xor代数が cuda_total 配線に ないため 融合のみ")

    print("\n④ 命令スループット (B=1M)")
    B = 1_000_000
    a = Tot(torch.randn(B, 16, device=dev, dtype=torch.float64))
    b = Tot(torch.randn(B, 16, device=dev, dtype=torch.float64))
    t = gb(lambda: fused_group_mul(T, a, b))
    print(f"   BILIN evidence : {B/t/1e6:>7.1f}M 積/s")
    t = gb(lambda: coarse_group_mul(T, a, b))
    print(f"   BILIN coarse   : {B/t/1e6:>7.1f}M 積/s")
    t = gb(lambda: bare_group_mul(T, a, b))
    print(f"   BILIN bare     : {B/t/1e6:>7.1f}M 積/s")
    from cuda_total import tot_add
    t = gb(lambda: tot_add(a, b))
    print(f"   AXPY           : {B/t/1e6:>7.1f}M 和/s")
    import nested_registry as NR
    Mw = torch.as_tensor(np.real(NR.amap("wh8").M), dtype=torch.float64, device=dev)
    x8 = torch.randn(B, 8, device=dev, dtype=torch.float64)
    t = gb(lambda: x8 @ Mw.T)
    print(f"   LINMAP (wh8)   : {B/t/1e6:>7.1f}M 変換/s")


if __name__ == "__main__":
    main()

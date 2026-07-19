#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証
"""CUDA(torch) 全域算術 + 配線表ライブラリ — GPU 版の「配線＝計算」。

  ・数 = (val: float32, flag: uint8)。flag ビット: GE=1(≥) LE=2(≤) SUNK=4(符号不明)。
  ・全域化: overflow→±MAX+GE / underflow→±MIN(=ε・向き保持)+LE / a/0=0 / **NaN·Inf は 決して 出さない**。
  ・配線表 = 構造テンソル T[k,i,j]（σ(i,j)·δ_{k,i∘j}）。**T を 差し替えると 同じカーネルが
    複素/四元/セデニオン/行列積/畳み込みに 変わる**（wiring_registry の GPU 版）。
  ・群積/MAC は float64 で 貯めて **飽和（丸め）は 最後に 1 回**（multi_add/mul_fused の 哲学）。

  **ゲート版との 意味論の違い（正直な 但し書き）**: フラグは 全域化イベント（飽和±MAX/ε=±MIN/
  0除算/相殺）**のみ**。float32 の 通常丸め（最近接）は フラグしない — ゲート版は 切り捨てごとに
  ge を 立てるが、float の 最近接丸めは 方向を 持たないため 片側境界に できない。
  実測: 「フラグなしで float64 真値と 違う」320,019 件は 全件 float32 丸めで 説明・飽和フラグの嘘 0。
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch

GE, LE, SUNK = 1, 2, 4
F32 = torch.float32
MAX = torch.finfo(F32).max            # 飽和天井
MIN = torch.finfo(F32).tiny           # ε = 最小正規数（向き付き 無限小）


def _sat(raw64, dev):
    """float64 の 生値 → (float32 値, フラグ)。溢れ→±MAX+GE / 潰れ→±MIN+LE / NaN 出さない。"""
    sign = torch.sign(raw64)
    a = raw64.abs()
    over = a > MAX
    under = (a > 0) & (a < MIN)
    val = raw64.clone()
    val = torch.where(over, sign * MAX, val)
    val = torch.where(under, sign * MIN, val)
    flag = torch.zeros(raw64.shape, dtype=torch.uint8, device=dev)
    flag |= over.to(torch.uint8) * GE
    flag |= under.to(torch.uint8) * LE
    return val.to(F32), flag


class Tot:
    """全域数のテンソル: val float32・flag uint8（同形）。"""
    __slots__ = ('val', 'flag')
    def __init__(self, val, flag=None):
        self.val = val.to(F32)
        self.flag = torch.zeros(val.shape, dtype=torch.uint8, device=val.device) \
                    if flag is None else flag
    @property
    def device(self): return self.val.device


def _mul_flags(fa, fb):
    """積の フラグ合成（E1 保守則）: ≥·≥=≥, ≤·≤=≤, =·x=x, ≥·≤=境界なし。SUNK は 伝播。"""
    ga, la = fa & GE, (fa >> 1) & 1
    gb, lb = fb & GE, (fb >> 1) & 1
    ge_o = ((ga | gb) & ~(la | lb)) & 1            # どちらかが ≥・どちらも ≤ でない
    le_o = ((la | lb) & ~(ga | gb)) & 1
    nb = ((ga | gb) & (la | lb)) & 1               # 混在 → 境界なし
    out = (ge_o * GE) | (le_o * LE) | (nb * (GE | LE))
    out = out.to(torch.uint8) | ((fa | fb) & SUNK)
    return out

def tot_mul(a, b):
    raw = a.val.double() * b.val.double()
    # x×0=0 厳密（IEEE で 0×inf は 出ない: 入力に inf が 無い 不変条件）
    val, sflag = _sat(raw, a.device)
    return Tot(val, sflag | _mul_flags(a.flag, b.flag))

def tot_add(a, b):
    raw = a.val.double() + b.val.double()
    val, sflag = _sat(raw, a.device)
    # 飽和同士の 相殺（MAX−MAX 型）: 両入力 GE で 異符号 → 符号不明+境界なし（§8）
    clash = ((a.flag & GE) > 0) & ((b.flag & GE) > 0) & (torch.sign(a.val) != torch.sign(b.val)) \
            & (a.val != 0) & (b.val != 0)
    f = sflag | (a.flag | b.flag)                   # 保守合成（≥+≥同符号=≥ 等は 単純和で 健全側）
    f = f | clash.to(torch.uint8) * (SUNK | GE | LE)
    return Tot(val, f)

def tot_div(a, b):
    bz = (b.val == 0)
    raw = a.val.double() / torch.where(bz, torch.ones_like(b.val), b.val).double()
    raw = torch.where(bz, torch.zeros_like(raw), raw)          # a/0 = 0（Moore–Penrose）
    val, sflag = _sat(raw, a.device)
    fin = a.flag | b.flag
    nb = (fin & (GE | LE)) > 0                                  # 入力に 境界 → 商は 保守的に 境界なし
    f = sflag | torch.where(nb, torch.full_like(fin, GE | LE), torch.zeros_like(fin)) \
        | (fin & SUNK)
    return Tot(val, f)


# ---------------------------------------------------------------- 配線表（構造テンソル）
def wiring_tensor(kind, M, device):
    """配線表 T[k,i,j]。kind: 'cd'（Cayley–Dickson XOR経路）/ 'cyclic'（巡回畳み込み）。"""
    T = torch.zeros(M, M, M, dtype=torch.float32, device=device)
    if kind == "cd":
        from nd_algebra import cd_omega
        OM = cd_omega(M)
        for i in range(M):
            for j in range(M):
                T[i ^ j, i, j] = float(OM[i, j])
    elif kind == "cyclic":
        for i in range(M):
            for j in range(M):
                T[(i + j) % M, i, j] = 1.0
    else:
        raise ValueError(kind)
    return T

def group_mul(T, a, b):
    """配線積（バッチ）: c[...,k] = Σ_ij T[k,i,j]·a[...,i]·b[...,j]。
       float64 で 貯めて 飽和は 最後に 1 回（融合 MAC の 哲学）。"""
    raw = torch.einsum('kij,...i,...j->...k', T.double(), a.val.double(), b.val.double())
    val, sflag = _sat(raw, a.val.device)
    fin = (a.flag | b.flag)
    anyf = (fin & (GE | LE)) > 0
    anyf = anyf.any(dim=-1, keepdim=True).expand(sflag.shape)   # 成分間で 混ざる ⟹ 保守的
    f = sflag | anyf.to(torch.uint8) * (GE | LE)
    return Tot(val, f)


# ---------------------------------------------------------------- 自己テスト
def self_test():
    import numpy as np
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'})")
    rng = np.random.default_rng(20260810)

    print("=" * 76)
    print("① 全域化: NaN/Inf を 決して 出さない・フラグは 嘘をつかない（敵対的）")
    print("=" * 76)
    N = 1_000_000
    # 敵対的入力: 極大・極小・ゼロ・普通 を 混ぜる
    pool = np.concatenate([
        rng.uniform(-1e38, 1e38, N // 4), rng.uniform(-1e-38, 1e-38, N // 4),
        np.zeros(N // 4), rng.standard_normal(N - 3 * (N // 4)) ])
    rng.shuffle(pool)
    av = torch.tensor(pool[:N], dtype=F32, device=dev)
    bv = torch.tensor(np.roll(pool[:N], 7), dtype=F32, device=dev)
    A, B = Tot(av), Tot(bv)
    bad_naninf = 0; lies = 0
    for name, op in [("mul", tot_mul), ("add", tot_add), ("div", tot_div)]:
        r = op(A, B)
        bad_naninf += int(torch.isnan(r.val).sum() + torch.isinf(r.val).sum())
        # 真値（float64・inf 可）と 照合: GE⟹|真|≥|表示|, LE⟹|真|≤|表示|, 無フラグ⟹一致
        a64, b64 = av.double(), bv.double()
        t = {"mul": a64 * b64, "add": a64 + b64,
             "div": torch.where(bv == 0, torch.zeros_like(a64), a64 / b64.where(b64 != 0, torch.ones_like(b64)))}[name]
        ge = (r.flag & GE) > 0; le = (r.flag & LE) > 0; ex = r.flag == 0
        lies += int((ge & ~le & (t.abs() < r.val.double().abs())).sum())        # GE: |真|≥|表示| か
        lies += int((le & ~ge & (t.abs() > r.val.double().abs())).sum())        # LE: |真|≤|表示| か
        # 無フラグ = float32 に 丸めた 真値と 一致（丸めは フラグ対象外・docstring の 意味論）
        lies += int((ex & (t.to(F32).double() != r.val.double())).sum())
    print(f"  {N:,} 件 × mul/add/div: NaN/Inf **{bad_naninf}**・フラグの嘘 **{lies}**"
          f"（意味論: 飽和/ε/0除算のみ フラグ・最近接丸めは 対象外）")

    print()
    print("=" * 76)
    print("② 配線表の 差し替え = 同じカーネルが 別の代数に（GPU 版 wiring_registry）")
    print("=" * 76)
    from nd_algebra import cd_omega, ref_mult_M
    for kind, M, name in [("cd", 2, "複素"), ("cd", 4, "四元数"), ("cd", 16, "セデニオン"),
                          ("cyclic", 8, "巡回畳み込みZ/8")]:
        T = wiring_tensor(kind, M, dev)
        bad = 0
        for _ in range(200):
            a = rng.integers(-9, 10, M).astype(np.float32)
            b = rng.integers(-9, 10, M).astype(np.float32)
            c = group_mul(T, Tot(torch.tensor(a, device=dev)), Tot(torch.tensor(b, device=dev)))
            got = [int(v) for v in c.val.cpu().numpy()]
            if kind == "cd":
                ref = ref_mult_M([int(x) for x in a], [int(x) for x in b], cd_omega(M), M)
            else:
                ref = [int(sum(a[i] * b[(k - i) % M] for i in range(M))) for k in range(M)]
            if got != ref: bad += 1
        print(f"  {name:<14} M={M:>2}: 違反 {bad}/200 {'✓' if bad == 0 else '×'}")

    print()
    print("=" * 76)
    print("③ スループット（5090・セデニオン積 バッチ）— 参考値")
    print("=" * 76)
    import time
    M = 16; T = wiring_tensor("cd", M, dev)
    for NB in (10_000, 1_000_000):
        a = Tot(torch.randn(NB, M, device=dev))
        b = Tot(torch.randn(NB, M, device=dev))
        group_mul(T, a, b)                                     # ウォームアップ
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            group_mul(T, a, b)
        if dev.type == "cuda": torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        print(f"  バッチ {NB:>9,}: {dt*1e3:7.2f} ms/回 = {NB/dt/1e6:8.1f} M sed積/s"
              f"（フラグ・無NaN 込み）")

    print()
    print("GPU 版: 全域算術（無NaN・フラグ）+ 配線表差し替え + 飽和は最後に1回、が torch で 動く。")


if __name__ == "__main__":
    self_test()

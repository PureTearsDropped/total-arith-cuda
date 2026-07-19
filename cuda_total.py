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
import numpy as np
import torch

GE, LE, SUNK = 1, 2, 4
F32 = torch.float32
MAX = torch.finfo(F32).max            # 飽和天井
MIN = torch.finfo(F32).tiny           # ε = 最小正規数（向き付き 無限小）


def _sat(raw64, dev):
    """float64 の 生値 → (float32 値, フラグ)。溢れ→±MAX+GE / 潰れ→±MIN+LE / NaN 出さない。
       NaN 入力は (0, 境界なし+SUNK) に 全域化（外部AI監査 2026-07-19 の 指摘で 追加）。"""
    nan = torch.isnan(raw64)
    raw64 = torch.where(nan, torch.zeros_like(raw64), raw64)
    sign = torch.sign(raw64)
    a = raw64.abs()
    over = a > MAX                                   # inf も ここで 捕まる
    under = (a > 0) & (a < MIN)
    val = raw64.clone()
    val = torch.where(over, sign * MAX, val)
    val = torch.where(under, sign * MIN, val)
    flag = torch.zeros(raw64.shape, dtype=torch.uint8, device=dev)
    flag |= over.to(torch.uint8) * GE
    flag |= under.to(torch.uint8) * LE
    flag |= nan.to(torch.uint8) * (GE | LE | SUNK)
    return val.to(F32), flag


class Tot:
    """全域数のテンソル: val float32・flag uint8（同形）。

       flag=None（利用者の 入口）では 入力を **全域化してから** 受け入れる:
       NaN→(0, 境界なし+SUNK) / ±Inf・float32範囲外→±MAX+GE / 非正規化数→±MIN+LE。
       「NaN/Inf を 決して 作らない」は この入口で 初めて 不変条件になる
       （外部AI監査 2026-07-19: 旧版は 入口が 素通しで 0×Inf=NaN の 経路が あった）。"""
    __slots__ = ('val', 'flag')
    def __init__(self, val, flag=None):
        if flag is None:
            self.val, self.flag = _sat(val.double(), val.device)
        else:
            self.val = val.to(F32)
            self.flag = flag
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
    """加算の フラグ則（2026-07-19 改訂・外部AI監査の 反例 (+MIN,LE)+(−MIN,=)→(0,LE) を 受けて）:
       同符号（符号既知）なら 単純和が 健全（|和|=|a|+|b| は 単調: ≥+≥=≥, ≤+≤=≤）。
       **相殺が 起こりうる**（異符号・どちらか0・符号不明）とき、入力に 境界が 一つでも あれば
       片側境界は 維持できない → 境界なし+符号不明。旧 clash 則（両GE異符号のみ）は これに 包含される。"""
    raw = a.val.double() + b.val.double()
    val, sflag = _sat(raw, a.device)
    fin = a.flag | b.flag
    anyb = (fin & (GE | LE)) > 0
    sign_known = (fin & SUNK) == 0
    same_sign = (torch.sign(a.val) * torch.sign(b.val)) > 0        # 厳密（0 は 同符号に 含めない）
    cancel = ~(sign_known & same_sign)
    f = torch.where(anyb & cancel, torch.full_like(fin, GE | LE | SUNK), fin)
    return Tot(val, sflag | f)

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
# Cayley–Dickson 構成（自己完結・nd_algebra と 同一規約: (a,b)(c,d) = (ac − d̄b, da + bc̄)）
def _cd_conj(x):
    n = len(x)
    if n == 1: return x.copy()
    h = n // 2
    return np.concatenate([_cd_conj(x[:h]), -x[h:]])

def _cd_prod(x, y):
    n = len(x)
    if n == 1: return x * y
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    return np.concatenate([_cd_prod(a, c) - _cd_prod(_cd_conj(d), b),
                           _cd_prod(d, a) + _cd_prod(b, _cd_conj(c))])

def cd_omega(M):
    """符号表 OMEGA[i,j] ∈ {−1,+1}, 経路 = i⊕j（XOR routing）。"""
    E = np.eye(M)
    OM = np.zeros((M, M), dtype=int)
    for i in range(M):
        for j in range(M):
            v = _cd_prod(E[i], E[j])
            k = int(np.argmax(np.abs(v)))
            assert k == (i ^ j), f"XOR routing 破れ M={M} ({i},{j})"
            OM[i, j] = int(np.sign(v[k]))
    return OM

def wiring_tensor(kind, M, device):
    """配線表 T[k,i,j]。kind: 'cd'（Cayley–Dickson XOR経路）/ 'cyclic'（巡回畳み込み）。"""
    T = torch.zeros(M, M, M, dtype=torch.float32, device=device)
    if kind == "cd":
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
    def ref_mult_M(x, y, OM, M):
        r = [0] * M
        for i in range(M):
            for j in range(M):
                r[i ^ j] += OM[i, j] * x[i] * y[j]
        return r
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
    print("=" * 76)
    print("④ 入口の全域化 + 回帰（外部AI監査 2026-07-19 の 反例を 常設化）")
    print("=" * 76)
    t = Tot(torch.tensor([float('nan'), float('inf'), float('-inf'), 1e300, -1e300],
                         dtype=torch.float64, device=dev))
    ok_entry = (not torch.isnan(t.val).any()) and (not torch.isinf(t.val).any())
    print(f"  Tot([NaN,±Inf,±1e300]) → NaN/Inf 残留 {'なし ✓' if ok_entry else 'あり ×'}"
          f"（flag={t.flag.tolist()}）")
    zero = tot_mul(Tot(torch.zeros(1, device=dev)),
                   Tot(torch.tensor([1e300], dtype=torch.float64, device=dev)))
    zok = not torch.isnan(zero.val).any()
    print(f"  0 × Tot(1e300): val={zero.val.item():g}（旧版は NaN）{'✓' if zok else '×'}")
    ra = Tot(torch.tensor([MIN], device=dev)); ra.flag = torch.tensor([LE], dtype=torch.uint8, device=dev)
    rr = tot_add(ra, Tot(torch.tensor([-MIN], device=dev)))
    reg_ok = int(rr.flag.item()) == (GE | LE | SUNK)
    print(f"  (+MIN,LE)+(−MIN,=): flag={int(rr.flag.item())} = 境界なし+SUNK {'✓' if reg_ok else '×（旧版: LE=嘘）'}")
    assert ok_entry and zok and reg_ok

    print()
    print("=" * 76)
    print("⑤ フラグ代数の オラクル検査（外部AI監査の 盲点指摘に 応答）")
    print("   フラグ付き入力の 許容真値集合から 真値を 乱択し、出力フラグの 主張と 照合")
    print("=" * 76)
    K = 200_000
    rng5 = np.random.default_rng(11)
    def rand_flagged(K):
        mag = torch.tensor(10.0 ** rng5.uniform(-20, 20, K), device=dev)
        sgn = torch.tensor(rng5.choice([-1.0, 1.0], K), device=dev)
        val = (mag * sgn).to(F32)
        fl = torch.tensor(rng5.choice([0, GE, LE, GE | SUNK, LE | SUNK, GE | LE], K).astype(np.uint8),
                          device=dev)
        u = torch.tensor(rng5.uniform(0, 1, K), device=dev)
        ge_, le_ = (fl & GE) > 0, (fl & LE) > 0
        m = torch.ones(K, dtype=torch.float64, device=dev)
        m = torch.where(ge_ & ~le_, 1 + 7 * u, m)               # GE: |真| ∈ |val|·[1,8]
        m = torch.where(le_ & ~ge_, u, m)                        # LE: |真| ∈ |val|·[0,1]
        m = torch.where(ge_ & le_, 8 * u, m)                     # 境界なし: 何でも
        ts = torch.where((fl & SUNK) > 0,
                         torch.tensor(rng5.choice([-1.0, 1.0], K), device=dev).double(),
                         torch.sign(val).double())               # SUNK: 符号も 乱択
        tv = val.double().abs() * m * ts
        tt = Tot(val); tt.flag = fl
        return tt, tv
    A2, ta = rand_flagged(K); B2, tb = rand_flagged(K)
    lies5 = 0
    for name, op in [("mul", tot_mul), ("add", tot_add), ("div", tot_div)]:
        r = op(A2, B2)
        t = {"mul": ta * tb, "add": ta + tb,
             "div": torch.where(tb == 0, torch.zeros_like(ta),
                                ta / torch.where(tb == 0, torch.ones_like(tb), tb))}[name]
        ge = (r.flag & GE) > 0; le = (r.flag & LE) > 0; sk = (r.flag & SUNK) > 0
        vo = r.val.double(); slack = 2.0 ** -20                  # f32 丸め分の 猶予
        lies5 += int((ge & ~le & (t.abs() < vo.abs() * (1 - slack))).sum())
        lies5 += int((le & ~ge & (t.abs() > vo.abs() * (1 + slack))).sum())
        lies5 += int(((~sk) & (vo != 0) & (t != 0)
                      & (torch.sign(t) != torch.sign(vo))).sum())  # 符号は 値が 運ぶ（SUNK 以外）
        lies5 += int(torch.isnan(r.val).sum() + torch.isinf(r.val).sum())
    print(f"  {3*K:,} 件: 嘘 **{lies5}**（片側境界・符号・無NaN の 三契約）")
    assert lies5 == 0

    print()
    print("GPU 版: 全域算術（無NaN・フラグ）+ 配線表差し替え + 飽和は最後に1回、が torch で 動く。")


if __name__ == "__main__":
    self_test()

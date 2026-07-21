#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""tbm — 総ビリニア機械 (Total Bilinear Machine) v0.1 の アセンブラ。TBM_SPEC.md の 実行係。

  薄い層の 規律: 意味論は ここに 住まない。各命令は 監査済み 実装
  (cuda_total / cuda_fused / nested_registry / total-arith-hardware の golden) を
  呼ぶだけ。本モジュールが 持ち込む 新規の 意味論は **coarse (粗誠実)** ただ1つ (SPEC §3)。

  命令 6 種 (SPEC §2): TOTALIZE / BILIN / LINMAP / AXPY / NORM / CHECK。
  バックエンド: cpu (torch cpu) / gpu (torch cuda + 融合カーネル) / hw (run_everywhere.py が
  cocotb+iverilog で 駆動 — サブセット ISA)。適合表は SPEC §4。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from cuda_total import Tot, GE, LE, SUNK, group_mul, tot_add, wiring_tensor, _sat
import nested_registry as NR

HW_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "total-arith-hardware"))


# ================================================================ coarse (SPEC §3 唯一の新規)
def coarse_group_mul(T, a, b):
    """粗誠実の BILIN。値経路は evidence と 同一 (f64 蓄積 → 最後に 1 回 飽和)。
       フラグ: 入力の どこかに 札が あれば 出力 **全成分**に GE|LE|SUNK —
       「境界事象に 触れた・向きと 符号は 追跡していない」の 粗い 上界 (過剰警報あり・嘘なし)。
       + 出力 自身の 飽和札 (こちらは 向きつきで 正確)。
       非融合 (einsum) のまま: 大バッチの クリーン経路は einsum が 既に メモリ最適で、
       融合しても 買うものが ない (実測 1.13×)。"""
    M = T.shape[0]
    shp = a.val.shape
    av = a.val.reshape(-1, M).double()
    bv = b.val.reshape(-1, M).double()
    raw = torch.einsum('kij,bi,bj->bk', T.double(), av, bv)
    val, sflag = _sat(raw, a.device)
    dirty = ((a.flag.reshape(-1, M) | b.flag.reshape(-1, M)).amax(1, keepdim=True) > 0)
    fin = dirty.to(torch.uint8) * (GE | LE | SUNK)
    return Tot(val.reshape(shp), (sflag | fin).reshape(shp))


def bare_group_mul(T, a, b):
    "値のみ (NaN 非生成は 維持・フラグは 運ばない)。両端を evidence が 守る 区間の 内側 専用。"
    M = T.shape[0]
    shp = a.val.shape
    raw = torch.einsum('kij,bi,bj->bk', T.double(),
                       a.val.reshape(-1, M).double(), b.val.reshape(-1, M).double())
    val, sflag = _sat(raw, a.device)
    return Tot(val.reshape(shp), torch.zeros_like(sflag).reshape(shp))


# ================================================================ 5枚目の棚: LAWS (SPEC §6)
def _law_rank_exact(impl_name, alg_name):
    return NR.impl_verify(NR.impl(impl_name), NR.alg(alg_name))

def _law_powerassoc(alg_name, seed=0):
    return NR.powerassoc_defect(NR.alg(alg_name), np.random.default_rng(seed))

def _law_assoc(alg_name, seed=0):
    return NR.assoc_defect(NR.alg(alg_name), np.random.default_rng(seed))

def _law_homomorphy(map_name):
    w, u = NR.map_verify(NR.amap(map_name))
    return max(w, u)

LAWS = {
    "rank_exact":  _law_rank_exact,    # ΣUVW ≡ T (IMPLS の 正しさ)
    "powerassoc":  _law_powerassoc,    # exp∘log の 門番
    "assoc":       _law_assoc,
    "homomorphy":  _law_homomorphy,    # MAPS の 正しさ (因子積 ≡ M 込み)
}


# ================================================================ プログラム (命令列)
class Program:
    "命令列。ビルダー = 6 命令のみ。実行は run()。"
    def __init__(self, name="tbm"):
        self.name = name
        self.ins = []

    def TOTALIZE(self, dst, src):
        self.ins.append(("TOTALIZE", dict(dst=dst, src=src)))
        return self

    def BILIN(self, dst, a, b, alg="sedenion", honesty="evidence"):
        assert honesty in ("evidence", "coarse", "bare")
        self.ins.append(("BILIN", dict(dst=dst, a=a, b=b, alg=alg, honesty=honesty)))
        return self

    def LINMAP(self, dst, src, map_name, honesty="bare"):
        self.ins.append(("LINMAP", dict(dst=dst, src=src, map_name=map_name,
                                        honesty=honesty)))
        return self

    def AXPY(self, dst, src, c=1.0):
        "dst ← dst + c·src。c は 合成時 定数 (テープ係数)。src の 札は 保守的に 通す。"
        self.ins.append(("AXPY", dict(dst=dst, src=src, c=float(c))))
        return self

    def NORM(self, dst, src, block=4, Ein=0):
        "先頭 block 成分を ブロック正規化 (golden = gate_fast.block_normalize_g_fast)。"
        self.ins.append(("NORM", dict(dst=dst, src=src, block=block, Ein=Ein)))
        return self

    def CHECK(self, dst, law, **args):
        self.ins.append(("CHECK", dict(dst=dst, law=law, args=args)))
        return self

    def describe(self):
        return "\n".join(f"  {op:<9} {a}" for op, a in self.ins)


_ALG_T = {}
def _wiring(alg, device):
    key = (alg, str(device))
    if key not in _ALG_T:
        kind, M = {"sedenion": ("cd", 16), "octonion": ("cd", 8),
                   "quaternion": ("cd", 4), "complex": ("cd", 2),
                   "cyclic8": ("cyclic", 8)}[alg]
        _ALG_T[key] = wiring_tensor(kind, M, device)
    return _ALG_T[key]


def _norm_golden(vals, Ein, W=6, Win=24, Emax=20, EW=12):
    "total-arith-hardware の 監査済み golden で ブロック正規化 (整数値 前提)。"
    sys.path.insert(0, HW_REPO)
    from gate_fast import block_normalize_g_fast
    from gate_exponent import bus_const, bus_val
    from gate_bilinear import to_sd, from_sd, new_counter
    og, Eg, fg = block_normalize_g_fast([to_sd(int(v), Win) for v in vals],
                                        bus_const(Ein, EW), W, Emax, new_counter())
    return dict(o=[from_sd(d) for d in og],
                flags=[(int(g), int(l)) for g, l, _ in fg],
                Eout=bus_val([int(b) for b in Eg]) % (1 << EW))


def run(prog, feed, where="cpu"):
    """cpu / gpu バックエンドで 実行。feed: {名前: 配列}。返り値: {名前: Tot | dict | float}。
       gpu が 未対応の 命令 (NORM) は 適合表の 空欄 通り '—' を 返す (偽装しない)。"""
    dev = torch.device(where if where != "gpu" else "cuda")
    env = {}
    for op, p in prog.ins:
        if op == "TOTALIZE":
            x = torch.as_tensor(np.asarray(feed[p["src"]], dtype=np.float64), device=dev)
            if where == "gpu":
                from cuda_fused import fused_totalize
                env[p["dst"]] = fused_totalize(x)              # 税関 1 カーネル (Tot と bit一致)
            else:
                env[p["dst"]] = Tot(x)
        elif op == "BILIN":
            T = _wiring(p["alg"], dev)
            a, b = env[p["a"]], env[p["b"]]
            if p["honesty"] == "evidence":
                if where == "gpu":
                    from cuda_fused import fused_group_mul
                    env[p["dst"]] = fused_group_mul(T, a, b)
                else:
                    env[p["dst"]] = group_mul(T, a, b)
            elif p["honesty"] == "coarse":
                env[p["dst"]] = coarse_group_mul(T, a, b)
            else:
                env[p["dst"]] = bare_group_mul(T, a, b)
        elif op == "LINMAP":
            mp = NR.amap(p["map_name"])
            M = torch.as_tensor(np.real(mp.M), dtype=torch.float64, device=dev)
            x = env[p["src"]]
            raw = x.val.double() @ M.T
            val, sflag = _sat(raw, dev)
            if p["honesty"] == "bare":
                flag = torch.zeros_like(sflag)
            else:                                            # coarse 規則 (BILIN と 同じ)
                dirty = (x.flag.amax(-1, keepdim=True) > 0)
                flag = sflag | dirty.to(torch.uint8) * (GE | LE | SUNK)
            env[p["dst"]] = Tot(val, flag)
        elif op == "AXPY":
            x = env[p["src"]]
            if p["c"] != 1.0:
                x = Tot(x.val.double() * p["c"])
                x = Tot(x.val, x.flag | env[p["src"]].flag)  # 札は 保守的に 通す
            env[p["dst"]] = tot_add(env[p["dst"]], x)
        elif op == "NORM":
            if where == "gpu":
                env[p["dst"]] = "—"                          # 適合表 §4: GPU NORM は 空欄
                continue
            x = env[p["src"]]
            v = x.val.reshape(-1, x.val.shape[-1])
            assert torch.all(v == v.round()), "NORM v1 は 整数値 Tot のみ (SD golden の 定義域)"
            env[p["dst"]] = [_norm_golden([int(t) for t in row[:p["block"]]], p["Ein"])
                             for row in v.cpu().numpy()]
        elif op == "CHECK":
            env[p["dst"]] = float(LAWS[p["law"]](**p["args"]))
    return env


# ================================================================ 標準ライブラリ (SPEC §5)
def macro_exp(prog, dst, x, alg="sedenion", order=8, honesty="coarse"):
    """EXP マクロの 展開形: { BILIN; AXPY(1/k!) } × order。命令 だけで 書けることの 実証。
       (融合 実行係は cuda_fused_pipeline series — 本展開は 仕様どおりの 逐次形)"""
    import math
    prog.TOTALIZE(dst, x + "__unit")                          # acc = e0 (feed 側で 供給)
    prog.TOTALIZE("_term", x + "__unit")
    for k in range(1, order + 1):
        prog.BILIN("_term", "_term", x, alg=alg, honesty=honesty)
        prog.AXPY(dst, "_term", c=1.0 / math.factorial(k))
    return prog


# ================================================================ self-test
def self_test():
    print("tbm — アセンブラ self-test (意味論の 正: cuda_total / nested_registry / HW golden)")
    dev_ok = torch.cuda.is_available()
    rng = np.random.default_rng(0)

    print("① coarse の 契約: 値 ≡ evidence 値 / 汚れ入力 → 全札 / 清潔入力 → 飽和札のみ")
    T = _wiring("sedenion", torch.device("cpu"))
    a = Tot(torch.tensor(rng.integers(-9, 10, (64, 16)), dtype=torch.float64))
    b = Tot(torch.tensor(rng.integers(-9, 10, (64, 16)), dtype=torch.float64))
    ev = group_mul(T, a, b)
    co = coarse_group_mul(T, a, b)
    assert torch.equal(ev.val, co.val), "coarse の 値経路が evidence と 不一致"
    assert int(co.flag.max()) == 0, "清潔な 整数入力で 札が 立った"
    f = torch.zeros(64, 16, dtype=torch.uint8); f[3, 7] = GE
    ad = Tot(a.val, f)
    cod = coarse_group_mul(T, ad, b)
    assert torch.all(cod.flag[3] == (GE | LE | SUNK)), "汚れ行の 全成分に 札が 立っていない"
    assert int(cod.flag[torch.arange(64) != 3].max()) == 0, "札が 他の 行へ 漏れた"
    print("   値 bit一致 ✓ / 汚れ1行 → その行の 全成分 GE|LE|SUNK・他行 0 ✓")

    print("② LAWS 棚: 反証子は 走る (合格も 不合格も 測って 言う)")
    r1 = LAWS["rank_exact"]("sedenion_naive", "sedenion")
    r2 = LAWS["powerassoc"]("octonion")
    r3 = LAWS["assoc"]("octonion")
    r4 = LAWS["homomorphy"]("wh8")
    assert r1 == 0.0 and r2 < 1e-12 and r4 < 1e-12 and r3 > 1e-3
    print(f"   rank_exact(sed)={r1:.1e} powerassoc(oct)={r2:.1e} "
          f"homomorphy(wh8)={r4:.1e} / assoc(oct)={r3:.2f} (破れを 正しく 検出) ✓")

    print("③ プログラム: t=a·b; t+=c を cpu で 実行し 素の 参照と 一致")
    feed = {"in_a": rng.integers(-9, 10, (8, 16)),
            "in_b": rng.integers(-9, 10, (8, 16)),
            "in_c": rng.integers(-9, 10, (8, 16))}
    P = (Program("mac").TOTALIZE("a", "in_a").TOTALIZE("b", "in_b").TOTALIZE("c", "in_c")
         .BILIN("t", "a", "b").AXPY("t", "c"))
    out = run(P, feed, "cpu")
    A16 = NR.alg("sedenion")
    ref = np.stack([NR.rawmul(A16, feed["in_a"][i].astype(float),
                              feed["in_b"][i].astype(float)) + feed["in_c"][i]
                    for i in range(8)])
    assert np.array_equal(out["t"].val.numpy(), ref.astype(np.float32))
    print("   cpu: 値 一致 (整数 厳密) ✓")

    if dev_ok:
        print("④ 同じ プログラムを gpu (融合 evidence) で — 値・フラグ bit一致")
        outg = run(P, feed, "gpu")
        assert np.array_equal(outg["t"].val.cpu().numpy(), out["t"].val.numpy())
        assert np.array_equal(outg["t"].flag.cpu().numpy(), out["t"].flag.numpy())
        print("   cpu ≡ gpu (値 bit一致・フラグ bit一致) ✓")

    print("⑤ EXP マクロ: 6命令 展開 ≡ nested_registry.nexp (四元数)")
    x4 = 0.3 * rng.standard_normal((4, 4))
    feedx = {"in_x": x4, "x__unit": np.tile([1.0, 0, 0, 0], (4, 1))}
    Pe = Program("exp")
    Pe.TOTALIZE("x", "in_x")
    macro_exp(Pe, "acc", "x", alg="quaternion", order=12)
    oute = run(Pe, feedx, "cpu")
    A4 = NR.alg("quaternion")
    refe = np.stack([NR.nexp(A4, NR.nel(A4, x4[i]), order=12).c for i in range(4)])
    d = np.abs(oute["acc"].val.numpy() - refe).max()
    assert d < 1e-6, d
    print(f"   マクロ展開 vs nexp: 最大差 {d:.1e} ✓")
    print("done — 薄い層は 薄いまま (意味論は 全部 呼び先)")


if __name__ == "__main__":
    self_test()

#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""tbm — 総ビリニア機械 (Total Bilinear Machine) v0.1 の アセンブラ。TBM_SPEC.md の 実行係。

  薄い層の 規律: 意味論は ここに 住まない。各命令は 監査済み 実装
  (cuda_total / cuda_fused / nested_registry / total-arith-hardware の golden) を
  呼ぶだけ。本モジュールが 持ち込む 新規の 意味論は **coarse (粗誠実)** ただ1つ (SPEC §3)。

  命令 7 種 (SPEC §2): TOTALIZE / BILIN / LINMAP / AXPY / NORM / CHECK / SELECT。
  バックエンド: cpu (torch cpu) / gpu (torch cuda + 融合カーネル) / hw (run_everywhere.py が
  cocotb+iverilog で 駆動 — サブセット ISA)。適合表は SPEC §4。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from cuda_total import Tot, GE, LE, SUNK, group_mul, tot_add, wiring_tensor, _sat
import nested_registry as NR

INEXACT = 8   # 定義恒等式が 検算に 通らなかった (cuda_total nsolve の 0x08 と 同値・GE|LE|SUNK=7 と 直交)

HW_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "total-arith-hardware"))


# ================================================================ coarse (SPEC §3 唯一の新規)
_WIDTH = {"f64": torch.float64, "f32": torch.float32}

def _raw_bilin(T, a, b, width):
    """蓄積の 芯。width は 誠実さと 独立の **貯め幅ダイヤル** (SPEC §3 三分解):
       f64 = quire 規律 (民生 GPU で 7.4×の 請求書) / f32 = 相対誤差 ~1e-7・7.2× 速い。
       half (f16/bf16) は 実測で 却下 — この einsum 路では 速度 利得 0 のまま 誤差 1e-3 級。"""
    dt = _WIDTH[width]
    M = T.shape[0]
    raw = torch.einsum('kij,bi,bj->bk', T.to(dt),
                       a.val.reshape(-1, M).to(dt), b.val.reshape(-1, M).to(dt))
    return _sat(raw.double(), a.device)


def coarse_group_mul(T, a, b, width="f64"):
    """粗誠実の BILIN。width='f64' なら 値経路は evidence と 同一 (bit一致)。
       フラグ: 入力の どこかに 札が あれば 出力 **全成分**に GE|LE|SUNK —
       「境界事象に 触れた・向きと 符号は 追跡していない」の 粗い 上界 (過剰警報あり・嘘なし)。
       + 出力 自身の 飽和札 (こちらは 向きつきで 正確)。
       非融合 (einsum) のまま: 大バッチの クリーン経路は einsum が 既に メモリ最適で、
       融合しても 買うものが ない (実測 1.13×)。"""
    M = T.shape[0]
    shp = a.val.shape
    val, sflag = _raw_bilin(T, a, b, width)
    dirty = ((a.flag.reshape(-1, M) | b.flag.reshape(-1, M)).amax(1, keepdim=True) > 0)
    fin = dirty.to(torch.uint8) * (GE | LE | SUNK)
    return Tot(val.reshape(shp), (sflag | fin).reshape(shp))


def bare_group_mul(T, a, b, width="f64"):
    "値のみ (NaN 非生成は 維持・フラグは 運ばない)。両端を evidence が 守る 区間の 内側 専用。"
    shp = a.val.shape
    val, sflag = _raw_bilin(T, a, b, width)
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

    def BILIN(self, dst, a, b, alg="sedenion", honesty="evidence", width="f64"):
        "width: 貯め幅ダイヤル ('f64'=quire 規律 / 'f32'=~1e-7 誤差で 7.2× — coarse/bare のみ)"
        assert honesty in ("evidence", "coarse", "bare")
        assert width in _WIDTH and (width == "f64" or honesty != "evidence"), \
            "evidence は f64 固定 (bit一致 契約)"
        self.ins.append(("BILIN", dict(dst=dst, a=a, b=b, alg=alg, honesty=honesty,
                                       width=width)))
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
        "law='residual' は 成分ごと モード: 残差バッファ → 行マスク (SELECT の 述語)。"
        self.ins.append(("CHECK", dict(dst=dst, law=law, args=args)))
        return self

    def SELECT(self, dst, mask, a, b, orflag_false=0):
        """第7命令 (SPEC §2.5): ビット選択 (mask∧a)∨(¬mask∧b) を val と flag の 双方に。
           算術混合 p·x+(1−p)·y と 違い 非選択枝は 値も 札も 一切 漏れない。
           BILIN/AXPY は 札を OR で 合流させるため 既存命令では 表現不能 (昇格の 根拠)。
           乗算 0 本・HW では ただの MUX。orflag_false: 不合格側に 貼る 名札 (INEXACT 等)。"""
        self.ins.append(("SELECT", dict(dst=dst, mask=mask, a=a, b=b,
                                        orflag_false=int(orflag_false))))
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


def run(prog, feed, where="cpu", env0=None):
    """cpu / gpu バックエンドで 実行。feed: {名前: 配列}。返り値: {名前: Tot | dict | float}。
       gpu が 未対応の 命令 (NORM) は 適合表の 空欄 通り '—' を 返す (偽装しない)。
       env0: 構築済み バッファ (Tot / mask) の 持ち込み (テスト・部分実行 用)。"""
    dev = torch.device(where if where != "gpu" else "cuda")
    env = dict(env0) if env0 else {}
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
                env[p["dst"]] = coarse_group_mul(T, a, b, width=p.get("width", "f64"))
            else:
                env[p["dst"]] = bare_group_mul(T, a, b, width=p.get("width", "f64"))
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
            if p["law"] == "residual":
                # 成分ごと モード: 行の 全成分 |r| ≤ tol かつ 残差経路が 清潔 ⟺ 合格。
                # 残差計算 自身が 飽和した 行は 「検算 不能」であり 合格に できない (嘘なし)。
                r = env[p["args"]["src"]]
                tol = float(p["args"].get("tol", 1e-6))
                env[p["dst"]] = ((r.val.double().abs().amax(-1) <= tol)
                                 & (r.flag.amax(-1) == 0))
            else:
                env[p["dst"]] = float(LAWS[p["law"]](**p["args"]))
        elif op == "SELECT":
            m = env[p["mask"]]
            mk = m.reshape(-1, 1) if m.dim() == 1 else m
            a, b = env[p["a"]], env[p["b"]]
            val = torch.where(mk, a.val, b.val)
            flag = torch.where(mk, a.flag, b.flag | np.uint8(p["orflag_false"]))
            env[p["dst"]] = Tot(val, flag)
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


def _certify(prog, dst, resid, tol):
    "候補パターンの 共通尾部: CHECK(residual) → SELECT (合格=候補 素通し / 不合格=INEXACT 名札)。"
    prog.CHECK("_m", law="residual", src=resid, tol=tol)
    prog.SELECT(dst, "_m", dst, dst, orflag_false=INEXACT)
    return prog


def macro_sqrt(prog, dst, x, cand, alg="quaternion", honesty="evidence", tol=1e-6):
    """SQRT マクロ (SPEC §5): 候補は 信じない・検算だけ 信じる。
       cand は feed 供給の **無審査 oracle** (どこから 来ても よい)。機械が するのは
       BILIN で 自乗 → x を 引いた 残差 → CHECK → SELECT のみ。
       合格行: cand の 値と 札が そのまま 通る (検算経路の 札は SELECT が 捨てる)。
       不合格行: 値は 通す (全域性 — 例外を 投げない) が INEXACT を 行名指しで 立てる。"""
    prog.TOTALIZE(dst, cand)
    prog.BILIN("_r", dst, dst, alg=alg, honesty=honesty)
    prog.AXPY("_r", x, c=-1.0)
    return _certify(prog, dst, "_r", tol)


def macro_inv(prog, dst, x, cand, unit, alg="quaternion", honesty="evidence", tol=1e-6):
    """INV マクロ: 検算は cand·x − e₀。x=0 行は oracle が 0 を 返せば (Moore-Penrose:
       a/0=0 は 定理) 恒等式 0·0=e₀ が 成り立たないので INEXACT が 正しく 立つ —
       値 0 のまま 通り、名札が 「逆元では ない」ことを 言う。"""
    prog.TOTALIZE(dst, cand)
    prog.BILIN("_r", dst, x, alg=alg, honesty=honesty)
    prog.AXPY("_r", unit, c=-1.0)
    return _certify(prog, dst, "_r", tol)


def macro_log(prog, dst, x, cand, alg="quaternion", honesty="coarse",
              order=12, tol=1e-4):
    """LOG マクロ: 検算は exp(cand) − x — **log の 門番は exp** (LAWS powerassoc と
       同じ 思想が プログラムに なった 形)。exp は macro_exp の 級数 展開なので
       定義域は 級数の 収束域 (‖cand‖ 小)。feed に f'{dst}__unit' (=e₀) が 要る。"""
    prog.TOTALIZE(dst, cand)
    macro_exp(prog, "_ey", dst, alg=alg, order=order, honesty=honesty)
    prog.AXPY("_ey", x, c=-1.0)
    return _certify(prog, dst, "_ey", tol)


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

    print("①b 貯め幅ダイヤル: coarse/bare は f32 蓄積を 選べる (evidence は f64 固定)")
    ar = Tot(torch.randn(4096, 16, dtype=torch.float64))
    br = Tot(torch.randn(4096, 16, dtype=torch.float64))
    e64 = group_mul(T, ar, br)
    c32 = coarse_group_mul(T, ar, br, width="f32")
    rel = float((c32.val - e64.val).abs().max() / e64.val.abs().max())
    assert 0 < rel < 1e-5, rel
    try:
        Program("x").BILIN("s", "a", "b", honesty="evidence", width="f32")
        raise RuntimeError("evidence×f32 が 通ってしまった")
    except AssertionError:
        pass
    print(f"   coarse(f32) vs f64: 相対差 {rel:.1e} (実測どおり ~1e-7) ✓ / evidence×f32 拒否 ✓")

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

    print("⑥ SELECT の 契約: 非選択枝は 値も 札も 漏れない (AXPY の OR 合流とは 別物)")
    dirty = Tot(torch.full((4, 4), 7.0), torch.full((4, 4), GE, dtype=torch.uint8))
    clean = Tot(torch.ones(4, 4))
    mask = torch.tensor([True, True, False, False])
    Ps = Program("sel"); Ps.ins.append(("SELECT", dict(dst="s", mask="m", a="c", b="d",
                                                       orflag_false=INEXACT)))
    out_s = run(Ps, {}, "cpu", env0={"m": mask, "c": clean, "d": dirty})
    s = out_s["s"]
    assert torch.all(s.val[:2] == 1.0) and int(s.flag[:2].max()) == 0, "選択枝が 汚れた"
    assert torch.all(s.val[2:] == 7.0) and torch.all(s.flag[2:] == (GE | INEXACT))
    leak = tot_add(clean, dirty)                             # 対照: 算術合流は 必ず 漏れる
    assert int(leak.flag.max()) > 0
    print("   mask=True 行: 値1・札0 (GE を 捨てた) ✓ / False 行: GE|INEXACT ✓ / "
          "tot_add 対照は 札が 漏れる ✓")

    print("⑦ SQRT マクロ: 候補は 信じない・検算だけ 信じる (complex)")
    y = rng.standard_normal((6, 2))
    A2 = NR.alg("complex")
    x2 = np.stack([NR.rawmul(A2, y[i], y[i]) for i in range(6)])
    y_bad = y.copy(); y_bad[2] += 0.5                        # 行2 の 候補を 汚す
    Pq = Program("sqrt"); Pq.TOTALIZE("x", "in_x")
    macro_sqrt(Pq, "s", "x", "in_cand", alg="complex")
    outq = run(Pq, {"in_x": x2, "in_cand": y_bad}, "cpu")
    fl = outq["s"].flag.amax(-1)
    assert int(fl[2]) == INEXACT and int(fl[torch.arange(6) != 2].max()) == 0
    assert np.allclose(outq["s"].val.numpy(), y_bad.astype(np.float32))
    print("   汚した 行2 だけ INEXACT・他 5 行 清潔 ✓ / 値は 全行 通貨 (全域性) ✓")

    print("⑧ INV マクロ: x=0 行は Moore-Penrose 候補 0 → INEXACT が 正しく 立つ (quaternion)")
    x4i = rng.standard_normal((5, 4)); x4i[3] = 0.0
    n2 = (x4i**2).sum(-1, keepdims=True); n2[3] = 1.0
    cand = x4i * np.array([1.0, -1, -1, -1]) / n2            # conj/|x|² (行3 は 0)
    Pi = Program("inv"); Pi.TOTALIZE("x", "in_x"); Pi.TOTALIZE("one", "in_e0")
    macro_inv(Pi, "v", "x", "in_cand", "one", alg="quaternion")
    outi = run(Pi, {"in_x": x4i, "in_cand": cand,
                    "in_e0": np.tile([1.0, 0, 0, 0], (5, 1))}, "cpu")
    fli = outi["v"].flag.amax(-1)
    assert int(fli[3]) == INEXACT and int(fli[torch.arange(5) != 3].max()) == 0
    assert np.all(outi["v"].val.numpy()[3] == 0.0)
    print("   正則 4 行 合格・零因子行 のみ INEXACT (値 0 のまま 通貨) ✓")

    print("⑨ LOG マクロ: 検算は exp — log の 門番が プログラムに なった (quaternion)")
    u = 0.3 * rng.standard_normal((4, 4))
    xl = np.stack([NR.nexp(A4, NR.nel(A4, u[i]), order=16).c for i in range(4)])
    u_bad = u.copy(); u_bad[1] += 0.3
    Pl = Program("log"); Pl.TOTALIZE("x", "in_x")
    macro_log(Pl, "L", "x", "in_cand", alg="quaternion")
    outl = run(Pl, {"in_x": xl, "in_cand": u_bad,
                    "L__unit": np.tile([1.0, 0, 0, 0], (4, 1))}, "cpu")
    fll = outl["L"].flag.amax(-1)
    assert int(fll[1]) == INEXACT and int(fll[torch.arange(4) != 1].max()) == 0
    print("   真の log 3 行 合格・汚した 行1 のみ INEXACT ✓")

    if dev_ok:
        print("⑩ SQRT マクロを gpu で — cpu と 値・フラグ bit一致")
        outg2 = run(Pq, {"in_x": x2, "in_cand": y_bad}, "gpu")
        assert np.array_equal(outg2["s"].val.cpu().numpy(), outq["s"].val.numpy())
        assert np.array_equal(outg2["s"].flag.cpu().numpy(), outq["s"].flag.numpy())
        print("   cpu ≡ gpu ✓")
    print("done — 薄い層は 薄いまま (意味論は 全部 呼び先)")


if __name__ == "__main__":
    self_test()

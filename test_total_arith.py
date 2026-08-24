#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証
"""test_total_arith — 二つの半身が **同じ規約を 守っている** ことの 恒久検査。

  既存の self_test 群（各モジュールの `__main__`）は 中身の 正しさを 見る。ここが 見るのは
  **境界** — numpy 側と torch 側の 間で 転置・重複実装・旗の 語彙が ずれていないか。
  ずれても 単体テストは 全部 通ってしまうので、ここが 無いと 誰も 気づけない。

    ① Cayley–Dickson の 表が 二通りの 作り方で 一致（XOR 経路 ⇔ 直接の 積 — Julia 双子と 同じ道）
    ② 構造テンソルの 添字順（numpy T[i,j,k] ⇔ torch T[k,i,j]）の 変換が 恒等
    ③ 旗の 二語彙の 橋（順序層 ⇔ 検算層）が 情報を 黙って 落とさない
    ④ 任意の Alg が そのまま GPU 配線として 走り、numpy の 積と 一致する
    ⑤ 三値正規形の 門番が 非三値の 表を ちゃんと 拒む

  実行: python test_total_arith.py          （軽い ①〜⑤ だけ・GPU 不要）
        TOTAL_ARITH_SLOW=1 python test_total_arith.py   （各モジュールの self_test も 続けて）
  pytest が あれば pytest test_total_arith.py でも 同じ（test_* 関数の 集まり）。

  **「変えたが 結果は 変わっていない」を 言いたいときは これでは 足りない** — ここは 規約の
  検査であって 回帰の 検査ではない。二つの 版を 全数値で ビット比較するのは tools/ab_check.py。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import total_core as tc
from total_core import GE, LE, SUNK, UNKNOWN, NO_BOUND, SING, CPLX, OVER, INEXACT
from cuda_total import wiring_tensor, group_mul, Tot
import nested_registry as nr

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CD_DIMS = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------- ① 表の 二通りの 作り方
def test_cd_table_two_constructions():
    """XOR 経路 (cd_omega) と 基底積の 総当り (cd_prod) が 同じ T を 与える。

       python 側は 前者・Julia 双子 (NestedSeries.jl の `_from_mul(_cdprod)`) は 後者で
       表を 作っている。ここが 割れると **双子が 別の 代数を 計算する** — 単体では 気づけない。"""
    for M in CD_DIMS:
        OM = tc.cd_omega(M)
        T_xor = np.zeros((M, M, M))
        for i in range(M):
            for j in range(M):
                T_xor[i, j, i ^ j] = OM[i, j]
        E = np.eye(M)
        T_dir = np.zeros((M, M, M))
        for i in range(M):
            for j in range(M):
                T_dir[i, j] = tc.cd_prod(E[i], E[j])
        assert np.array_equal(T_xor, T_dir), f"CD 表が 二通りで 割れた: M={M}"
        assert np.array_equal(T_xor, nr.cd_alg(M).T), f"nested_registry の 表と 不一致: M={M}"
        assert tc.is_ternary(T_xor), f"CD 表が 三値でない: M={M}"


def test_cd_conventions():
    "共役・単位元・交代性の 基本規約（規約が 変わったら ここが 落ちる）"
    for M in CD_DIMS:
        E = np.eye(M)
        assert np.array_equal(tc.cd_prod(E[0], E[0]), E[0]), f"1·1≠1: M={M}"
        for i in range(1, M):                       # 虚部の 二乗は −1（CD の 定義）
            assert np.allclose(tc.cd_prod(E[i], E[i]), -E[0]), f"e{i}²≠−1: M={M}"
        x = np.arange(1.0, M + 1)
        assert np.allclose(tc.cd_conj(x), np.concatenate([x[:1], -x[1:]])), \
            f"共役は 実部以外の 反転: M={M}"


# ---------------------------------------------------------------- ② 添字順
def test_index_order_roundtrip():
    rng = np.random.default_rng(0)
    T = rng.standard_normal((5, 5, 5))
    assert np.array_equal(tc.to_ijk(tc.to_kij(T)), T)
    assert np.array_equal(tc.to_kij(tc.to_ijk(T)), T)


def test_index_order_across_halves():
    """numpy 側 `Alg.T`[i,j,k] と torch 側 `wiring_tensor`[k,i,j] が 同じ表の 転置であること。

       **この検査が この改修の 主目的**: 同じ名前 T の 添字順が 半身ごとに 違うので、
       片方の 規約で もう片方を 触ると 静かに 間違う（einsum は 形が 合えば 通る）。"""
    for M in CD_DIMS:
        got = wiring_tensor("cd", M, DEV).cpu().numpy()
        assert np.array_equal(got, tc.to_kij(nr.cd_alg(M).T)), f"cd{M} の 添字順が ずれた"
    for M in (2, 3, 5, 8):
        got = wiring_tensor("cyclic", M, DEV).cpu().numpy()
        assert np.array_equal(got, tc.to_kij(nr.cyclic_alg(M).T)), f"cyc{M} の 添字順が ずれた"


# ---------------------------------------------------------------- ③ 旗の 二語彙の 橋
def test_flag_bitmap_is_documented():
    "ビット地図そのもの（total_core の docstring の 表と 一致しているか）"
    assert (GE, LE, SUNK) == (0x01, 0x02, 0x04)
    assert (SING, CPLX, OVER, INEXACT) == (0x01, 0x02, 0x04, 0x08)
    assert UNKNOWN == GE | LE | SUNK and NO_BOUND == GE | LE
    # 二語彙は 下位2ビットで **衝突している**（意味が 違う）— 混ぜてはならない、の 根拠
    assert GE == SING and LE == CPLX
    # 順序層は bit3 を 使わない ⟹ cuda_fused_solve の INEXACT 相乗りは 合法
    assert (tc.ORDER_BITS & INEXACT) == 0
    assert tc.SING_F == UNKNOWN and tc.INEXACT_F == INEXACT


def test_flag_bridge_no_silent_loss():
    "橋は 訳せない ビットを 黙って 落とさない（residue で 返す）"
    # 意味が 一致する 唯一の点: 完全無知 ⇔ 一意解なし
    assert tc.to_verify(UNKNOWN) & SING
    o, res = tc.to_order(SING)
    assert o == UNKNOWN and res == 0
    # 飽和は 向きを 失う（GE も LE も OVER・戻すと 両側境界）
    assert tc.to_verify(GE) == OVER and tc.to_verify(LE) == OVER
    o, res = tc.to_order(OVER)
    assert o == NO_BOUND and res == 0
    # 検算層 固有の ビットは 像を 持たない ⟹ residue に 出る
    for b in (INEXACT, CPLX):
        o, res = tc.to_order(b)
        assert o == 0 and res == b, f"{b:#x} が 黙って 消えた"
    o, res = tc.to_order(SING | INEXACT)
    assert o == UNKNOWN and res == INEXACT
    # 厳密は 厳密のまま（偽陽性を 作らない）
    assert tc.to_verify(0) == 0 and tc.to_order(0) == (0, 0)


def test_flag_bridge_vectorized():
    "スカラーと 配列で 同じ答え・語彙の 外の ビットを 作らない"
    fs = np.arange(8, dtype=np.uint8)
    v = tc.to_verify(fs)
    assert v.shape == fs.shape and int((v & ~np.uint8(tc.VERIFY_BITS)).max()) == 0
    assert [int(x) for x in v] == [tc.to_verify(int(f)) for f in fs]
    o, res = tc.to_order(np.arange(16, dtype=np.uint8))
    assert int((o & ~np.uint8(tc.ORDER_BITS)).max()) == 0
    assert [(int(a), int(b)) for a, b in zip(o, res)] == \
           [tc.to_order(f) for f in range(16)]


def test_flag_bridge_is_conservative():
    """橋が 主張を **強めない**（弱める向きにしか 倒さない）ことを 全 8 通りで。

       順序旗 f を 検算層に 訳して 戻したとき、元より 弱い（ビットが 増えるか 同じ）こと。
       強くなったら それは 検算層に 無い 保証を でっち上げたということ。"""
    for f in range(8):
        back, res = tc.to_order(tc.to_verify(f))
        assert res == 0
        assert (back | f) == back or back == f, \
            f"旗 {f:#x} が 往復で 強くなった: {back:#x}"


# ---------------------------------------------------------------- ④ 任意の Alg を GPU 配線に
def _agree(A, name, ntrial=8, seed=0, ternary=True):
    "GPU の group_mul（T 差し替え）と numpy の rawmul が 一致するか"
    T = wiring_tensor(A, device=DEV, ternary=ternary)
    rng = np.random.default_rng(seed)
    d = A.dim
    for _ in range(ntrial):
        x = rng.standard_normal(d) * 0.5
        y = rng.standard_normal(d) * 0.5
        ref = nr.rawmul(A, x, y)
        a = Tot(torch.as_tensor(x, dtype=torch.float32, device=DEV))
        b = Tot(torch.as_tensor(y, dtype=torch.float32, device=DEV))
        got = group_mul(T, a, b).val.cpu().numpy().astype(np.float64)
        assert np.allclose(got, ref, rtol=2e-6, atol=1e-6), \
            f"{name}: GPU 配線 ≠ numpy 積\n got={got}\n ref={ref}"


def test_arbitrary_alg_runs_on_gpu():
    """**この改修で 新しく できるように なったこと**: Clifford / Grassmann / 行列代数 /
       テンソル積 が そのまま GPU カーネルに 載る（以前は 'cd' と 'cyclic' の 2 種だけ）。"""
    cases = [(nr.clifford_alg(2), "Cl2"), (nr.clifford_alg(3), "Cl3"),
             (nr.grassmann_alg(1), "Λ1(dual)"), (nr.grassmann_alg(2), "Λ2"),
             (nr.matn_alg(2), "mat2"), (nr.cd_alg(16), "sedenion"),
             (nr.tensor(nr.grassmann_alg(1), nr.cd_alg(4)), "dualquat")]
    for A, name in cases:
        _agree(A, name)


def test_preset_name_lookup():
    "文字列 プリセット名でも 引ける（nested_registry.ALGS 経由）"
    T = wiring_tensor("cl3", device=DEV)
    assert tuple(T.shape) == (8, 8, 8)
    assert np.array_equal(T.cpu().numpy(), tc.to_kij(nr.clifford_alg(3).T))
    for bad in ("nope", "cd"):                  # 'cd' は M 無しでは 引けない
        try:
            wiring_tensor(bad, device=DEV)
        except (ValueError, AssertionError):
            continue
        raise AssertionError(f"{bad!r} が 通ってしまった")


def test_dim_mismatch_is_caught():
    try:
        wiring_tensor(nr.clifford_alg(3), M=4, device=DEV)
    except AssertionError:
        return
    raise AssertionError("次元 不一致が 素通りした")


# ---------------------------------------------------------------- ⑤ 三値正規形の 門番
def test_ternary_gate():
    """係数 ½ を 持つ 表（jordan など）は 「配線＝計算」が 成り立たない ⟹ 既定で 拒む。
       拒むだけでなく **明示すれば 通る**（ternary=False）— 禁止でなく 申告。"""
    J = nr.jordan(nr.matn_alg(2))
    assert not tc.is_ternary(J.T), "前提が 崩れた: jordan(mat2) は 非三値のはず"
    try:
        wiring_tensor(J, device=DEV)
    except AssertionError as e:
        assert "三値" in str(e)
    else:
        raise AssertionError("非三値の 表が 門番を 素通りした")
    T = wiring_tensor(J, device=DEV, ternary=False)          # 申告すれば 通る
    assert np.array_equal(T.cpu().numpy(), tc.to_kij(J.T))
    _agree(J, "jordan(mat2)", ternary=False)     # 非三値でも 値は 正しい（門は 正直さの ため）


def test_cd_omega_rejects_non_power_of_two():
    for M in (3, 5, 6, 12):
        try:
            tc.cd_omega(M)
        except AssertionError:
            continue
        raise AssertionError(f"M={M} の CD が 通ってしまった")


# ---------------------------------------------------------------- 重い: 既存 self_test 群
SLOW_MODULES = ["cuda_total", "nested_registry", "hyper_transcend",
                "nested_series", "total_pipeline"]


def test_module_self_tests():
    "既存の self_test を まとめて 呼ぶ（TOTAL_ARITH_SLOW=1 のときだけ）"
    if os.environ.get("TOTAL_ARITH_SLOW") != "1":
        print("  (skip: TOTAL_ARITH_SLOW=1 で 各モジュールの self_test も 走る)")
        return
    import importlib
    for m in SLOW_MODULES:
        mod = importlib.import_module(m)
        print(f"\n===== {m}.self_test() =====")
        mod.self_test()


# ---------------------------------------------------------------- runner
def main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    print(f"device = {DEV}\n")
    bad = 0
    for n, f in fns:
        try:
            f()
            print(f"  ✓ {n}")
        except Exception as e:
            bad += 1
            print(f"  ✗ {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} 通過")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

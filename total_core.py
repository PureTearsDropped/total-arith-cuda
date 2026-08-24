#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証
"""total_core — 二つの半身が **共有する 規約** だけを 置く 土台（numpy のみ・torch 不要）。

このリポジトリは 二つの 半身で できている:

  · 配線・算術層 (`cuda_total.Tot`, torch/GPU)   — 数 = (値, 旗)。旗は **順序境界** の 語彙。
  · 超越・検算層 (`nested_registry.Nel`, numpy)  — 数 = (係数, 旗)。旗は **検算** の 語彙。

両者は 別の ことを 言うために 別の 語彙を 持つ（統合しない）。ただし **規約は 一つ** でなければ
ならない。この モジュールは その規約 — 旗のビット割当・層間の橋・Cayley–Dickson の 符号表・
構造テンソルの 添字順 — を 一箇所に 集める。

--------------------------------------------------------------------------- 旗の ビット地図

  bit | 順序層 (Tot)                        | 検算層 (Nel / Hyper)
  ----+-------------------------------------+--------------------------------------------
  0x01| GE   真値は この値 **以上**          | SING    一意の 逆/解が ない（零因子・NaN 入口）
  0x02| LE   真値は この値 **以下**          | CPLX    実数の 外に 出た（hyper_transcend のみ）
  0x04| SUNK 符号 不明                       | OVER    ±MAX に 飽和（範囲溢れ）
  0x08| （未使用 — 拡張ビット）              | INEXACT 定義恒等式が **検証できなかった**

  **同じビットが 層ごとに 別の意味**（0x01 = GE と SING、0x02 = LE と CPLX）。したがって
  二つの語彙を 一つのワードに OR してはならない。例外は bit3 (0x08) で、順序層が 使わないため
  `cuda_fused_solve` は ここに INEXACT を 相乗りさせている（合法な 拡張・下位3ビットは 順序層）。

--------------------------------------------------------------------------- 層間の 橋

  対応は **全単射でない**（両方向で 情報が 落ちる）。落ちる場所を 明示するのが この橋の 仕事:

    順序 → 検算 : GE|LE|SUNK（境界なし+符号不明 = NaN 由来）→ SING   ← 意味が 一致する 唯一の点
                  GE 単独 / LE 単独（±MAX 飽和 / ±MIN 潰れ）  → OVER  ← **向きが 落ちる**
                  SUNK 単独（符号だけ 不明）                  → SING  ← 保守的に 強く 倒す
    検算 → 順序 : SING    → GE|LE|SUNK
                  OVER    → GE|LE           （どちらの 端で 飽和したか 検算層は 覚えていない）
                  INEXACT → **像なし**       ← 値の 境界の 主張では ないから 順序層に 訳せない
                  CPLX    → **像なし**       ← 同上（実部だけ 見ると 嘘に なる）

  像のない ビットは `to_order` が 黙って 捨てず、第2返り値 `residue` で 返す。呼び出し側が
  「捨ててよい」と 決めるまで 情報は 消えない（このリポジトリの 教義: 嘘ゼロ）。

--------------------------------------------------------------------------- 構造テンソルの 添字順

  同じ 双線形写像を、二つの半身が **転置した 順** で 持っている。どちらも その側では 自然:

    T_ijk[i,j,k]  … numpy 側 (`Alg.T`)      「e_i·e_j が e_k に 落ちる 係数」と 読める
    T_kij[k,i,j]  … torch 側 (`wiring_tensor`) 出力成分 k の 行 `T[k]` が 一枚で 取れる
                                              （group_mul の パターン則が 成分ごとに 回るため）

  統一しない代わりに、**名前で 区別し**（`T_ijk` / `T_kij`）、変換は この一箇所だけに 置く。
  `test_total_arith.py` が 両側の 表の 一致を 恒久検査する。
"""
import numpy as np

# --------------------------------------------------------------- 順序層の 語彙 (Tot)
GE, LE, SUNK = 0x01, 0x02, 0x04
ORDER_BITS = GE | LE | SUNK
NO_BOUND = GE | LE                      # 「上からも 下からも 抑えられない」
UNKNOWN = GE | LE | SUNK                # NaN 由来の 完全無知（_sat(NaN) が 作る形）

# --------------------------------------------------------------- 検算層の 語彙 (Nel / Hyper)
SING, CPLX, OVER, INEXACT = 0x01, 0x02, 0x04, 0x08
VERIFY_BITS = SING | CPLX | OVER | INEXACT
VERIFY_ONLY = CPLX | INEXACT            # 順序層に 像を 持たない ビット

# 順序層の 語彙で 書いた 「一意の 厳密解なし」。cuda_fused_solve が この名前で 使う。
SING_F = UNKNOWN
INEXACT_F = INEXACT                     # 順序層が 空けている bit3 への 合法な 相乗り


def to_verify(f_order):
    """順序層の 旗 → 検算層の 旗（保守的・嘘を 増やさない 向きにだけ 倒す）。int/ndarray 両対応。"""
    f = np.asarray(f_order)
    out = np.zeros(f.shape, dtype=np.uint8)
    unknown = (f & UNKNOWN) == UNKNOWN
    out |= np.where(unknown, SING, 0).astype(np.uint8)
    # GE/LE は 飽和イベント → OVER（向きは 落ちる）
    out |= np.where((f & (GE | LE)) > 0, OVER, 0).astype(np.uint8)
    # SUNK 単独（境界の 主張は 生きているが 符号が 不明）→ SING に 倒す
    out |= np.where(((f & SUNK) > 0) & ~unknown, SING, 0).astype(np.uint8)
    return int(out) if out.ndim == 0 else out


def to_order(f_verify):
    """検算層の 旗 → (順序層の 旗, residue)。residue = 訳せずに 残った ビット (CPLX/INEXACT)。

       residue を 捨てるかどうかは **呼び出し側の 判断**。黙って 落とさない。"""
    f = np.asarray(f_verify)
    out = np.zeros(f.shape, dtype=np.uint8)
    out |= np.where((f & SING) > 0, UNKNOWN, 0).astype(np.uint8)
    out |= np.where((f & OVER) > 0, NO_BOUND, 0).astype(np.uint8)
    res = (f & VERIFY_ONLY).astype(np.uint8)
    if out.ndim == 0:
        return int(out), int(res)
    return out, res


def order_name(f):
    "順序層の 旗を 人間語に（デバッグ用）"
    f = int(f)
    if f == 0: return "="
    if f & UNKNOWN == UNKNOWN: return "無知(境界なし+符号不明)"
    parts = [n for b, n in ((GE, "≥"), (LE, "≤"), (SUNK, "符号不明")) if f & b]
    return "+".join(parts)


def verify_name(f):
    "検算層の 旗を 人間語に（デバッグ用）"
    f = int(f)
    if f == 0: return "検証済"
    return "+".join(n for b, n in ((SING, "零因子/解なし"), (CPLX, "ℂ"),
                                   (OVER, "飽和"), (INEXACT, "未検証")) if f & b)


# --------------------------------------------------------------- Cayley–Dickson（唯一の実装）
def cd_conj(x):
    "Cayley–Dickson 共役: 実部を 残し 虚部を 反転（再帰形 — 分解の 定義通り）"
    n = len(x)
    if n == 1:
        return x.copy()
    h = n // 2
    return np.concatenate([cd_conj(x[:h]), -x[h:]])


def cd_prod(x, y):
    "Cayley–Dickson 積 (a,b)(c,d) = (ac − d̄b, da + bc̄) — 全リポ共通の 規約"
    n = len(x)
    if n == 1:
        return x * y
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    return np.concatenate([cd_prod(a, c) - cd_prod(cd_conj(d), b),
                           cd_prod(d, a) + cd_prod(b, cd_conj(c))])


_OMEGA_CACHE = {}


def cd_omega(M):
    """符号表 OMEGA[i,j] ∈ {−1,+1}、経路 = i⊕j（XOR routing）。M は 2 の冪。

       XOR 経路が 破れないことを その場で 検査する — 「配線＝計算」が 成り立つ 根拠。"""
    if M in _OMEGA_CACHE:
        return _OMEGA_CACHE[M]
    assert M >= 1 and (M & (M - 1)) == 0, f"Cayley–Dickson は 2 の冪のみ: M={M}"
    E = np.eye(M)
    OM = np.zeros((M, M), dtype=int)
    for i in range(M):
        for j in range(M):
            v = cd_prod(E[i], E[j])
            k = int(np.argmax(np.abs(v)))
            assert k == (i ^ j), f"XOR routing 破れ M={M} ({i},{j})"
            OM[i, j] = int(np.sign(v[k]))
    OM.setflags(write=False)
    _OMEGA_CACHE[M] = OM
    return OM


# --------------------------------------------------------------- 構造テンソルの 添字順
def to_kij(T_ijk):
    "numpy 規約 T[i,j,k] → torch 規約 T[k,i,j]"
    return np.ascontiguousarray(np.transpose(np.asarray(T_ijk, float), (2, 0, 1)))


def to_ijk(T_kij):
    "torch 規約 T[k,i,j] → numpy 規約 T[i,j,k]"
    return np.ascontiguousarray(np.transpose(np.asarray(T_kij, float), (1, 2, 0)))


def is_ternary(T):
    "配線正規形 (TBM_SPEC §1.5): 係数が {−1,0,+1} だけ ⟹ 配線段に 丸めが 無い＝厳密"
    return set(np.unique(np.asarray(T, float)).tolist()) <= {-1.0, 0.0, 1.0}


def alg_kij(A):
    """`Alg`（`.T` が T[i,j,k] の 何か）→ torch 規約の numpy 配列 T[k,i,j]。

       nested_registry を import しない（循環を 避ける）— 属性だけで 受ける。"""
    T = getattr(A, "T", None)
    assert T is not None, f"Alg らしきもの (.T を 持つ) が 要る: {type(A)}"
    T = np.asarray(T, float)
    assert T.ndim == 3 and T.shape[0] == T.shape[1] == T.shape[2], \
        f"構造テンソルは 立方体 (d,d,d): {T.shape}"
    return to_kij(T)

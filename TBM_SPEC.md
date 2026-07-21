# TBM_SPEC — 総ビリニア機械 v0.1

⚠️ AI-assisted; verify. / 生成AI使用・要検証

> **Total Bilinear Machine**: a one-multiply-instruction computer.
> 掛け算する命令は **BILIN** ただ1つ。残りは足し算・検査・引っ越しの脇役。
> 同じプログラムが CPU / GPU / FPGA の3つのシリコンで走り、**どの高さでも嘘をつかない**。
> 複素・四元数・セデニオン・行列積・畳み込み・Strassen・exp・solve・FFT・物理法則発見は
> すべてこの機械のプログラム(マクロ)であり、機械の部品ではない。

本書は**仕様**である。実行は各バックエンドが担い、意味論の正は既存コードに置く
(仕様と実行は兼ねない)。数値はすべて実測 (2026-07-21, RTX 5090 / Arty A7 は iverilog+cocotb)。

---

## 0. 実体

- **プログラム** = 命令列。命令は6種のみ (§2)。
- **オペランド** = 名前付きバッファ `(val: float32[B,d], flag: uint8[B,d])`。
  値とフラグは常に対で流れる。フラグのビット割当 (GE/LE/SUNK) の正は `cuda_total.py`。
  - GE = 「|x| ≥ MAX」— 上へはみ出た(向きつき)
  - LE = 「|x| ≤ MIN」— 下へ潰れた(向きつき)
  - SUNK = 符号情報喪失
  - フラグ 0 かつ val=0 は **真の零** (予約語)。飽和値は点でなく境界を意味する。
- **契約**: 機械は NaN / Inf を決して生成しない。TOTALIZE (§2.1) を通らない値は機械に入れない。

## 1. 一文の定義

TBM の唯一の乗算命令は、構造テンソル T[i,j,k] のランク分解 (U,V,W) による縮約

```
c = Wᵀ((U·a) ⊙ (V·b))        (⊙ は成分ごとの積・R = 実乗算の本数)
```

を全域算術 (val,flag) の上で行うことである。T を差し替えれば代数が替わり (N)、
(U,V,W) を差し替えればアルゴリズムが替わり (IMPLS)、テープを替えれば関数が替わる (O)。
配線 = 計算。

## 2. 命令セット (6命令)

全命令に共通フィールド **honesty ∈ {evidence, coarse, bare}** (§3)。

| # | 命令 | 日本語 | 形式 | 意味論の正 |
|---|------|--------|------|-----------|
| 1 | **TOTALIZE** | 税関 | `TOTALIZE dst, src_raw` | `cuda_total.Tot` / `ScalarTot.TotNum` |
| 2 | **BILIN** | 唯一の掛け算 | `BILIN[h] dst, a, b, impl=<IMPLS名>` | `nested_registry.rawmul` + `cuda_total.group_mul` |
| 3 | **LINMAP** | 座標替え | `LINMAP[h] dst, src, map=<MAPS名>[, factor=k]` | `nested_registry.MAPS` (apply_fast≡密行列を検証済) |
| 4 | **AXPY** | 足し算 | `AXPY[h] dst, src, c` (dst←dst+c·src) | `cuda_total.tot_add` |
| 5 | **NORM** | 桁揃え | `NORM dst, src` | `gate_bfp.blocknorm` (total-arith-hardware) |
| 6 | **CHECK** | 検算 | `CHECK law=<LAWS名>, args...` → INEXACT | probe 関数群 (assoc_defect 等・LAWS棚 §6) |

**退化定理** (機械が小さいことの証明): LINMAP と AXPY は BILIN の退化形である —
片腕を定数 e₀ に固定した `BILIN(x, e₀; U=M, V=1, W=I)` は任意の線形写像になる。
ゆえに理論上の命令数は「掛け算1・税関1・桁揃え1・検算1」。実装が LINMAP/AXPY を
別に持つのは加減算だけで済む道を乗算器に通さないためであり、意味論は増えていない。

**NORM の規律**: 機械の中で丸めが起きる場所は NORM ただ1箇所。丸めは方向づき
(嘘なし丸め — 実値がどちら側かをフラグが保存する)。BILIN/AXPY は内部 f64 蓄積とし、
表現替えを伴う丸めを行わない。

**CHECK の規律**: 「候補を出す → 検算して合格だけ通す」(nsolve, hlog が既に実施) を
命令に昇格したもの。残差が閾を超えたら結果に INEXACT を立てる — 例外を投げない。
法則は仮定せず測る。

## 3. 誠実さのダイヤル (honesty)

税金は一枚岩でないことが実測で判明した (セデニオン積, honesty_tax 測定):

| honesty | 何を運ぶか | 実測税金 (対 裸einsum) | 使う場所 |
|---------|-----------|------------------------|----------|
| **evidence** | 犯人名指し (どの成分が・どちら向きに・確実ゼロ/危険ゼロの模様則) | B=1k: **0.9×(タダ)** / B=1M: 37× | 境界 (入口/出口)・制御ループ帯 (B小) |
| **coarse** | 飽和クランプ + フラグOR + GE検知のみ | B=1M: **1.13× ≒ タダ** (非融合上界) | 内部の大バッチ計算 |
| **bare** | 値のみ (NaN非生成は維持) | 1.4× (融合) / 1.0× | 両端を evidence が守る検証済み区間の内側 |

**傾斜誠実性の法則**: 境界=evidence・内部=coarse は妥協ではなく最適である。
根拠は2つの独立な実測 — (1) evidence の税金は B が大きいほど重い (上表)、
(2) 深い連鎖では evidence フラグは「境界なし」に退化して情報が薄まる (bfp_series)。
つまり内部の証拠級は「高くて実りが薄い」。

**税金の三分解** (tbm_bench ①b, B=1M セデニオン, RTX 5090): 総額 36.9× の内訳は
フラグ **1.18×** / f64 蓄積 (quire 規律) **7.4×** / 証拠級の模様則 **4.2×**。
最大の請求書 (f64 quire) は誠実さではなく**精度の規律**であり、独立のダイヤルである
(民生 GPU は fp64=1/64 レート — データセンター級では この項がほぼ消える)。
上の coarse 1.13× は f32 蓄積時の値。**フラグそのものは常にほぼ無料**。

**貯め幅ダイヤル `width`** (BILIN の第2フィールド・honesty と直交): `f64` = quire 規律 /
`f32` = 相対誤差 ~1e-7 (単発 1.2e-7・級数14段 9.4e-7・100段連鎖 2e-8/段) で **7.2× 速い**
(598M/s vs 83M/s)。coarse/bare のみ — evidence は f64 固定 (bit一致契約)。
**half (f16/bf16) は実測で却下**: この einsum 路では速度利得 0 のまま誤差 1e-3 級
(テンソルコア経由に書き直したときに再測)。幅を狭めても嘘は増えない —
範囲事故は飽和+GE で名指しされ (f16 の MAX=65504 でも NaN 0)、増えるのは丸めだけ。

## 4. バックエンド適合表

**HW はサブセット ISA を実装する** (誇張しない)。空欄 = 未対応。

| 命令 × honesty | CPU (nested_registry / ScalarTot.jl) | GPU (Triton) | HW (rtl/ 自動生成SV) |
|---|---|---|---|
| TOTALIZE | ✅ `Tot()` / `TotNum` | ✅ 融合 `fused_totalize` (Tot と bit一致・未融合比 7–11×) | ✅ 入口全域化 |
| BILIN bare | ✅ rawmul / `tbm.bare_group_mul` | ✅ `cuda_fused_pipeline` | — |
| BILIN coarse | ✅ `tbm.coarse_group_mul` | ✅ 同左 (非融合 einsum = 大バッチ最適・1.13×) | — |
| BILIN evidence | ✅ group_mul | ✅ `cuda_fused` (フラグbit一致契約) | ✅ `sd_mult10` (状態7ゲート/成分) |
| LINMAP | ✅ MAPS.apply_fast | ✅ (torch matmul = bare) | — |
| AXPY | ✅ tot_add | ✅ | ✅ `sd_add2` |
| NORM | ✅ gate_bfp (pyシミュ) | — | ✅ `blocknorm` |
| CHECK | ✅ probe 群 | (CPUで実行 — 検算は高さを選ばない) | — |

適合水準: **L0** = bare で値一致 / **L1** = + coarse フラグ一致 / **L2** = + evidence bit一致。
現状: CPU=L2, GPU=L2, HW=L2 (サブセット: TOTALIZE/BILIN/AXPY/NORM)。
アセンブラは `tbm.py` (LAWS 棚込み)・適合試験は `run_everywhere.py` (§7) — **2026-07-21 合格**:
同一プログラムが 3 バックエンドで 値 bit一致・フラグ bit一致 (敵対的 Inf/NaN 注入・飽和フラグ込み)。

## 5. 標準ライブラリ (マクロ)

命令には**しない**。6命令の並びで書けること自体が価値の証明。既存実装は各マクロの実行係。

| マクロ | 展開 (骨格) | 融合実行係 |
|--------|------------|-----------|
| **EXP / SIN / COS** | `TOTALIZE → (scale) → { BILIN; AXPY(c_k) }×order → { BILIN(自乗) }×sq` テープ差し替え=関数差し替え | `cuda_fused_pipeline` series (GPU) / `gate_series` (HW仕様=Fraction一致) |
| **LOG / SQRT / INV** | 候補生成 → **CHECK** (定義恒等式) → 不合格は INEXACT | hyper_transcend (nlog/nsqrt/ninv) |
| **SOLVE** | Ben-Israel 反復 `X ← X(2I−LX)` = { BILIN; AXPY }ループ → **CHECK** (残差) → 零因子は SING | `cuda_fused_solve` (67.4M/s) / `gate_solve` (HW) |
| **CONV** (高速畳み込み) | `LINMAP(F の因子列) → BILIN(対角) → LINMAP(F⁻¹ の因子列)` — FFT は「LINMAP を log n 回」というプログラム | 融合WH畳み込み 7.4G/s・n≤16 融合DFT が cuFFT 経路の 9× |
| **DISCOVER** (法則発見) | ライブラリ行列を BILIN/LINMAP で構築 → evidence フラグで汚染行を名指し除外 → 零空間 SVD → **CHECK** (ギャップ・恒真式) | implicit_discovery / Discovery.jl |

## 6. 5枚目の棚 — LAWS (反証子レジストリ)

CHECK が参照する `LAWS = {名前: (T, 作用) → 残差}`。既存 probe のラップ:
`powerassoc` (exp∘log の門番) / `hurwitz` (合成性 dim1,2,4,8) / `artin_bch` /
`homomorphy` (MAPS 検証) / `rank_exact` (ΣUVW≡T) / `residual` (SOLVE)。
法則は第一級の住人であり、仮定ではなく実行可能な反証子である。

## 7. 適合試験 (旗艦デモの契約) — 実測済 ✅

`run_everywhere.py`: 同一のプログラム (HW サブセット内の BILIN/AXPY/NORM 列・数十バイト) を
3バックエンドでアセンブル実行し、

1. 値: CPU↔GPU↔HW で diff = 0 (整数系入力では文字通り 0、実数系は f32 1ulp 以内を明記)
2. フラグ: bit 一致 (evidence)
3. 敵対的入力 (0除算・オーバーフロー・危険ゼロ) を含めて 1–2 を満たす

を assert する。**"compile once, run on three silicons, never lie."**

## 8. 検証規約

- 本書のすべての主張は対応する self_test / 実測に還元されること (測って主張)。
- 意味論の正は常に実装ファイル側 (本書は指し示すだけ・二重管理しない)。
- 適合表の空欄は空欄のまま公表する。埋めたければコードで埋める。

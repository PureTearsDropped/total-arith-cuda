#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証
"""A/B compare — ab_dump.py の 二つの 出力を **ビット単位で** 突き合わせる (2/2)。

  浮動小数は 値でなく **ビットパターン** で 比べる (最下位 1 ビットの 差も 見逃さない・
  NaN の 位置も 一致を 要求する)。旗・整数は 値で 比べる。

  Usage: ab_cmp.py <a.npz> <b.npz>   終了コード 0 = 完全一致 / 1 = 差分あり
"""
import sys
import numpy as np

if len(sys.argv) != 3:
    sys.exit("Usage: ab_cmp.py <a.npz> <b.npz>")
a = np.load(sys.argv[1], allow_pickle=True)
b = np.load(sys.argv[2], allow_pickle=True)
ka, kb = set(a.files), set(b.files)
bad = []
if ka - kb:
    bad.append(f"A のみ: {sorted(ka - kb)[:5]}")
if kb - ka:
    bad.append(f"B のみ: {sorted(kb - ka)[:5]}")

nfloat = nint = 0
for k in sorted(ka & kb):
    x, y = a[k], b[k]
    if x.shape != y.shape:
        bad.append(f"{k}: 形 {x.shape} vs {y.shape}")
        continue
    if x.dtype.kind == "f":
        # bit-for-bit: NaN も同じ位置なら一致とみなす（値としては NaN != NaN）
        xb = x.view(np.uint64 if x.dtype == np.float64 else np.uint32)
        yb = y.view(np.uint64 if y.dtype == np.float64 else np.uint32)
        nfloat += x.size
        if not np.array_equal(xb, yb):
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            with np.errstate(invalid="ignore"):
                m = np.nanmax(d) if d.size else 0.0
            bad.append(f"{k}: ビット不一致 {int((xb != yb).sum())}/{x.size} 件・最大差 {m:.3e}")
    else:
        nint += x.size
        if not np.array_equal(x, y):
            bad.append(f"{k}: 不一致 {int((x != y).sum())}/{x.size} 件")

print(f"配列 {len(ka & kb)} 本 / 浮動小数 {nfloat:,} 要素 / 整数・旗 {nint:,} 要素")
if bad:
    print(f"\n差分 {len(bad)} 件:")
    for line in bad[:60]:
        print("  ✗", line)
    sys.exit(1)
print("\n**完全一致**（浮動小数はビットパターン単位・旗は値単位）")

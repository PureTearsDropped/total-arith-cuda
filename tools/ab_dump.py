#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証
"""A/B dump — 「作り直しても 結果は 1 ビットも 変わっていない」を 言えるようにする 道具 (1/2)。

  リファクタは 誰かに 検算させないと 主張に ならない。self_test の 出力を 目で 追うのでは
  不十分で、(a) 速度行が 実行ごとに 動く (b) 未シードな 乱数が 混ざる (c) そもそも
  self_test が 表示していない 数が 大半、の 三つで すぐ 破綻する。

  そこで: 引数で 渡された **リポジトリの 版** を import して、固定入力の バッテリを 走らせ、
  出てきた 数を 全部 .npz に 落とす。二つの 版で 落として ab_cmp.py で ビット比較する。

  決定性は 作りで 保証: 乱数は numpy default_rng(固定シード) だけ。torch の RNG も 時刻も
  使わない。したがって 差が 出たら それは **コードの 差** であって 実行の 揺れではない。

  Usage: ab_dump.py <repo_dir> <out.npz> <cpu|cuda>
"""
import sys, os
if len(sys.argv) != 4:
    sys.exit(__doc__.strip().splitlines()[-1])
repo, out, devname = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.abspath(repo))
import numpy as np
import torch

import cuda_total as ct
import nested_registry as nr
import hyper_transcend as ht

dev = torch.device(devname)
R = {}


def put(key, x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    R[key] = x


def put_tot(key, t):
    put(key + ".val", t.val.double())          # exact widening, keeps every f32 bit
    put(key + ".flag", t.flag)


# ---------------------------------------------------------------- 1. CD sign tables
for M in (1, 2, 4, 8, 16):
    put(f"omega.{M}", np.asarray(ct.cd_omega(M), dtype=np.int64))

# ---------------------------------------------------------------- 2. wiring tensors
for M in (1, 2, 4, 8, 16):
    put(f"T.cd.{M}", ct.wiring_tensor("cd", M, dev))
for M in (2, 3, 5, 8):
    put(f"T.cyc.{M}", ct.wiring_tensor("cyclic", M, dev))

# ---------------------------------------------------------------- 3. scalar total ops
# every flag (0..7) x representative values, all pairs
vals = np.array([0.0, 1.0, -1.0, 3.0, -0.5, 1e30, -1e30, 5e-40, np.finfo(np.float32).max,
                 np.finfo(np.float32).tiny, 1e-45, 2.5], dtype=np.float32)
flags = np.arange(8, dtype=np.uint8)
V, F = np.meshgrid(vals, flags, indexing="ij")
V, F = V.ravel(), F.ravel()
n = V.size
ai = np.repeat(np.arange(n), n)
bi = np.tile(np.arange(n), n)
A = ct.Tot(torch.as_tensor(V[ai], device=dev), torch.as_tensor(F[ai], device=dev))
B = ct.Tot(torch.as_tensor(V[bi], device=dev), torch.as_tensor(F[bi], device=dev))
put_tot("tot_mul", ct.tot_mul(A, B))
put_tot("tot_add", ct.tot_add(A, B))
put_tot("tot_div", ct.tot_div(A, B))
# entry totalization (Tot(x) 1-arg form, incl. NaN/Inf/subnormal/declared domain)
raw = np.array([0.0, 1.0, -1.0, np.nan, np.inf, -np.inf, 1e39, -1e39, 1e-46, 3.25],
               dtype=np.float64)
put_tot("entry", ct.Tot(torch.as_tensor(raw, device=dev)))
put_tot("entry.dom", ct.Tot(torch.as_tensor(raw, device=dev), max=16.0, min=1e-3))

# ---------------------------------------------------------------- 4. group_mul
rng = np.random.default_rng(20260824)
for kind, M in (("cd", 2), ("cd", 4), ("cd", 8), ("cd", 16), ("cyclic", 8), ("cyclic", 3)):
    T = ct.wiring_tensor(kind, M, dev)
    KB = 4000
    for tag, mk in (
        ("dense", lambda: rng.standard_normal((KB, M))),
        ("pos",   lambda: np.abs(rng.standard_normal((KB, M)))),
        ("sparse", lambda: rng.standard_normal((KB, M)) *
                   (rng.random((KB, M)) < 2.0 / M)),
        ("wide",  lambda: rng.standard_normal((KB, M)) * 10.0 ** rng.integers(-40, 39, (KB, M))),
    ):
        av = mk().astype(np.float32); bv = mk().astype(np.float32)
        af = rng.integers(0, 8, (KB, M)).astype(np.uint8)
        bf = rng.integers(0, 8, (KB, M)).astype(np.uint8)
        a = ct.Tot(torch.as_tensor(av, device=dev), torch.as_tensor(af, device=dev))
        b = ct.Tot(torch.as_tensor(bv, device=dev), torch.as_tensor(bf, device=dev))
        put_tot(f"gmul.{kind}{M}.{tag}", ct.group_mul(T, a, b))
        # flag-free fast path too
        a0 = ct.Tot(torch.as_tensor(av, device=dev), torch.zeros_like(a.flag))
        b0 = ct.Tot(torch.as_tensor(bv, device=dev), torch.zeros_like(b.flag))
        put_tot(f"gmul0.{kind}{M}.{tag}", ct.group_mul(T, a0, b0))

# ---------------------------------------------------------------- 5. ekernel (both twins)
poison = np.array([0.0, 0.5, 1.0, 2.0, -1.0, 9.0, 1e-8, 1e8, np.nan, np.inf, -np.inf,
                   1e300, -1e300, 0.25, 100.0, 1e-300], dtype=np.float64)
for name in sorted(nr.OPS):
    v, f = ct.ekernel_gpu(name, poison, device=dev)
    put(f"ek_gpu.{name}.val", v)
    put(f"ek_gpu.{name}.flag", f)
    e = nr.ekernel(name, poison)
    put(f"ek_np.{name}.val", np.asarray(e[0] if isinstance(e, tuple) else e.c, dtype=np.float64))
    put(f"ek_np.{name}.flag", np.asarray(e[1] if isinstance(e, tuple) else e.flag))

# ---------------------------------------------------------------- 6. numpy shelf
rng2 = np.random.default_rng(7)
for aname in sorted(nr.ALGS):
    A_ = nr.alg(aname)
    d = A_.dim
    xs = rng2.standard_normal((6, d)) * 0.3
    ys = rng2.standard_normal((6, d)) * 0.3
    put(f"rawmul.{aname}", np.array([nr.rawmul(A_, x, y) for x, y in zip(xs, ys)]))
    put(f"Lmat.{aname}", nr.Lmat(A_, xs[0]))
    put(f"Rmat.{aname}", nr.Rmat(A_, xs[0]))
    put(f"algT.{aname}", A_.T)
    put(f"unit.{aname}", A_.unit)
    x = nr.Nel(xs[0])
    for op in ("exp", "sqrt", "log", "inv"):
        for br in ("left", "right", "symmetric"):
            try:
                y = nr.nop(A_, op, x, bracket=br)
                put(f"nop.{aname}.{op}.{br}.c", y.c)
                put(f"nop.{aname}.{op}.{br}.f", np.array([y.flag]))
            except Exception as e:
                put(f"nop.{aname}.{op}.{br}.err", np.array([str(type(e).__name__)], dtype=object))
    for side in ("left", "right", "symmetric"):
        try:
            s = nr.nsolve(A_, nr.Nel(xs[1]), nr.Nel(ys[1]), side=side)
            put(f"nsolve.{aname}.{side}.c", s.c)
            put(f"nsolve.{aname}.{side}.f", np.array([s.flag]))
        except Exception as e:
            put(f"nsolve.{aname}.{side}.err", np.array([str(type(e).__name__)], dtype=object))

for iname in sorted(nr.IMPLS):
    im = nr.impl(iname)
    put(f"implT.{iname}", nr.impl_tensor(im))

# ---------------------------------------------------------------- 7. hyper_transcend
for M in (2, 4, 8, 16):
    xs = np.random.default_rng(99).standard_normal((4, M)) * 0.3
    for i, x in enumerate(xs):
        z = ht.Hyper(x)
        for fn in ("hexp", "hlog", "hinv", "hsqrt"):
            if not hasattr(ht, fn):
                continue
            y = getattr(ht, fn)(z)
            put(f"ht.{M}.{i}.{fn}.c", y.c)
            put(f"ht.{M}.{i}.{fn}.f", np.array([y.flag]))

np.savez(out, **{k: v for k, v in R.items()})
print(f"{len(R)} arrays -> {out}")

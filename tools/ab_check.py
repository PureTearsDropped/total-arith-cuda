#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証
"""ab_check — 「この改修で 結果は 1 ビットも 変わっていない」を 機械に 言わせる 一発の 玄関。

  ab_dump.py で 二つの 版の 数を 全部 落とし、ab_cmp.py で ビット比較する、までを 自動で。

    python tools/ab_check.py                     # 作業ツリー vs main (GPU があれば GPU)
    python tools/ab_check.py --rev HEAD~1        # 作業ツリー vs 一つ前の コミット
    python tools/ab_check.py --ref /path/to/old  # 作業ツリー vs 任意の ディレクトリ
    python tools/ab_check.py --device cpu        # 装置を 指定 (既定: あれば cuda)
    python tools/ab_check.py --negative-control  # **検査に 歯が あるか** を 検査する

  --negative-control が 要るのは、「一致」は 検査が 緩いだけでも 出る 答えだから。
  作業ツリーの コピーに 1 行だけ 細工を 入れた 変異体を 作り、比較器が それを **見つける**
  ことを 確かめる。見つけられなければ 検査の 方が 壊れている。
"""
import argparse, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

# 変異体: (名前, 対象ファイル, 置換前, 置換後, 何を 壊すか)
MUTANTS = [
    ("wiring", "cuda_total.py",
     "Tn[(i + j) % M, i, j] = 1.0", "Tn[(i + j + 1) % M, i, j] = 1.0",
     "巡回代数の 配線を 1 つ ずらす (値が 変わる)"),
    ("flag-rule", "cuda_total.py",
     "f = torch.where((fin > 0) & cancel & ~ident,", "f = torch.where((fin > 0) & cancel,",
     "tot_add の 「真の零は 加法単位元」則を 落とす (旗だけ 変わる)"),
]


def run(cmd, **kw):
    return subprocess.run(cmd, check=False, **kw)


def dump(repo, out, device):
    r = run([PY, os.path.join(HERE, "ab_dump.py"), repo, out, device],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        sys.exit(f"ab_dump が 失敗: {repo}")
    return out


def compare(a, b, quiet=False):
    r = run([PY, os.path.join(HERE, "ab_cmp.py"), a, b],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not quiet:
        print(r.stdout.rstrip())
    return r.returncode == 0, r.stdout


def negative_control(device, tmp):
    "変異体を 作って、比較器が それを 見つけることを 確かめる"
    base = dump(ROOT, os.path.join(tmp, "base.npz"), device)
    ok = True
    for name, fname, old, new, what in MUTANTS:
        mdir = os.path.join(tmp, f"mut_{name}")
        shutil.copytree(ROOT, mdir, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".venv", "*.npz"))
        p = os.path.join(mdir, fname)
        s = open(p, encoding="utf-8").read()
        if old not in s:
            print(f"  ? {name}: 細工の 目印が 見つからない ({fname}) — 変異体の 定義が 古い")
            ok = False
            continue
        open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
        same, text = compare(base, dump(mdir, os.path.join(tmp, f"{name}.npz"), device),
                             quiet=True)
        n = text.count("✗")
        if same:
            print(f"  ✗ {name}: 見つけられなかった — **比較器が 壊れている** ({what})")
            ok = False
        else:
            print(f"  ✓ {name}: 差分 {n} 件を 検出 ({what})")
    return ok


def main():
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rev", help="比較先の git リビジョン (既定: main)")
    g.add_argument("--ref", help="比較先の ディレクトリ (git を 使わない)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default=None)
    ap.add_argument("--negative-control", action="store_true",
                    help="比較器に 歯が あるかを 検査する")
    ap.add_argument("--keep", action="store_true", help="中間の .npz を 消さない")
    a = ap.parse_args()

    if a.device is None:
        import torch
        a.device = "cuda" if torch.cuda.is_available() else "cpu"

    tmp = tempfile.mkdtemp(prefix="ab_check_")
    wt = None
    try:
        if a.negative_control:
            print(f"陰性対照 (device={a.device}) — 1 行だけ 細工した 版を 比較器に かける\n")
            ok = negative_control(a.device, tmp)
            print("\n比較器は 変異を 検出できる ✓" if ok else "\n**比較器が 変異を 見逃した**")
            return 0 if ok else 1

        if a.ref:
            ref, label = os.path.abspath(a.ref), a.ref
        else:
            rev = a.rev or "main"
            wt = os.path.join(tmp, "ref")
            r = run(["git", "-C", ROOT, "worktree", "add", "--detach", wt, rev],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if r.returncode != 0:
                print(r.stdout, file=sys.stderr)
                sys.exit(f"git worktree add に 失敗: {rev}")
            ref, label = wt, rev

        print(f"A = {label}\nB = 作業ツリー\ndevice = {a.device}\n")
        same, _ = compare(dump(ref, os.path.join(tmp, "a.npz"), a.device),
                          dump(ROOT, os.path.join(tmp, "b.npz"), a.device))
        if a.keep:
            print(f"\n中間ファイル: {tmp}")
        return 0 if same else 1
    finally:
        if wt:
            run(["git", "-C", ROOT, "worktree", "remove", "--force", wt],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

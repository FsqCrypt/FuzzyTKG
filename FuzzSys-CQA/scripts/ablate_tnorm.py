#!/usr/bin/env python3
"""t-范数消融:同一批交集查询,路径分数一次算好,换组合算子族重排名。

每条路径先做逐路秩归一(ops.norm,值域 [0,1] —— product/Łukasiewicz 等
t-范数只在有界值域上有意义),再按组合算子聚合:

    min     ── 当前基线(朴素 min 合取)
    prod    ── 乘积 t-范数(FLEX 变体用的)
    luk     ── Łukasiewicz:max(0, Σa − (n−1))
    yager2  ── Yager p=2:max(0, 1 − √(Σ(1−a)²))
    amean   ── 广义平均 p=1(参考:根本不是合取,只是取平均)
    max     ── 上界参考(等价于并集)

单一趟内路径只算一次,每种组合只多一次逐查询排名,所以全量跑
比"每种算子单独跑一遍评测"便宜一个量级。

这只是诊断脚本:结果用来决定 M3 的交集算子该用什么,以及给
"可学习 t-范数参数"的动机段提供数字。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, List

import torch
from torch import Tensor

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

import tkg_data as D  # noqa: E402
from eval_tkg import checkpoint_path, split_answers  # noqa: E402
from tcomplex import TComplExScorer  # noqa: E402
from tkg_ops import NaiveOperators  # noqa: E402

TNORMS: dict[str, Callable[[List[Tensor]], Tensor]] = {
    "min": lambda xs: torch.stack(xs).amin(0),
    "prod": lambda xs: torch.stack(xs).prod(0),
    "luk": lambda xs: torch.clamp(torch.stack(xs).sum(0) - (len(xs) - 1), min=0.0),
    "yager2": lambda xs: torch.clamp(
        1.0 - torch.sqrt(torch.stack([(1.0 - x) ** 2 for x in xs]).sum(0)), min=0.0),
    "amean": lambda xs: torch.stack(xs).mean(0),
    "max": lambda xs: torch.stack(xs).amax(0),
}


def reciprocals(S: Tensor, qs: D.QuerySet, idxs: List[int]) -> List[float]:
    """每个 hard 答案的 1/(rank+1),协议与 eval_tkg 一致。"""
    out: List[float] = []
    for j, qi in enumerate(idxs):
        hard, easy_mask = split_answers(qs.answers[qi], qs.easy[qi])
        if not hard:
            continue
        s = S[j].clone()
        if easy_mask:
            s[torch.tensor(sorted(easy_mask), dtype=torch.long)] = float("-inf")
        order = s.argsort(descending=True)
        ht = torch.tensor(sorted(hard), dtype=torch.long)
        pos = (order.view(1, -1) == ht.view(-1, 1)).nonzero()[:, 1]
        out.extend((1.0 / (pos + 1)).tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="icews14")
    ap.add_argument("--types", default="e2i,e3i,t2i,t3i")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="每题型取样数(冒烟用)")
    args = ap.parse_args()

    sc = TComplExScorer(checkpoint_path(args.dataset))
    ops = NaiveOperators(sc, topk=10, calibrate="rank")  # 逐路秩归一,见 docstring

    for name in args.types.split(","):
        qs = D.load_queries(args.dataset, "test", [name])[name]
        q = qs.query
        idxs = list(range(len(qs)))[: args.limit]
        n_path = 2 if name in ("e2i", "t2i") else 3
        atom = ops.Pe if name.startswith("e") else ops.Pt  # e*i 实体域,t*i 时间域

        acc: dict[str, List[float]] = defaultdict(list)
        single: List[List[float]] = [[] for _ in range(n_path)]
        with torch.no_grad():
            for lo in range(0, len(idxs), args.chunk):
                hi = min(lo + args.chunk, len(idxs))
                chunk = idxs[lo:hi]
                qb = q[chunk]
                paths = [
                    ops.norm(atom(qb[:, 3 * p], qb[:, 3 * p + 1], qb[:, 3 * p + 2]))
                    for p in range(n_path)
                ]
                for p in range(n_path):
                    single[p].extend(reciprocals(paths[p], qs, chunk))
                for tn, f in TNORMS.items():
                    acc[tn].extend(reciprocals(f(paths), qs, chunk))

        print(f"\n{name}  (n={len(idxs)}, {n_path} 路, 逐路秩归一后组合)")
        for p in range(n_path):
            print(f"  单路{p+1:<4} {'—':<8} {sum(single[p]) / len(single[p]):.4f}")
        for tn in TNORMS:
            v = sum(acc[tn]) / len(acc[tn])
            print(f"  {tn:<10} {v:.4f}")


if __name__ == "__main__":
    main()

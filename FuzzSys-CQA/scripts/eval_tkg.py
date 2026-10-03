#!/usr/bin/env python3
"""TKG 评测入口。

两种模式:

  --mode lp-check   用原始四元组做标准 filtered LP 评测(链路核验硬锚点)
  --mode query      按题型评测 CQA(默认)

评测协议逐条镜像 TFLEX(TFLEX/main_TCQE_TFLEX.py:1227-1264 与
ComplexTemporalQueryDataloader.py:129-140):

  1. easy 掩码:`if len(easy) >= len(full): easy = {}`,然后 `hard = full − easy`
  2. 分数矩阵里把 easy 置 -inf
  3. argsort 降序,取每个 hard 答案的 0-based 位置
  4. MRR 累计 1/(rank+1);hits@k 为 rank < k

注意第 1 步的比较对象是 **full**(TFLEX 的 TestDataset 变量名叫 hard_answer,
但存的是 full 集合),不是 hard。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set

import torch
from torch import Tensor

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

import tkg_data as D  # noqa: E402
from tcomplex import TComplExScorer  # noqa: E402

# 数据集 -> checkpoint 目录名(目录用连字符,数据集键用下划线,故需映射)
CKPT_DIR = {
    "icews14": "tcomplex-icews14",
    "icews05_15": "tcomplex-icews05-15",
}

# 7 组宏平均,逐字抄自 TFLEX/ComplexTemporalQueryData.py:1095-1110
GROUPS: Dict[str, List[str]] = {
    "avg_e": ["Pe", "Pe2", "Pe3", "e2i", "e3i"],
    "avg_t": ["Pt", "aPt", "bPt", "Pe_Pt", "Pt_sPe_Pt", "Pt_oPe_Pt", "t2i", "t3i"],
    "avg_eCe": ["e2i_N", "e3i_N", "Pe_e2i_Pe_NPe", "e2i_PeN", "e2i_NPe"],
    "avg_tCt": ["t2i_N", "t3i_N", "Pe_t2i_PtPe_NPt", "t2i_PtN", "t2i_NPt"],
    "avg_Ue": ["e2u", "Pe_e2u"],
    "avg_Ut": ["t2u", "Pe_t2u"],
    "avg_x": ["t2i_Pe", "Pe_t2i", "e2i_Pe", "Pe_e2i", "between", "Pe_aPt", "Pe_bPt",
              "Pt_sPe", "Pt_oPe", "Pt_se2i", "Pt_oe2i", "Pe_at2i", "Pe_bt2i"],
}


def checkpoint_path(dataset: str) -> Path:
    if dataset not in CKPT_DIR:
        raise KeyError(f"{dataset} 没有 TComplEx 权重(现有: {sorted(CKPT_DIR)})")
    d = D.ROOT / "data/tkg/checkpoints" / CKPT_DIR[dataset]
    hits = sorted(d.glob("*_tflex_ids.pt"))
    if not hits:
        raise FileNotFoundError(f"{d} 下没有 *_tflex_ids.pt")
    return hits[0]


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

class Metric:
    """累加 hard 答案的排名,输出 MRR / Hits@k。

    注意 MRR 等指标是对 **hard 答案**取平均(TFLEX 口径),不是对查询取平均;
    一个查询有几个 hard 答案就贡献几个样本。
    """

    def __init__(self) -> None:
        self.ranks: List[int] = []
        self.n_query = 0

    def add_ranks(self, ranks: Sequence[int]) -> None:
        self.ranks.extend(int(r) for r in ranks)
        if ranks:
            self.n_query += 1

    @property
    def n_answer(self) -> int:
        return len(self.ranks)

    def summary(self) -> Dict[str, float]:
        if not self.ranks:
            return {"MRR": float("nan"), "H@1": float("nan"), "H@3": float("nan"),
                    "H@10": float("nan"), "n_answer": 0, "n_query": self.n_query}
        t = torch.tensor(self.ranks, dtype=torch.float64)
        return {
            "MRR": float((1.0 / (t + 1)).mean()),
            "H@1": float((t < 1).float().mean()),
            "H@3": float((t < 3).float().mean()),
            "H@10": float((t < 10).float().mean()),
            "n_answer": len(self.ranks),
            "n_query": self.n_query,
        }


def hard_answers(full: Set[int], easy: Set[int]) -> Set[int]:
    """镜像 TFLEX 的 easy/hard 切分(比较对象是 full)。"""
    if len(easy) >= len(full):
        easy = set()
    return full - easy


# ---------------------------------------------------------------------------
# lp-check:原始四元组的 filtered LP
# ---------------------------------------------------------------------------

def run_lp_check(dataset: str, split: str = "test", batch_size: int = 256,
                 limit: int | None = None) -> Dict[str, Dict[str, float]]:
    scorer = TComplExScorer(checkpoint_path(dataset))
    print(scorer.describe())

    vocab = D.load_vocab(dataset)
    store = D.load_triples(dataset, vocab)
    tails, heads = store.filtered()

    triples = getattr(store, split)
    if limit:
        triples = triples[:limit]
    print(f"{split} 三元组 {len(triples)}")

    m_tail, m_head = Metric(), Metric()
    n_batch = (len(triples) + batch_size - 1) // batch_size

    with torch.no_grad():
        for bi in range(n_batch):
            chunk = triples[bi * batch_size:(bi + 1) * batch_size]
            s = torch.tensor([x[0] for x in chunk], dtype=torch.long)
            r = torch.tensor([x[1] for x in chunk], dtype=torch.long)
            o = torch.tensor([x[2] for x in chunk], dtype=torch.long)
            t = torch.tensor([x[3] for x in chunk], dtype=torch.long)

            # 尾实体预测:mask 掉 (s,r,τ) 上全部已知真尾,目标自身还原
            st = scorer.score_entities(s, r, t)
            # 头实体预测:mask 掉 (r,o,τ) 上全部已知真头
            sh = scorer.score_entities_as_subject(r, o, t)

            for i, (ss, rr, oo, tt) in enumerate(chunk):
                # 先存目标分数,再把已知真答案 mask 掉,最后还原目标自身
                # (目标本身也在 known 里,不还原就会被自己的过滤打掉)
                orig = st[i, oo].item()
                known = tails.get((ss, rr, tt), set())
                if known:
                    st[i, torch.tensor(sorted(known), dtype=torch.long)] = float("-inf")
                st[i, oo] = orig
                m_tail.add_ranks([int((st[i] > orig).sum())])

                orig_h = sh[i, ss].item()
                known_h = heads.get((rr, oo, tt), set())
                if known_h:
                    sh[i, torch.tensor(sorted(known_h), dtype=torch.long)] = float("-inf")
                sh[i, ss] = orig_h
                m_head.add_ranks([int((sh[i] > orig_h).sum())])

            if (bi + 1) % 10 == 0 or bi + 1 == n_batch:
                print(f"  [{bi+1}/{n_batch}] tail MRR {m_tail.summary()['MRR']:.4f} "
                      f"head MRR {m_head.summary()['MRR']:.4f}", flush=True)

    res = {"tail": m_tail.summary(), "head": m_head.summary()}
    avg = {k: (res["tail"][k] + res["head"][k]) / 2 for k in ("MRR", "H@1", "H@3", "H@10")}
    avg["n_answer"] = res["tail"]["n_answer"]
    avg["n_query"] = res["tail"]["n_query"]
    res["avg"] = avg
    return res


# ---------------------------------------------------------------------------
# query:按题型评测
# ---------------------------------------------------------------------------

def score_query_type(scorer: TComplExScorer, qs: D.QuerySet,
                     batch_size: int = 256) -> Metric:
    """对单个题型出分数矩阵并排名。目前仅支持纯 LP 原子(Pe / Pt)。"""
    m = Metric()
    q = qs.query
    n = len(qs)
    n_batch = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for bi in range(n_batch):
            lo, hi = bi * batch_size, min((bi + 1) * batch_size, n)
            qb = q[lo:hi]

            if qs.name == "Pe":            # e1, r1, t1 → 实体全表
                scores = scorer.score_entities(qb[:, 0], qb[:, 1], qb[:, 2])
            elif qs.name == "Pt":          # e1, r1, e2 → 时间全表
                scores = scorer.score_times(qb[:, 0], qb[:, 1], qb[:, 2])
            else:
                raise NotImplementedError(f"{qs.name} 的算子尚未实现(M2/M3)")

            scores = scores.clone()
            for i in range(hi - lo):
                easy = qs.easy[lo + i]
                full = qs.answers[lo + i]
                hard = hard_answers(full, easy)
                if easy:
                    scores[i, torch.tensor(sorted(easy), dtype=torch.long)] = float("-inf")
                if not hard:
                    continue
                order = scores[i].argsort(descending=True)
                hard_t = torch.tensor(sorted(hard), dtype=torch.long)
                pos = (order.view(1, -1) == hard_t.view(-1, 1)).nonzero()[:, 1]
                m.add_ranks(pos.tolist())
    return m


def run_query(dataset: str, split: str = "test",
              types: Sequence[str] | None = None,
              limit_types: int | None = None,
              limit: int | None = None) -> Dict[str, Dict[str, float]]:
    scorer = TComplExScorer(checkpoint_path(dataset))
    print(scorer.describe())

    qs_all = D.load_queries(dataset, split, types)
    names = [n for n in D.official_types() if n in qs_all]
    if limit_types:
        names = names[:limit_types]
    print(f"{split} 题型 {len(names)}")

    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        qs = qs_all[name]
        if limit:
            qs = D.QuerySet(name=qs.name, query=qs.query[:limit], answers=qs.answers[:limit],
                            easy=qs.easy[:limit], is_entity_answer=qs.is_entity_answer)
        try:
            m = score_query_type(scorer, qs)
        except NotImplementedError as e:
            print(f"  {name:<18} 跳过 —— {e}")
            continue
        s = m.summary()
        out[name] = s
        print(f"  {name:<18} 查询{s['n_query']:>6} 答案{s['n_answer']:>6}  "
              f"MRR {s['MRR']:.4f}  H@1 {s['H@1']:.4f}  "
              f"H@3 {s['H@3']:.4f}  H@10 {s['H@10']:.4f}", flush=True)
    return out


def print_table(res: Dict[str, Dict[str, float]]) -> None:
    if not res:
        return
    print("\n" + "=" * 90)
    print(f"{'题型':<18}{'MRR':>9}{'H@1':>9}{'H@3':>9}{'H@10':>9}{'查询数':>12}{'hard答案':>12}")
    print("-" * 90)
    for name, s in res.items():
        print(f"{name:<18}{s['MRR']*100:>9.2f}{s['H@1']*100:>9.2f}"
              f"{s['H@3']*100:>9.2f}{s['H@10']*100:>9.2f}"
              f"{s['n_query']:>12}{s['n_answer']:>12}")
    print("-" * 90)

    group_scores = {}
    for g, members in GROUPS.items():
        vals = [res[m]["MRR"] for m in members if m in res]
        if vals:
            group_scores[g] = sum(vals) / len(vals) * 100

    if group_scores:
        print(f"\n{'分组':<10}{'MRR':>9}   成员覆盖")
        for g, v in group_scores.items():
            hit = [m for m in GROUPS[g] if m in res]
            print(f"{g:<10}{v:>9.2f}   {len(hit)}/{len(GROUPS[g])}")
        avg = sum(group_scores.values()) / len(group_scores)
        print(f"\n{'AVG':<10}{avg:>9.2f}   (仅覆盖已实现题型,非最终值)")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="icews14")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", choices=["lp-check", "query"], default="query")
    ap.add_argument("--types", default=None,
                    help="逗号分隔的题型白名单,如 Pe,Pt")
    ap.add_argument("--limit", type=int, default=None, help="每题型取样数(冒烟用)")
    ap.add_argument("--limit-types", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    if args.mode == "lp-check":
        res = run_lp_check(args.dataset, args.split, args.batch_size, args.limit)
        print(f"\n{'':<8}{'MRR':>9}{'H@1':>9}{'H@3':>9}{'H@10':>9}")
        for k in ("tail", "head", "avg"):
            s = res[k]
            print(f"{k:<8}{s['MRR']*100:>9.2f}{s['H@1']*100:>9.2f}"
                  f"{s['H@3']*100:>9.2f}{s['H@10']*100:>9.2f}")
        print("\n对照:已发表 TComplEx ICEWS14 filtered MRR ≈ 52.0 (头尾平均)")
    else:
        types = args.types.split(",") if args.types else None
        res = run_query(args.dataset, args.split, types, args.limit_types, args.limit)
        print_table(res)


if __name__ == "__main__":
    main()

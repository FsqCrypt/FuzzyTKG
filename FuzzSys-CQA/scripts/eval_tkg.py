#!/usr/bin/env python3
"""TKG 评测入口。

两种模式:

  --mode lp-check   用原始四元组做标准 filtered LP 评测(链路核验硬锚点)
  --mode query      按题型评测 CQA(默认)

评测协议逐条镜像 TFLEX(TFLEX/main_TCQE_TFLEX.py:1227-1264 与
ComplexTemporalQueryDataloader.py:129-140):

  1. easy 规则:`if len(easy) >= len(full): easy = {}`(比较对象是 **full**)
  2. 分数矩阵里把 easy 置 -inf,再 `hard = full − easy`
  3. argsort 降序,取每个 hard 答案的 0-based 位置
  4. MRR 累计 1/(rank+1);hits@k 为 rank < k

两个易错点,都逐字对齐 TFLEX:

- 第 1 步的比较对象是 **full**,不是 hard。TFLEX 的 TestDataset 里那个变量
  名叫 `hard_answer`,但存的其实是 full 集合
  (`ComplexTemporalQueryDataloader.py:130-134`)。
- 规则**同时清空掩码**:TFLEX 先 rebind `easy_answer = set()` 再拿它建
  `easy_answer_mask`(同文件 132-139 行),所以规则触发时**什么都不掩**,
  hard 就是 full。这不是可忽略的边界:ICEWS14 上 e3i 有 45.4% 的查询、
  aPt/bPt/between 有 100% 触发该规则。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import torch
from torch import Tensor

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

import tkg_data as D  # noqa: E402
from tcomplex import TComplExScorer  # noqa: E402
from tkg_ops import NaiveOperators  # noqa: E402

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


def split_answers(full: Set[int], easy: Set[int]) -> Tuple[Set[int], Set[int]]:
    """镜像 TFLEX 的 easy/hard 切分,返回 (hard, 用于掩码的 easy)。

    规则 `len(easy) >= len(full)` 同时作用于两者(TFLEX 先 rebind 再建掩码),
    故触发时返回的掩码也是空集。比较对象是 full,不是 hard。
    """
    if len(easy) >= len(full):
        easy = set()
    return full - easy, easy


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

def score_batch(ops: NaiveOperators, name: str, qb: Tensor) -> Tensor:
    """按题型对一批查询出分数矩阵 [b, R]。

    参数位下标与 D.QUERY_TYPES 的类型串一一对应,语义见
    data/tkg/<DS>/<DS>.py 的 query_structures。
    """
    if name == "Pe":
        return ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2])
    if name == "Pt":
        return ops.Pt(qb[:, 0], qb[:, 1], qb[:, 2])

    # 实体多跳:Pe(Pe(e1,r1,t1), r2,t2) → 投影
    if name == "Pe2":
        return ops.project(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]), qb[:, 3], qb[:, 4])
    if name == "Pe3":
        s = ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2])
        s = ops.project(s, qb[:, 3], qb[:, 4])
        return ops.project(s, qb[:, 5], qb[:, 6])

    # 实体交集:And(Pe(e1,r1,t1), Pe(e2,r2,t2)[, Pe(e3,r3,t3)])
    if name == "e2i":
        return ops.And(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                       ops.Pe(qb[:, 3], qb[:, 4], qb[:, 5]))
    if name == "e3i":
        return ops.And(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                       ops.Pe(qb[:, 3], qb[:, 4], qb[:, 5]),
                       ops.Pe(qb[:, 6], qb[:, 7], qb[:, 8]))

    # 时间交集:TimeAnd(Pt(e1,r1,e2), Pt(e3,r2,e4)[, Pt(e5,r3,e6)])
    if name == "t2i":
        return ops.And(ops.Pt(qb[:, 0], qb[:, 1], qb[:, 2]),
                       ops.Pt(qb[:, 3], qb[:, 4], qb[:, 5]))
    if name == "t3i":
        return ops.And(ops.Pt(qb[:, 0], qb[:, 1], qb[:, 2]),
                       ops.Pt(qb[:, 3], qb[:, 4], qb[:, 5]),
                       ops.Pt(qb[:, 6], qb[:, 7], qb[:, 8]))

    # 时间多跳
    if name == "aPt":
        return ops.after(ops.Pt(qb[:, 0], qb[:, 1], qb[:, 2]))
    if name == "bPt":
        return ops.before(ops.Pt(qb[:, 0], qb[:, 1], qb[:, 2]))
    if name == "Pt_sPe":        # Pt(Pe(e1,r1,t1), r2, e2)
        return ops.project_time(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                qb[:, 3], qb[:, 4], "h")
    if name == "Pt_oPe":        # Pt(e1, r1, Pe(e2,r2,t1))
        return ops.project_time(ops.Pe(qb[:, 2], qb[:, 3], qb[:, 4]),
                                qb[:, 1], qb[:, 0], "o")
    if name == "Pe_Pt":         # Pe(e1, r1, Pt(e2,r2,e3))
        return ops.project_via_time(ops.Pt(qb[:, 2], qb[:, 3], qb[:, 4]),
                                    qb[:, 0], qb[:, 1])
    if name == "Pe_aPt":        # Pe(e1, r1, after(Pt(e2,r2,e3)))
        return ops.project_via_time(ops.after(ops.Pt(qb[:, 2], qb[:, 3], qb[:, 4])),
                                    qb[:, 0], qb[:, 1])
    if name == "Pe_bPt":        # Pe(e1, r1, before(Pt(e2,r2,e3)))
        return ops.project_via_time(ops.before(ops.Pt(qb[:, 2], qb[:, 3], qb[:, 4])),
                                    qb[:, 0], qb[:, 1])
    if name == "Pt_sPe_Pt":     # Pt(Pe(s1, r1, Pt(s2,r2,o1)), r3, o2)
        t = ops.Pt(qb[:, 2], qb[:, 3], qb[:, 4])
        e = ops.project_via_time(t, qb[:, 0], qb[:, 1])
        return ops.project_time(e, qb[:, 5], qb[:, 6], "h")
    if name == "Pt_oPe_Pt":     # Pt(s1, r1, Pe(s2, r2, Pt(s3,r3,o1)))
        t = ops.Pt(qb[:, 4], qb[:, 5], qb[:, 6])
        e = ops.project_via_time(t, qb[:, 2], qb[:, 3])
        return ops.project_time(e, qb[:, 1], qb[:, 6], "o")

    # 复合时间交集
    if name == "e2i_Pe":        # And(Pe(Pe(e1,r1,t1), r2, t2), Pe(e2,r3,t3))
        return ops.And(ops.project(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                   qb[:, 3], qb[:, 4]),
                       ops.Pe(qb[:, 5], qb[:, 6], qb[:, 7]))
    if name == "Pe_e2i":        # Pe(e2i(e1,r1,t1, e2,r2,t2), r3, t3)
        return ops.project(ops.And(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                   ops.Pe(qb[:, 3], qb[:, 4], qb[:, 5])),
                           qb[:, 6], qb[:, 7])
    if name == "Pt_se2i":       # Pt(e2i(e1,r1,t1, e2,r2,t2), r3, e3)
        return ops.project_time(ops.And(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                        ops.Pe(qb[:, 3], qb[:, 4], qb[:, 5])),
                                qb[:, 6], qb[:, 7], "h")
    if name == "Pt_oe2i":       # Pt(e1, r1, e2i(e2,r2,t1, e3,r3,t2))
        return ops.project_time(ops.And(ops.Pe(qb[:, 2], qb[:, 3], qb[:, 4]),
                                        ops.Pe(qb[:, 5], qb[:, 6], qb[:, 7])),
                                qb[:, 1], qb[:, 0], "o")
    if name == "t2i_Pe":        # TimeAnd(Pt(Pe(e1,r1,t1), r2, e2), Pt(e3,r3,e4))
        return ops.And(ops.project_time(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                        qb[:, 3], qb[:, 4], "h"),
                       ops.Pt(qb[:, 5], qb[:, 6], qb[:, 7]))
    if name == "Pe_t2i":        # Pe(e1, r1, t2i(e2,r2,e3, e4,r3,e5))
        return ops.project_via_time(ops.And(ops.Pt(qb[:, 2], qb[:, 3], qb[:, 4]),
                                            ops.Pt(qb[:, 5], qb[:, 6], qb[:, 7])),
                                    qb[:, 0], qb[:, 1])
    if name == "Pe_at2i":       # Pe(e1, r1, after(t2i(...)))
        return ops.project_via_time(ops.after(ops.And(ops.Pt(qb[:, 2], qb[:, 3], qb[:, 4]),
                                                      ops.Pt(qb[:, 5], qb[:, 6], qb[:, 7]))),
                                    qb[:, 0], qb[:, 1])
    if name == "Pe_bt2i":       # Pe(e1, r1, before(t2i(...)))
        return ops.project_via_time(ops.before(ops.And(ops.Pt(qb[:, 2], qb[:, 3], qb[:, 4]),
                                                       ops.Pt(qb[:, 5], qb[:, 6], qb[:, 7]))),
                                    qb[:, 0], qb[:, 1])
    if name == "between":       # TimeAnd(after(Pt(e1,r1,e2)), before(Pt(e3,r2,e4)))
        return ops.And(ops.after(ops.Pt(qb[:, 0], qb[:, 1], qb[:, 2])),
                       ops.before(ops.Pt(qb[:, 3], qb[:, 4], qb[:, 5])))

    # 实体否定
    if name == "e2i_N":         # And(Pe(e1,r1,t1), Not(Pe(e2,r2,t2)))
        return ops.And(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                       ops.Not(ops.Pe(qb[:, 3], qb[:, 4], qb[:, 5])))
    if name == "e3i_N":         # And3(Pe1, Pe2, Not(Pe3))
        return ops.And(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                       ops.Pe(qb[:, 3], qb[:, 4], qb[:, 5]),
                       ops.Not(ops.Pe(qb[:, 6], qb[:, 7], qb[:, 8])))
    if name == "Pe_e2i_Pe_NPe":  # Pe(And(Pe1, Not(Pe2)), r3, t3)
        return ops.project(ops.And(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                   ops.Not(ops.Pe(qb[:, 3], qb[:, 4], qb[:, 5]))),
                           qb[:, 6], qb[:, 7])
    if name == "e2i_PeN":       # And(Pe(Pe1, r2, t2), Not(Pe(e2,r3,t3)))
        return ops.And(ops.project(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                   qb[:, 3], qb[:, 4]),
                       ops.Not(ops.Pe(qb[:, 5], qb[:, 6], qb[:, 7])))
    if name == "e2i_NPe":       # And(Not(Pe(Pe1, r2, t2)), Pe(e2,r3,t3))
        return ops.And(ops.Not(ops.project(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                           qb[:, 3], qb[:, 4])),
                       ops.Pe(qb[:, 5], qb[:, 6], qb[:, 7]))

    # 时间否定
    if name == "t2i_N":         # TimeAnd(Pt1, TimeNot(Pt2))
        return ops.And(ops.Pt(qb[:, 0], qb[:, 1], qb[:, 2]),
                       ops.Not(ops.Pt(qb[:, 3], qb[:, 4], qb[:, 5])))
    if name == "t3i_N":         # TimeAnd3(Pt1, Pt2, TimeNot(Pt3))
        return ops.And(ops.Pt(qb[:, 0], qb[:, 1], qb[:, 2]),
                       ops.Pt(qb[:, 3], qb[:, 4], qb[:, 5]),
                       ops.Not(ops.Pt(qb[:, 6], qb[:, 7], qb[:, 8])))
    if name == "Pe_t2i_PtPe_NPt":  # Pe(e1, r1, TimeAnd(Pt(Pe(e2,r2,t2), r3, e3), TimeNot(Pt(e4,r4,e5))))
        inner = ops.And(ops.project_time(ops.Pe(qb[:, 2], qb[:, 3], qb[:, 4]),
                                         qb[:, 5], qb[:, 6], "h"),
                        ops.Not(ops.Pt(qb[:, 7], qb[:, 8], qb[:, 9])))
        return ops.project_via_time(inner, qb[:, 0], qb[:, 1])
    if name == "t2i_PtN":       # TimeAnd(Pt(Pe(e1,r1,t1), r2, e2), TimeNot(Pt(e3,r3,e4)))
        return ops.And(ops.project_time(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                        qb[:, 3], qb[:, 4], "h"),
                       ops.Not(ops.Pt(qb[:, 5], qb[:, 6], qb[:, 7])))
    if name == "t2i_NPt":       # TimeAnd(TimeNot(Pt(Pe(e1,r1,t1), r2, e2)), Pt(e3,r3,e4))
        return ops.And(ops.Not(ops.project_time(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                                qb[:, 3], qb[:, 4], "h")),
                       ops.Pt(qb[:, 5], qb[:, 6], qb[:, 7]))

    # 并集
    if name == "e2u":           # Or(Pe(e1,r1,t1), Pe(e2,r2,t2))
        return ops.Or(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                      ops.Pe(qb[:, 3], qb[:, 4], qb[:, 5]))
    if name == "Pe_e2u":        # Pe(Or(Pe1, Pe2), r3, t3)
        return ops.project(ops.Or(ops.Pe(qb[:, 0], qb[:, 1], qb[:, 2]),
                                  ops.Pe(qb[:, 3], qb[:, 4], qb[:, 5])),
                           qb[:, 6], qb[:, 7])
    if name == "t2u":           # TimeOr(Pt(e1,r1,e2), Pt(e3,r2,e4))
        return ops.Or(ops.Pt(qb[:, 0], qb[:, 1], qb[:, 2]),
                      ops.Pt(qb[:, 3], qb[:, 4], qb[:, 5]))
    if name == "Pe_t2u":        # Pe(e1, r1, TimeOr(Pt(e2,r2,e3), Pt(e4,r3,e5)))
        return ops.project_via_time(ops.Or(ops.Pt(qb[:, 2], qb[:, 3], qb[:, 4]),
                                           ops.Pt(qb[:, 5], qb[:, 6], qb[:, 7])),
                                    qb[:, 0], qb[:, 1])

    raise NotImplementedError(f"{name} 的算子尚未实现")


# 用了 project 的题型,内存占用随 batch 线性放大,用小批
_HEAVY = {"Pe2", "Pe3", "e2i_Pe", "Pe_e2i", "e2i_PeN", "e2i_NPe",
          "Pe_e2u", "Pe_e2i_Pe_NPe"}


def score_query_type(ops: NaiveOperators, qs: D.QuerySet,
                     batch_size: int = 256) -> Metric:
    """对单个题型出分数矩阵并排名。"""
    m = Metric()
    q = qs.query
    n = len(qs)
    bs = min(batch_size, 64) if qs.name in _HEAVY else batch_size
    n_batch = (n + bs - 1) // bs

    with torch.no_grad():
        for bi in range(n_batch):
            lo, hi = bi * bs, min((bi + 1) * bs, n)
            qb = q[lo:hi]
            scores = score_batch(ops, qs.name, qb).clone()
            for i in range(hi - lo):
                hard, easy_mask = split_answers(qs.answers[lo + i], qs.easy[lo + i])
                if easy_mask:
                    scores[i, torch.tensor(sorted(easy_mask), dtype=torch.long)] = float("-inf")
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
              limit: int | None = None,
              batch_size: int = 256,
              topk: int = 10,
              calibrate: str = "none") -> Dict[str, Dict[str, float]]:
    scorer = TComplExScorer(checkpoint_path(dataset))
    print(scorer.describe())
    print(f"朴素算子: top-k 投影 k={topk},校准 calibrate={calibrate}")

    ops = NaiveOperators(scorer, topk=topk, calibrate=calibrate)

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
            m = score_query_type(ops, qs, batch_size)
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
    ap.add_argument("--topk", type=int, default=10, help="多跳投影保留的中间实体数")
    ap.add_argument("--calibrate", choices=["none", "rank"], default="none",
                    help="组合前的逐路校准;none=计划里的字面 min(默认)")
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
        res = run_query(args.dataset, args.split, types, args.limit_types, args.limit,
                        args.batch_size, args.topk, args.calibrate)
        print_table(res)


if __name__ == "__main__":
    main()

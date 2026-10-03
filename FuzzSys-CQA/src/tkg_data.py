"""TKG 数据集加载 —— TFLEX 格式的查询 jsonl 与原始四元组。

数据布局(相对仓库根 CQA/):

    data/tkg/<dataset>/meta.json                 词表:entity2idx/relation2idx/timestamp2idx
    data/tkg/<dataset>/{train,valid,test}.jsonl  查询与答案
    FuzzSys-CQA/third_party/TFLEX/data/<DS>/     原始四元组 TSV(s\tr\to\tt)

查询 jsonl 每行:

    {"query_name": "Pe", "query": [e, r, t], "answer": [...], "easy_answer": [...]}

- `answer` 是 **full 答案**(该 split 知识图上的全部答案)
- `easy_answer` 是 **train 图上的答案**,是 full 的子集;部分行没有该字段
- 缺失 easy_answer 视为空集

题型与参数位的对应关系逐字抄自 `data/tkg/<dataset>/<DS>.py` 的 `query_name_to_args`
(TFLEX 官方随数据集发布的脚本,是这份数据的权威定义)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]          # CQA/
_SRC = Path(__file__).resolve().parent              # FuzzSys-CQA/src/
TFLEX_DATA = _SRC.parent / "third_party/TFLEX/data"  # CQA/FuzzSys-CQA/third_party/TFLEX/data

# 数据集名 -> (本地查询目录名, TFLEX 原始四元组目录名)
DATASETS: Dict[str, Tuple[str, str]] = {
    "icews14": ("icews14", "ICEWS14"),
    "icews05_15": ("icews05_15", "ICEWS05-15"),
    "gdelt": ("gdelt", "GDELT"),
}

# ---------------------------------------------------------------------------
# 题型定义
# ---------------------------------------------------------------------------

# 逐字抄自 ICEWS14.py:38-89 的 query_name_to_args。
# 参数位命名约定:e*/s*/o* = 实体,r* = 关系,t* = 时间戳(TFLEX 官方规则)。
QUERY_ARGS: Dict[str, List[str]] = {
    # 1. 一跳 Pe / Pt
    "Pe": ["e1", "r1", "t1"],
    "Pt": ["e1", "r1", "e2"],
    # 2. 实体多跳
    "Pe2": ["e1", "r1", "t1", "r2", "t2"],
    "Pe3": ["e1", "r1", "t1", "r2", "t2", "r3", "t3"],
    # 3. 时间多跳
    "aPt": ["s", "r", "o"],
    "bPt": ["s", "r", "o"],
    "Pt_sPe": ["e1", "r1", "t1", "r2", "e2"],
    "Pt_oPe": ["e1", "r1", "e2", "r2", "t1"],
    "Pe_Pt": ["e1", "r1", "e2", "r2", "e3"],
    "Pe_aPt": ["e1", "r1", "e2", "r2", "e3"],
    "Pe_bPt": ["e1", "r1", "e2", "r2", "e3"],
    "Pe_nPt": ["e1", "r1", "e2", "r2", "e3"],
    "Pt_sPe_Pt": ["s1", "r1", "s2", "r2", "o1", "r3", "o2"],
    "Pt_oPe_Pt": ["s1", "r1", "s2", "r2", "s3", "r3", "o1"],
    # 4. 实体交集 / 时间交集
    "e2i": ["e1", "r1", "t1", "e2", "r2", "t2"],
    "e3i": ["e1", "r1", "t1", "e2", "r2", "t2", "e3", "r3", "t3"],
    "t2i": ["e1", "r1", "e2", "e3", "r2", "e4"],
    "t3i": ["e1", "r1", "e2", "e3", "r2", "e4", "e5", "r3", "e6"],
    # 5. 复合时间交集
    "e2i_Pe": ["e1", "r1", "t1", "r2", "t2", "e2", "r3", "t3"],
    "Pe_e2i": ["e1", "r1", "t1", "e2", "r2", "t2", "r3", "t3"],
    "Pt_se2i": ["e1", "r1", "t1", "e2", "r2", "t2", "r3", "e3"],
    "Pt_oe2i": ["e1", "r1", "e2", "r2", "t1", "e3", "r3", "t2"],
    "t2i_Pe": ["e1", "r1", "t1", "r2", "e2", "e3", "r3", "e4"],
    "Pe_t2i": ["e1", "r1", "e2", "r2", "e3", "e4", "r3", "e5"],
    "Pe_at2i": ["e1", "r1", "e2", "r2", "e3", "e4", "r3", "e5"],
    "Pe_bt2i": ["e1", "r1", "e2", "r2", "e3", "e4", "r3", "e5"],
    "Pe_nt2i": ["e1", "r1", "e2", "r2", "e3", "e4", "r3", "e5"],
    "between": ["e1", "r1", "e2", "e3", "r2", "e4"],
    # 6. 实体否定
    "e2i_N": ["e1", "r1", "t1", "e2", "r2", "t2"],
    "e3i_N": ["e1", "r1", "t1", "e2", "r2", "t2", "e3", "r3", "t3"],
    "Pe_e2i_Pe_NPe": ["e1", "r1", "t1", "e2", "r2", "t2", "r3", "t3"],
    "e2i_NPe": ["e1", "r1", "t1", "r2", "t2", "e2", "r3", "t3"],
    "e2i_PeN": ["e1", "r1", "t1", "r2", "t2", "e2", "r3", "t3"],
    # 7. 时间否定
    "t2i_N": ["e1", "r1", "e2", "e3", "r2", "e4"],
    "t3i_N": ["e1", "r1", "e2", "e3", "r2", "e4", "e5", "r3", "e6"],
    "Pe_t2i_PtPe_NPt": ["e1", "r1", "e2", "r2", "t2", "r3", "e3", "e4", "r4", "e5"],
    "t2i_NPt": ["e1", "r1", "t1", "r2", "e2", "e3", "r3", "e4"],
    "t2i_PtN": ["e1", "r1", "t1", "r2", "e2", "e3", "r3", "e4"],
    # 8. 并集
    "e2u": ["e1", "r1", "t1", "e2", "r2", "t2"],
    "Pe_e2u": ["e1", "r1", "t1", "e2", "r2", "t2", "r3", "t3"],
    "t2u": ["e1", "r1", "e2", "e3", "r2", "e4"],
    "Pe_t2u": ["e1", "r1", "e2", "r2", "e3", "e4", "r3", "e5"],
}

# 本地数据比论文多出的题型:更早版本的生成脚本留下的,论文表格中没有。
# 评测必须剔除,否则与论文不可比。
NON_OFFICIAL_TYPES: Set[str] = {"Pe_nPt", "Pe_nt2i"}


def _arg_kind(arg: str) -> str:
    """参数位名字 → 'e' / 'r' / 't'(TFLEX 命名约定:r* 关系,t* 时间,其余实体)。"""
    if arg.startswith("r"):
        return "r"
    if arg.startswith("t"):
        return "t"
    return "e"


# 题型 -> 每个参数位的类型串,如 Pe -> "ert"。由 QUERY_ARGS 机械推导,避免手抄出错。
QUERY_TYPES: Dict[str, str] = {
    name: "".join(_arg_kind(a) for a in args) for name, args in QUERY_ARGS.items()
}


def official_types() -> List[str]:
    """论文口径的 40 个题型(按 QUERY_ARGS 的声明顺序)。"""
    return [q for q in QUERY_ARGS if q not in NON_OFFICIAL_TYPES]


def is_entity_answer(query_name: str) -> bool:
    """答案落在实体域还是在时间戳域。

    镜像 TFLEX_DSL.py:103-104 的 is_to_predict_entity_set。
    """
    return query_name.startswith("e") or query_name.startswith("Pe")


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclass
class QuerySet:
    """一个题型的全部查询。"""
    name: str
    query: Tensor                 # [N, L] long, 各参数位语义见 QUERY_TYPES[name]
    answers: List[Set[int]]       # full 答案
    easy: List[Set[int]]          # train 图答案(full 的子集)
    is_entity_answer: bool        # True=在实体域排序,False=在时间戳域

    def __len__(self) -> int:
        return self.query.shape[0]


@dataclass
class Vocab:
    entity: Dict[str, int]
    relation: Dict[str, int]
    timestamp: Dict[str, int]

    @property
    def n_entity(self) -> int:
        return len(self.entity)

    @property
    def n_relation(self) -> int:
        return len(self.relation)

    @property
    def n_time(self) -> int:
        return len(self.timestamp)


@dataclass
class TripleStore:
    """原始四元组,已 id 化。"""
    train: List[Tuple[int, int, int, int]]
    valid: List[Tuple[int, int, int, int]]
    test: List[Tuple[int, int, int, int]]

    def _index(self) -> Tuple[Dict[Tuple[int, int, int], Set[int]],
                              Dict[Tuple[int, int, int], Set[int]]]:
        tails: Dict[Tuple[int, int, int], Set[int]] = {}
        heads: Dict[Tuple[int, int, int], Set[int]] = {}
        for name in ("train", "valid", "test"):
            for s, r, o, t in getattr(self, name):
                tails.setdefault((s, r, t), set()).add(o)
                heads.setdefault((r, o, t), set()).add(s)
        return tails, heads

    def filtered(self) -> Tuple[Dict[Tuple[int, int, int], Set[int]],
                                Dict[Tuple[int, int, int], Set[int]]]:
        """LP 评测用的 filtered 索引,均为 train+valid+test 全量:

            tails[(s, r, τ)] = 全部真实尾实体
            heads[(r, o, τ)] = 全部真实头实体
        """
        return self._index()


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def dataset_paths(dataset: str) -> Tuple[Path, Path]:
    """返回 (本地查询目录, TFLEX 原始四元组目录)。"""
    if dataset not in DATASETS:
        raise KeyError(f"未知数据集 {dataset!r},可选 {sorted(DATASETS)}")
    local, raw = DATASETS[dataset]
    return ROOT / "data/tkg" / local, TFLEX_DATA / raw


def load_vocab(dataset: str) -> Vocab:
    local, _ = dataset_paths(dataset)
    meta = json.loads((local / "meta.json").read_text())
    return Vocab(
        entity={x["name"].strip(): x["id"] for x in meta["entity2idx"]},
        relation={x["name"].strip(): x["id"] for x in meta["relation2idx"]},
        timestamp={x["name"].strip(): x["id"] for x in meta["timestamp2idx"]},
    )


def load_triples(dataset: str, vocab: Vocab | None = None) -> TripleStore:
    """读 TFLEX 原始四元组 TSV(s\\tr\\to\\tt)并转成 id。"""
    vocab = vocab or load_vocab(dataset)
    _, raw_dir = dataset_paths(dataset)

    def read(split: str) -> List[Tuple[int, int, int, int]]:
        rows: List[Tuple[int, int, int, int]] = []
        with open(raw_dir / split) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                s, r, o, t = (p.strip() for p in parts)
                rows.append((vocab.entity[s], vocab.relation[r],
                             vocab.entity[o], vocab.timestamp[t]))
        return rows

    return TripleStore(train=read("train"), valid=read("valid"), test=read("test"))


def load_queries(dataset: str, split: str = "test",
                 types: Sequence[str] | None = None) -> Dict[str, QuerySet]:
    """读查询 jsonl,按题型归组。只返回官方 40 题型。"""
    wanted = set(official_types())
    if types:
        unknown = set(types) - set(QUERY_ARGS)
        if unknown:
            raise KeyError(f"未定义的题型: {sorted(unknown)}")
        excluded = set(types) & NON_OFFICIAL_TYPES
        if excluded:
            raise KeyError(f"非官方题型,评测口径不含: {sorted(excluded)}")
        wanted &= set(types)

    local, _ = dataset_paths(dataset)

    bucket: Dict[str, List[Tuple[List[int], Set[int], Set[int]]]] = {}
    with open(local / f"{split}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            name = row["query_name"]
            if name not in wanted:
                continue
            bucket.setdefault(name, []).append(
                (row["query"], set(row["answer"]), set(row.get("easy_answer") or ()))
            )

    out: Dict[str, QuerySet] = {}
    for name, rows in bucket.items():
        out[name] = QuerySet(
            name=name,
            query=torch.tensor([r[0] for r in rows], dtype=torch.long),
            answers=[r[1] for r in rows],
            easy=[r[2] for r in rows],
            is_entity_answer=is_entity_answer(name),
        )
    return out

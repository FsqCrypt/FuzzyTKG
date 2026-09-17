"""把 TComplEx checkpoint(按名字排列的 TSV)重排为 TFLEX 数据集 ID 顺序。

用法:
    python scripts/align_tcomplex_to_tflex.py [--dataset icews14|icews05_15]

输入:
    data/tkg/checkpoints/tcomplex-<dataset>/{entities,relations,times}.tsv   (word2vec TSV)
    data/tkg/<dataset>/meta.json                                           (TFLEX 词汇表)

输出:
    data/tkg/checkpoints/tcomplex-<dataset>/tcomplex_<dataset>_tflex_ids.pt
        {entity: [N, 512], relation: [M, 512], time: [T, 512]}
        全部按 TFLEX id 顺序;512 = 256 复数维(前 256 实部,后 256 虚部)。

对齐规则:按名字匹配;名字两侧空白统一剥离(已知差异:TFLEX 的
'Găgăuzia\\xa0' 与 checkpoint 的 'Găgăuzia' 是同一实体)。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]  # CQA/

DATASET_DIRS = {
    "icews14": "icews14",
    "icews05_15": "icews05_15",
}

CHECKPOINT_DIRS = {
    "icews14": "tcomplex-icews14",
    "icews05_15": "tcomplex-icews05-15",
}


def load_tsv(path: Path):
    """word2vec TSV -> (names: list[str], arr: np.ndarray [n, dim])"""
    with open(path) as f:
        n, d = map(int, f.readline().split())
        names, rows = [], []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            names.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    arr = np.asarray(rows, dtype=np.float32)
    assert arr.shape == (n, d), f"{path}: 期望 {(n, d)}, 实际 {arr.shape}"
    return names, arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_DIRS), default="icews14")
    args = parser.parse_args()

    dataset_dir = ROOT / "data/tkg" / DATASET_DIRS[args.dataset]
    ckpt_dir = ROOT / "data/tkg/checkpoints" / CHECKPOINT_DIRS[args.dataset]
    meta_path = dataset_dir / "meta.json"
    out_path = ckpt_dir / f"{CHECKPOINT_DIRS[args.dataset].replace('-', '_')}_tflex_ids.pt"

    meta = json.load(open(meta_path))

    result = {}
    for kind, key, fname in [
        ("entity", "entity2idx", "entities.tsv"),
        ("relation", "relation2idx", "relations.tsv"),
        ("time", "timestamp2idx", "times.tsv"),
    ]:
        tflex = {x["name"].strip(): x["id"] for x in meta[key]}
        names, arr = load_tsv(ckpt_dir / fname)
        row_of = {n.strip(): i for i, n in enumerate(names)}

        out = np.zeros((len(tflex), arr.shape[1]), dtype=np.float32)
        missing = []
        for name, tid in tflex.items():
            row = row_of.get(name)
            if row is None:
                missing.append(name)
                continue
            out[tid] = arr[row]
        if missing:
            print(f"[{kind}] 无法对齐 {len(missing)} 个: {missing[:5]}")
            if len(missing) > len(tflex) * 0.01:
                sys.exit(f"[{kind}] 对齐失败,缺失过多,请检查词汇表来源")
        result[kind] = torch.from_numpy(out)
        print(f"[{kind}] {len(tflex)} 个 -> 已按 TFLEX id 重排,维度 {arr.shape[1]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(f"已保存 {out_path}")


if __name__ == "__main__":
    main()

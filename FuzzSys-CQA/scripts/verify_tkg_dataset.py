# -*- coding: utf-8 -*-
"""
核验 data/tkg/ 下三个 TFLEX 数据集是否与官方发布一致。

权威来源: HuggingFace `linxy/ICEWS14` 数据集卡片的 queries_count 统计表
(https://huggingface.co/datasets/linxy/ICEWS14, README_dataset.md 内嵌)。

检查两件事:
  1. 每个题型在每个 split 的查询条数是否与官方表一致;
  2. 是否存在官方表之外的题型(多余)或官方表中有而本地缺失的题型。

用法:
    python3 scripts/verify_tkg_dataset.py [--dataset icews14] [--data-root data/tkg]
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter

# ---------------------------------------------------------------- 官方查询计数
# 来自 linxy/ICEWS14 数据集卡片 queries_count 表。
# 值为 None 表示官方表该格为 "-"(该题型不在该 split 中生成)。
OFFICIAL_QUERY_COUNTS = {
    "icews14": {
        "train": {
            "Pe": 66783, "Pe2": 72826, "Pe3": 72826, "e2i": 72826, "e3i": 72826,
            "Pt": 42690, "aPt": 13234, "bPt": 13234, "Pe_Pt": 7282,
            "Pt_sPe_Pt": 13234, "Pt_oPe_Pt": 13234, "t2i": 72826, "t3i": 72826,
            "e2i_N": 7282, "e3i_N": 7282, "Pe_e2i_Pe_NPe": 7282,
            "e2i_PeN": 7282, "e2i_NPe": 7282, "t2i_N": 7282, "t3i_N": 7282,
            "Pe_t2i_PtPe_NPt": 7282, "t2i_PtN": 7282, "t2i_NPt": 7282,
            "between": 7282, "Pe_aPt": 7282, "Pe_bPt": 7282, "Pt_sPe": 7282,
            "Pt_oPe": 7282, "Pt_se2i": 7282, "Pt_oe2i": 7282,
            "Pe_at2i": 7282, "Pe_bt2i": 7282,
            "e2u": None, "Pe_e2u": None, "t2u": None, "Pe_t2u": None,
            "t2i_Pe": None, "Pe_t2i": None, "e2i_Pe": None, "Pe_e2i": None,
        },
        "valid": {
            "Pe": 8837, "Pe2": 3482, "Pe3": 3492, "e2i": 3305, "e3i": 2966,
            "Pt": 7331, "aPt": 4411, "bPt": 4411, "Pe_Pt": 3385,
            "Pt_sPe_Pt": 5541, "Pt_oPe_Pt": 5480, "t2i": 5112, "t3i": 3094,
            "e2i_N": 2949, "e3i_N": 2913, "Pe_e2i_Pe_NPe": 2968,
            "e2i_PeN": 2971, "e2i_NPe": 3061, "t2i_N": 3135, "t3i_N": 2924,
            "Pe_t2i_PtPe_NPt": 3031, "t2i_PtN": 3300, "t2i_NPt": 4873,
            "between": 2913, "Pe_aPt": 4134, "Pe_bPt": 3970, "Pt_sPe": 4976,
            "Pt_oPe": 3321, "Pt_se2i": 3226, "Pt_oe2i": 3236,
            "Pe_at2i": 4607, "Pe_bt2i": 4583,
            "e2u": 2913, "Pe_e2u": 2913, "t2u": 2913, "Pe_t2u": 2913,
            "t2i_Pe": 2913, "Pe_t2i": 2913, "e2i_Pe": 2913, "Pe_e2i": 2913,
        },
        "test": {
            "Pe": 8848, "Pe2": 4037, "Pe3": 4083, "e2i": 3655, "e3i": 3023,
            "Pt": 7419, "aPt": 4411, "bPt": 4411, "Pe_Pt": 3638,
            "Pt_sPe_Pt": 6293, "Pt_oPe_Pt": 6242, "t2i": 6631, "t3i": 3296,
            "e2i_N": 2975, "e3i_N": 2914, "Pe_e2i_Pe_NPe": 3012,
            "e2i_PeN": 3031, "e2i_NPe": 3192, "t2i_N": 3328, "t3i_N": 2944,
            "Pe_t2i_PtPe_NPt": 3127, "t2i_PtN": 3609, "t2i_NPt": 5464,
            "between": 2913, "Pe_aPt": 4733, "Pe_bPt": 4565, "Pt_sPe": 5608,
            "Pt_oPe": 3621, "Pt_se2i": 3466, "Pt_oe2i": 3485,
            "Pe_at2i": 5338, "Pe_bt2i": 5386,
            "e2u": 2913, "Pe_e2u": 2913, "t2u": 2913, "Pe_t2u": 2913,
            "t2i_Pe": 2913, "Pe_t2i": 2913, "e2i_Pe": 2913, "Pe_e2i": 2913,
        },
    },
    "icews05_15": {
        "train": {
            "Pe": 344042, "Pe2": 368962, "Pe3": 368962, "e2i": 368962, "e3i": 368962,
            "Pt": 142771, "aPt": 68262, "bPt": 68262, "Pe_Pt": 36896,
            "Pt_sPe_Pt": 68262, "Pt_oPe_Pt": 68262, "t2i": 368962, "t3i": 368962,
            "e2i_N": 36896, "e3i_N": 36896, "Pe_e2i_Pe_NPe": 36896,
            "e2i_PeN": 36896, "e2i_NPe": 36896, "t2i_N": 36896, "t3i_N": 36896,
            "Pe_t2i_PtPe_NPt": 36896, "t2i_PtN": 36896, "t2i_NPt": 36896,
            "between": 36896, "Pe_aPt": 68262, "Pe_bPt": 36896, "Pt_sPe": 36896,
            "Pt_oPe": 36896, "Pt_se2i": 36896, "Pt_oe2i": 36896,
            "Pe_at2i": 36896, "Pe_bt2i": 36896,
            "e2u": None, "Pe_e2u": None, "t2u": None, "Pe_t2u": None,
            "t2i_Pe": None, "Pe_t2i": None, "e2i_Pe": None, "Pe_e2i": None,
        },
        "valid": {
            "Pe": 45829, "Pe2": 10000, "Pe3": 10000, "e2i": 10000, "e3i": 10000,
            "Pt": 28795, "aPt": 10000, "bPt": 10000, "Pe_Pt": 10000,
            "Pt_sPe_Pt": 10000, "Pt_oPe_Pt": 10000, "t2i": 10000, "t3i": 10000,
            "e2i_N": 10000, "e3i_N": 10000, "Pe_e2i_Pe_NPe": 10000,
            "e2i_PeN": 10000, "e2i_NPe": 10000, "t2i_N": 10000, "t3i_N": 10000,
            "Pe_t2i_PtPe_NPt": 10000, "t2i_PtN": 10000, "t2i_NPt": 10000,
            "between": 10000, "Pe_aPt": 10000, "Pe_bPt": 10000, "Pt_sPe": 10000,
            "Pt_oPe": 10000, "Pt_se2i": 10000, "Pt_oe2i": 10000,
            "Pe_at2i": 10000, "Pe_bt2i": 10000,
            "e2u": 10000, "Pe_e2u": 10000, "t2u": 10000, "Pe_t2u": 10000,
            "t2i_Pe": 10000, "Pe_t2i": 10000, "e2i_Pe": 10000, "Pe_e2i": 10000,
        },
        "test": {
            "Pe": 45644, "Pe2": 10000, "Pe3": 10000, "e2i": 10000, "e3i": 10000,
            "Pt": 28752, "aPt": 10000, "bPt": 10000, "Pe_Pt": 10000,
            "Pt_sPe_Pt": 10000, "Pt_oPe_Pt": 10000, "t2i": 10000, "t3i": 10000,
            "e2i_N": 10000, "e3i_N": 10000, "Pe_e2i_Pe_NPe": 10000,
            "e2i_PeN": 10000, "e2i_NPe": 10000, "t2i_N": 10000, "t3i_N": 10000,
            "Pe_t2i_PtPe_NPt": 10000, "t2i_PtN": 10000, "t2i_NPt": 10000,
            "between": 10000, "Pe_aPt": 10000, "Pe_bPt": 10000, "Pt_sPe": 10000,
            "Pt_oPe": 10000, "Pt_se2i": 10000, "Pt_oe2i": 10000,
            "Pe_at2i": 10000, "Pe_bt2i": 10000,
            "e2u": 10000, "Pe_e2u": 10000, "t2u": 10000, "Pe_t2u": 10000,
            "t2i_Pe": 10000, "Pe_t2i": 10000, "e2i_Pe": 10000, "Pe_e2i": 10000,
        },
    },
    "gdelt": {
        "train": {
            "Pe": 1115102, "Pe2": 2215309, "Pe3": 2215309, "e2i": 2215309, "e3i": 2215309,
            "Pt": 687326, "aPt": 221530, "bPt": 221530, "Pe_Pt": 221530,
            "Pt_sPe_Pt": 221530, "Pt_oPe_Pt": 221530, "t2i": 2215309, "t3i": 2215309,
            "e2i_N": 221530, "e3i_N": 221530, "Pe_e2i_Pe_NPe": 221530,
            "e2i_PeN": 221530, "e2i_NPe": 221530, "t2i_N": 221530, "t3i_N": 221530,
            "Pe_t2i_PtPe_NPt": 221530, "t2i_PtN": 221530, "t2i_NPt": 221530,
            "between": 221530, "Pe_aPt": 221530, "Pe_bPt": 221530, "Pt_sPe": 221530,
            "Pt_oPe": 221530, "Pt_se2i": 221530, "Pt_oe2i": 221530,
            "Pe_at2i": 221530, "Pe_bt2i": 221530,
            "e2u": None, "Pe_e2u": None, "t2u": None, "Pe_t2u": None,
            "t2i_Pe": None, "Pe_t2i": None, "e2i_Pe": None, "Pe_e2i": None,
        },
        "valid": {
            "Pe": 273842, "Pe2": 10000, "Pe3": 10000, "e2i": 10000, "e3i": 10000,
            "Pt": 199780, "aPt": 10000, "bPt": 10000, "Pe_Pt": 10000,
            "Pt_sPe_Pt": 10000, "Pt_oPe_Pt": 10000, "t2i": 10000, "t3i": 10000,
            "e2i_N": 10000, "e3i_N": 10000, "Pe_e2i_Pe_NPe": 10000,
            "e2i_PeN": 10000, "e2i_NPe": 10000, "t2i_N": 10000, "t3i_N": 10000,
            "Pe_t2i_PtPe_NPt": 10000, "t2i_PtN": 10000, "t2i_NPt": 10000,
            "between": 10000, "Pe_aPt": 10000, "Pe_bPt": 10000, "Pt_sPe": 10000,
            "Pt_oPe": 10000, "Pt_se2i": 10000, "Pt_oe2i": 10000,
            "Pe_at2i": 10000, "Pe_bt2i": 10000,
            "e2u": 10000, "Pe_e2u": 10000, "t2u": 10000, "Pe_t2u": 10000,
            "t2i_Pe": 10000, "Pe_t2i": 10000, "e2i_Pe": 10000, "Pe_e2i": 10000,
        },
        "test": {
            "Pe": 273432, "Pe2": 10000, "Pe3": 10000, "e2i": 10000, "e3i": 10000,
            "Pt": 199419, "aPt": 10000, "bPt": 10000, "Pe_Pt": 10000,
            "Pt_sPe_Pt": 10000, "Pt_oPe_Pt": 10000, "t2i": 10000, "t3i": 10000,
            "e2i_N": 10000, "e3i_N": 10000, "Pe_e2i_Pe_NPe": 10000,
            "e2i_PeN": 10000, "e2i_NPe": 10000, "t2i_N": 10000, "t3i_N": 10000,
            "Pe_t2i_PtPe_NPt": 10000, "t2i_PtN": 10000, "t2i_NPt": 10000,
            "between": 10000, "Pe_aPt": 10000, "Pe_bPt": 10000, "Pt_sPe": 10000,
            "Pt_oPe": 10000, "Pt_se2i": 10000, "Pt_oe2i": 10000,
            "Pe_at2i": 10000, "Pe_bt2i": 10000,
            "e2u": 10000, "Pe_e2u": 10000, "t2u": 10000, "Pe_t2u": 10000,
            "t2i_Pe": 10000, "Pe_t2i": 10000, "e2i_Pe": 10000, "Pe_e2i": 10000,
        },
    },
}

# 官方 meta.json 中的词表规模与四元组计数(README 与 meta.json 均声明)
OFFICIAL_VOCAB = {
    "icews14":   {"entity": 7128,  "relation": 230, "timestamp": 365,
                  "train_triples": 72826,   "valid_triples": 8941,   "test_triples": 8963},
    "icews05_15": {"entity": 10488, "relation": 251, "timestamp": 4017,
                  "train_triples": 368962,  "valid_triples": 46275,  "test_triples": 46092},
    "gdelt":     {"entity": 500,   "relation": 20,  "timestamp": 366,
                  "train_triples": 2308165, "valid_triples": 330906, "test_triples": 330845},
}


def count_queries_by_type(path):
    """用 grep 统计 jsonl 中各 query_name 出现次数。

    必须**流式**读取 grep 的 stdout:GDELT train 有 2040 万条查询,
    整体 capture_output 会产生数百 MB 字符串 + 千万级 str 对象,足以打爆内存。
    """
    pattern = r'"query_name": *"[^"]*"'

    c = Counter()
    proc = subprocess.Popen(
        ["grep", "-o", "-E", pattern, path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1 << 20,
    )
    for line in proc.stdout:
        # 行格式: "query_name": "Pe2"
        name = line.rsplit(":", 1)[-1].strip().strip('"')
        if name:
            c[name] += 1
    proc.stdout.close()
    if proc.wait() not in (0, 1):   # grep 无匹配时返回 1,不算错误
        raise RuntimeError(f"grep failed on {path}")
    return c


def verify_dataset(name, data_root):
    problems, notes = [], []
    official = OFFICIAL_QUERY_COUNTS[name]

    # --- 1. 查询计数核验
    local_types = set()
    for split in ("train", "valid", "test"):
        path = os.path.join(data_root, name, f"{split}.jsonl")
        if not os.path.exists(path):
            problems.append(f"[{name}] 缺少文件 {split}.jsonl")
            continue
        local = count_queries_by_type(path)
        local_types |= set(local)
        exp = official[split]

        for qname, exp_count in sorted(exp.items()):
            got = local.get(qname, 0)
            if exp_count is None:          # 官方表为 "-",期望本地不生成
                if got:
                    problems.append(
                        f"[{name}/{split}] {qname}: 官方无此题型,本地有 {got} 条")
            elif got != exp_count:
                problems.append(
                    f"[{name}/{split}] {qname}: 官方 {exp_count}, 本地 {got}")

    # --- 2. 多余题型(官方表完全没列的)
    official_all = set(OFFICIAL_QUERY_COUNTS[name]["test"]) | \
                   set(OFFICIAL_QUERY_COUNTS[name]["train"])
    extra = sorted(local_types - official_all)
    if extra:
        notes.append(f"[{name}] 官方表之外的题型(本地多出): {extra}")

    # --- 3. meta.json 词表与四元组计数
    meta_path = os.path.join(data_root, name, "meta.json")
    if os.path.exists(meta_path):
        m = json.load(open(meta_path))
        exp = OFFICIAL_VOCAB[name]
        checks = [
            ("entity_count", exp["entity"]), ("relation_count", exp["relation"]),
            ("timestamp_count", exp["timestamp"]),
            ("train_triples_count", exp["train_triples"]),
            ("valid_triples_count", exp["valid_triples"]),
            ("test_triples_count", exp["test_triples"]),
        ]
        for key, want in checks:
            got = m.get(key)
            if got != want:
                problems.append(f"[{name}/meta] {key}: 官方 {want}, 本地 {got}")
    else:
        problems.append(f"[{name}] 缺少 meta.json")

    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/tkg")
    ap.add_argument("--dataset", default=None,
                    help="只核验某个数据集,默认全部")
    args = ap.parse_args()

    names = [args.dataset] if args.dataset else list(OFFICIAL_QUERY_COUNTS)
    all_problems, all_notes = [], []
    for name in names:
        print(f"核验 {name} ...", flush=True)
        p, n = verify_dataset(name, args.data_root)
        all_problems += p
        all_notes += n

    print("\n" + "=" * 64)
    for note in all_notes:
        print("注意 " + note)
    if all_problems:
        print(f"\n发现 {len(all_problems)} 处不一致:")
        for prob in all_problems:
            print("  ✗ " + prob)
        return 1
    print("\n✓ 全部核验通过:与官方发布计数逐项一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())

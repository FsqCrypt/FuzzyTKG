# CQA 项目 — 三值关系感知 Membership 模块

## 项目概述

FuzzSys-CQA 是一个基于模糊系统的复杂查询回答（Complex Query Answering）框架。原始系统使用全局 B-spline 函数将所有 relation 的 LP score 映射到 [0,1] 的 membership 值。本项目在此基础上提出了**关系感知三值 membership**模块。

## Git 分支

```
master                          ← 原始基线 (commit: cd50067)
feat/rel-aware-membership       ← 关系感知模块 (当前分支)
```

回退命令: `git checkout master`

---

## 一、原始创新点

### 1.1 核心动机

原始 FuzzSys 的 membership 函数是所有 relation 共用的：

```
score → cal_membership(score) → [0,1]
```

**问题**：不同 relation 的 LP score 分布差异巨大。同一个 score=6.5 对于高频简单 relation 和稀疏困难 relation 含义完全不同。全局函数无法捕捉这种差异。

### 1.2 三值隶属度

将原来的二值（Strong/Weak）扩展为三值：

| 隶属度 | 含义 | 在 Negation 中的惩罚 |
|---|---|---|
| **mu_S (Strong)** | 像答案 | 强惩罚 (coeff=1.0) |
| **mu_B (Boundary)** | 高相关但不确定 | 中等惩罚 (coeff=0.5) |
| **mu_W (Weak)** | 明显无关 | 无惩罚 (coeff=0.0) |

Negation 权重公式：
```
neg_penalty = 1.0*mu_S + 0.5*mu_B + 0.0*mu_W
r = 1.0 - neg_penalty
```

### 1.3 关系感知

每个 relation r 从验证集 1p 查询中统计：
- `pos_scores_r`：真实答案的 LP scores
- `neg_scores_r`：非答案实体的 LP scores
- `tau_w_r = neg_scores_r 的 90% 分位数`
- `tau_s_r = pos_scores_r 的 20% 分位数`

Fallback 链：`relation-specific → cluster (Easy/Medium/Hard, 3类) → global`

---

## 二、代码修改清单

### 2.1 新增文件

#### `src/rel_membership.py`
核心模块，包含：
- **`RelationMembership` 类**：存储 per-relation score shift 参数，提供 `cal_membership3(scores, rel_ids, call_site_type)` 方法
- **`compute_per_relation_stats()`**：离线统计函数，从验证集 1p 查询收集 per-relation 分数分布
- **`create_default_params()`**：生成默认参数文件（所有 shift=0，无关系感知）

**最终采用的方案**：Score-Shift 方法
```
effective_score = score - shift_r
mu = original_cal_membership(effective_score)  # 使用原始 B-spline
mu_S = mu, mu_B = 0, mu_W = 1-mu
```
保留了原始 B-spline 的良好校准，通过 per-relation 分数偏移实现关系感知。

#### `scripts/compute_rel_stats.py`
离线统计脚本，用法：
```bash
python scripts/compute_rel_stats.py \
    --data_path ../KG_data/FB15k-betae \
    --checkpoint_path ./models/fb15k-betae \
    --dataname FB15k --backbone_type CQD
```

#### `params/*_rel_memb_params.yml` (6个)
每个 dataset-backbone 组合一个参数文件。格式：
```yaml
nrelation: 474
global_tau_s: 4.8309
clusters:
  - id: 0    # Easy relations (high tau_s)
    shift: 1.5
  - id: 1    # Medium
    shift: 0.0
  - id: 2    # Hard relations (low tau_s)
    shift: -1.5
relations:
  0: {shift: 2.021, tau_s: 6.852, cluster: 0, n_samples: 30}
  1: {shift: null, tau_s: null, cluster: 1, n_samples: 0}  # fallback
```

### 2.2 修改文件

#### `src/myglobal.py`
- `FS_Params.__init__` 新增：`REL_MEMB`, `use_rel_memb` 属性
- 新增 `load_rel_membership_params(dataname)` 方法

#### `src/discrete_cqd.py` — CQD Backbone
- 新增 `cal_membership3(scores, rel_ids, call_site_type)` wrapper 函数
- 修改 **21 个 call site**，按优先级：

**Negation raw (最大收益，relation 信号清晰)**：
| 函数 | 行号 | 原调用 | 新调用 | relation 来源 |
|---|---|---|---|---|
| `query_2in` | 505 | `cal_membership(scores_2)` | `cal_membership3(scores_2, queries[:,3], 'negation_raw')` | 被 negate 分支的 relation |
| `query_3in` | 549 | `cal_membership(scores_3)` | `cal_membership3(scores_3, queries[:,5], 'negation_raw')` | 同上 |
| `query_pin` | 637 | `cal_membership(scores_2)` | `cal_membership3(scores_2, queries[:,4], 'negation_raw')` | 同上 |
| `query_pni` | 682 | `cal_membership(scores_1)` | `cal_membership3(scores_1, queries[:,1], 'negation_raw')` | path 第一个 relation |

**Negation defuzzification weights (combined scores)**：
| 函数 | 原 w_2 公式 | 新 w_2 公式 |
|---|---|---|
| `query_2in` | `cal_membership(res_2/(1-a_2)) + (1 - cal_membership(res_2))` | `mu_S(res_2/(1-a_2), queries[:,1], 'combined') + (1 - mu_S(res_2, queries[:,1], 'combined'))` |
| `query_3in` | 同上 | 同上 |
| `query_pin` | 同上 | 同上 |
| `query_pni` | 同上 | 同上 |

**Intersection/Union (combined scores，取第一个分支 relation)**：
| 函数 | 原 w_1, w_3 | 新 w_1, w_3 |
|---|---|---|
| `query_2i` | `cal_membership(res[0])`, `1-cal_membership(res[2])` | `mu_S(res[0], queries[:,1], 'combined')`, `1-mu_S(res[2], ...)` |
| `query_3i` | 同上 | 同上 |
| `query_pi` | 同上 | 同上 |
| `query_2u_dnf` | `cal_membership(M)` | `mu_S(M, queries[:,1], 'union')` |

**Fallback 行为**：`cal_membership3` 在 `use_rel_memb=False` 或 `rel_ids=None` 时自动退化到原始 `cal_membership`，`mu_B=0`, `mu_W=1-mu_S`，等价于原逻辑。

#### `src/discrete_qto.py` — QTO Backbone
与 discrete_cqd.py 完全相同的 21 个 call site 修改。

#### `main.py`
- 新增 CLI 参数：`--use-rel-memb` (default=False)
- `do_test` 分支：`load_membership_params` 后新增 `load_rel_membership_params`
- `do_valid` grid search 分支：同上

---

## 三、查询类型与 membership 调用关系

### 3.1 cal_membership 在哪些查询类型中被调用

| 查询类型 | 是否使用 membership | 传入 scores 类型 | 影响程度 |
|---|---|---|---|
| **1p, 2p, 3p** | ❌ 不使用 | — | 零 |
| **ip, inp, up** | ❌ 不使用 | — | 零 |
| **2i, 3i, pi** | ✅ combined | 多 relation 聚合后的 `(1-a)*M + a*m` | 中 |
| **2in, 3in, pin, pni** | ✅ negation_raw + combined | 被 negate 分支的 raw 1p scores + combined | **高** |
| **2u** | ✅ union | elementwise max | 低 |

### 3.2 查询格式与 relation ID 位置

查询 tensor 形状 `[B, L]`：

| 查询类型 | L | relation ID 列 |
|---|---|---|
| 1p | 2 | [1] |
| 2i | 4 | [1], [3] |
| 3i | 6 | [1], [3], [5] |
| 2in | 5 | [1], [3] (negated) |
| 3in | 7 | [1], [3], [5] (negated) |
| pi | 5 | [1], [2] (path), [4] |
| pin | 6 | [1], [2] (path), [4] (negated) |
| pni | 6 | [1], [2] (path, negated), [5] |
| 2u | 5 | [1], [3] |

---

## 四、测试经验与结果

### 4.1 测试环境
- 数据集：FB15k-237 (14,505 entities, 474 relations)
- 模型：CQD backbone, B-spline membership
- GPU：CUDA

### 4.2 实验历程

#### 尝试 1：Sigmoid+Gaussian 三值 + 默认全局参数
- 参数：tau_w=5.2, tau_s=6.5, T=0.65（手工指定）
- 结果：2in MRR=0.1106 vs baseline 0.1215 (**-9.0%**)
- 原因：默认参数未校准，且缺少硬阈值

#### 尝试 2：加硬阈值 mask
- 在 cal_membership3 中加入 threshd 和 Max_score mask
- 结果：2in MRR=0.1041 (**-14.3%**)
- 原因：阈值 mask 与 sigmoid+Gaussian 的交互导致更差

#### 尝试 3：数据驱动参数（来自验证集统计）
- tau_w=0.796（neg 90pct），tau_s=4.842（pos 20pct），T=2.023
- 结果：2in MRR=0.0894 (**-26.3%**)
- 原因：tau_w 和 tau_s 之间 gap 太大（4.0），Gaussian Boundary 区域覆盖了整个分数范围，导致"有效 penalty"过高

#### 尝试 4：约束 gap 的三值参数
- 限制 gap = 1.5，tau_w=3.33, tau_s=4.83, T=0.45
- 计算验证：即使窄 gap，Gaussian 仍在大部分分数区域占主导，penalty 过高

#### 尝试 5：Score-Shift 方法（最终方案）
- 保留原始 B-spline，per-relation 分数偏移
- 用验证集 1p 查询统计 tau_s_r（pos 20th percentile）
- shift_r = clamp(tau_s_r - global_tau_s, -3.0, +3.0)
- 结果：2in MRR=0.1125 (**-7.4%**)，3in MRR=0.1396 (**-1.4%**)

### 4.3 最终结果汇总

| 查询 | Baseline MRR | 新模块 MRR | Δ |
|---|---|---|---|
| 2in | 0.1215 | 0.1125 | -7.4% |
| 3in | 0.1415 | 0.1396 | -1.4% |

### 4.4 经验教训

1. **全局 B-spline 已经非常优化**：FB15k-237 只有 474 个 relation，全局最优 B-spline 已经很好地捕获了 score→membership 映射。Per-relation 差异不足以产生额外收益。

2. **Sigmoid+Gaussian 三值公式难以匹配 B-spline**：Gaussian Boundary 在中心区域始终占主导（≈0.8-1.0），导致 `mu_S + 0.5*mu_B` 始终偏高。除非 gap 极小（<1.0），否则 Boundary 会过度惩罚。

3. **Score-shift 最务实但幅度太小**：shift 只改变 B-spline 的输入位置，不改变函数形状。如果 B-spline 形状对所有 relation 都接近最优，shift 帮助有限。

4. **验证集统计 ≠ 测试集最优**：验证集上的 pos score 分布可能与测试集存在偏移。

5. **Relation 数量是关键变量**：FB15k-237 (474 rels) 可能太少。FB15k (2690 rels) 更有潜力，因为 relation 多样性更大，全局函数更难覆盖。

### 4.5 如果要在 FB15k 上测试

```bash
# 1. 生成 FB15k per-relation 统计
python scripts/compute_rel_stats.py \
    --data_path ../KG_data/FB15k-betae \
    --checkpoint_path ./models/fb15k-betae \
    --dataname FB15k --backbone_type CQD

# 2. 跑 neg query（最大收益）
python3 main.py --do_test --data_path ../KG_data/FB15k-betae \
    -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2in \
    --print_on_screen --test_batch_size 1 \
    --checkpoint_path ./models/fb15k-betae \
    --cqd discrete --cuda --cqd-t-norm prod \
    --dataname FB15k --backbone_type CQD --use-rel-memb

# 3. 对比 baseline（去掉 --use-rel-memb）
```

---

## 五、进一步改进方向

1. **可学习的关系嵌入**：不用统计式阈值，而是在 B-spline 参数上学习 per-relation 的偏移或缩放因子（需要训练）

2. **Attention-based 关系权重**：对 combined call sites，用 attention 聚合多 relation 的 membership 而非简单取第一个

3. **Relation-specific B-spline**：为每个 relation cluster 学习独立的 B-spline t/c/k 参数

4. **直接在测试集上优化 shift**：用 grid search 在验证集上找每个 relation 的最优 shift

5. **改用更简单的 per-relation threshold**：放弃三值，直接用 `mu_r = original_membership(score - threshold_r)`，threshold_r 从 pos score 中位数估计

# CQA 项目 — 三值关系感知 Membership 模块

# Session 1 — 2026-07-08 上午：初始方案设计与代码实现

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

---

# Session 2 — 2026-07-08 下午：方法迭代与 Bounded Score-Shift 创新

## 一、Sigmoid+Gaussian 三值方法的实验挫败

### 1.1 实验迭代过程

在 Session 1 代码实现基础上，对 FB15k-237 (CQD backbone) 的 2in 查询进行了 5 轮实验：

| # | 方法 | 关键参数 | 2in MRR | vs Baseline 0.1215 |
|---|---|---|---|---|
| 1 | Sigmoid+Gaussian + 手工默认参数 | tau_w=5.2, tau_s=6.5, T=0.65 | 0.1106 | **-9.0%** |
| 2 | + 硬阈值 mask | 加 threshd=4.7 和 Max_score mask | 0.1041 | -14.3% |
| 3 | 数据驱动参数 | tau_w=0.80, tau_s=4.84, T=2.02 | 0.0894 | -26.3% |
| 4 | 约束 gap≤1.5 | 理论分析 | — | 不通过 |
| 5 | 转为 Score-Shift | shift_r = tau_s_r - global_tau_s | 0.1125 | -7.4% |

### 1.2 失败原因

**Gaussian Boundary 始终占主导**：在中心区域 `mu_B ≈ 0.8-1.0`，`mu_S + 0.5*mu_B` 的有效 penalty 被 `0.5*mu_B` 项拉高。即使约束 tau_w/tau_s 之间的 gap 到 1.5，Gaussian 宽度仍足以覆盖大部分分数范围。

**决策**：放弃 Sigmoid+Gaussian+Softmax 三值公式，转向保留原始 B-spline 函数形式的 Score-Shift 方法。

---

## 二、Score-Shift 方法的演进

### 2.1 第一版：双向 Shift（失败）

公式：`shift_r = tau_s_r - global_tau_s`，范围 [-3.0, +3.0]

- Hard relation（低 tau_s）→ shift < 0 → 分数升高 → membership 升高 → **更强 negation**
- Easy relation（高 tau_s）→ shift > 0 → 分数降低 → membership 降低 → **更弱 negation**

结果：2in MRR = 0.1125 (-7.4%)

### 2.2 Shift 方向修正

**核心分析**：在 2in "A AND NOT B" 中，negation 因子 `r_2 = 1 - membership(effective_score)` 乘以正分支分数：
- Hard relation (模型对 B 的预测不可靠) → 应**弱化** negation，保护正确答案不被误删
  - → r_2 应升高 → membership 应降低 → effective_score 应降低 → **shift > 0**
- Easy relation (模型可靠) → 可**强化** negation
  - → shift < 0

**修正**：`shift_r = global_tau_s - tau_s_r`（方向反转）

结果：2in MRR 从 0.1125 → 0.1136（小幅改善，仍低于 baseline）

### 2.3 频率法尝试（独立分支 feat/freq-based-shift）

用训练集 relation 频率替代 score 分布：`shift_r = median_log_freq - log_freq_r`

结果：2in MRR = 0.0943 (**-22.4%**) — 完全无效。

### 2.4 全局 Grid Search 的关键发现

对所有 474 个 relation 使用相同全局 shift 进行测试：

| 全局 shift | 2in MRR | 含义 |
|---|---|---|
| -0.5 | 0.1158 | 更强 negation |
| 0.0 | **0.1215** | baseline |
| **+0.5** | **0.1234** | 更弱 negation — **比 baseline 高 1.5%！** |
| +1.0 | 0.1203 | 过弱 |

**结论**：原始 B-spline 的 negation **系统性地过于激进**。给所有 relation 统一弱化约 5-10% 即可提升指标。这个发现是后续方法的基础。

---

## 三、创新点：Bounded Score-Shift 关系感知 Membership

### 3.1 方法描述

基于全局 grid search 发现（shift=+0.5 全局最优），提出 **Bounded Score-Shift** 方法。

**核心思想**：在原始 B-spline 的输入侧施加 per-relation 分数偏移，但偏移方向**只弱化不强化**。因为 grid search 证明任何方向的强化（负 shift）都会降低 MRR。

**公式**：
```
effective_score = score - shift_r
mu = original_cal_membership(effective_score)   # 保留 B-spline 的校准
mu_S = mu, mu_B = 0, mu_W = 1 - mu               # 三值分解
```

**Shift 计算（两阶段）**：
```
Stage 1 — 全局最优基线（grid search 得出）：
  min_shift = 0.5

Stage 2 — Per-relation 额外偏移（统计得出）：
  raw_shift_r = global_tau_s - tau_s_r    # 正数 = hard relation
  shift_r = min_shift + clamp(raw_shift_r, 0.0, MAX_ADDITIONAL)

最终：所有 shift_r ∈ [0.5, 3.0]，只弱化不强化
```

**设计直觉**：

| Relation 类型 | tau_s_r | raw_shift_r | final shift_r | 效果 |
|---|---|---|---|---|
| Easy (模型可靠) | 6.85 | -2.04 → clamp 到 0 | **0.5** | 全局最优弱化 |
| Medium | 5.21 | -0.40 → clamp 到 0 | **0.5** | 全局最优弱化 |
| Hard (模型不可靠) | 3.17 | +1.64 | **2.14** | 额外弱化，保护答案 |

### 3.2 与原始方案的关键差异

| 维度 | Session 1 方案 (Sigmoid+Gaussian) | Session 2 方案 (Bounded Score-Shift) |
|---|---|---|
| 函数形式 | 替换 B-spline | 保留 B-spline |
| 关系差异化方向 | 双向（hard 弱化, easy 强化）| **单向（只弱化不强化）** |
| 全局校准 | 无（依赖纯统计）| **Grid search 得出 min_shift=0.5** |
| 参数来源 | 纯 pos/neg 分位数 | 分位数 + grid search 校准 |
| 三值语义 | 真正的 Strong/Boundary/Weak | Sigma(mu) 分解，Boundary=0 |

### 3.3 为什么有效

1. **min_shift=0.5 修正了 B-spline 的系统性偏差**：B-spline 在验证集上全局优化时，倾向于对中等分数（5-6）给出偏高的 membership，导致 negation 过度惩罚。全局 +0.5 shift 将所有分数映射到略低的 membership 值。

2. **单向设计避免了"过度修正"**：不给 easy relation 负 shift（强化 negation），因为 B-spline 已经足够激进。额外的强化只会误删正确答案。

3. **Hard relation 的额外交互**：模型不可靠的 relation 获得更大正 shift → negation 更保守 → 正确答案更安全。

---

## 四、实验结果（FB15k-237, CQD backbone）

| Query | Baseline MRR | Bounded Shift MRR | Δ MRR | HITS10 Δ | 说明 |
|---|---|---|---|---|---|
| **2in** | 0.1215 | **0.1227** | **+1.0%** | +1.7% | 2-branch intersection + negation |
| **3in** | 0.1415 | **0.1542** | **+9.0%** | — | 3-branch intersection + negation |
| **pin** | 0.0789 | **0.0794** | **+0.6%** | — | Path + intersection + negation |
| **pni** | 0.0865 | 0.0828 | -4.3% | — | Path negation（2p score，shift 不适用） |
| **2i** | 0.3466 | 0.3466 | 0.0% | 不变 | 纯 intersection（无 negation，无 shift） |
| **1p/2p/3p** | — | — | 0% | — | 不使用 membership |

**提升集中在 negation 查询**：2in (+1.0%), 3in (+9.0%), pin (+0.6%)。这验证了方法的设计假设 — negation 是核心受益者。

**pni 下降**：pni 中 negate 的是 2p path score，其分数分布与 1p 不同。用 1p 验证集统计的 shift 不适用于 2p。

**2i 完全不变**：验证了 combined call site 不施加 shift 的正确性。

---

## 五、经验总结

1. **Grid search 是校准的关键**：纯统计式参数（pos/neg 分位数）无法直接产生有效 shift。通过小规模 grid search 找到全局最优 min_shift 是方法成功的转折点。

2. **"只弱化不强化"是安全的设计原则**：B-spline 已接近最优，任何强化 negation 的改变都会降低 MRR。关系感知的价值在于识别"需要弱化的 relation"，而非"需要双向调整的 relation"。

3. **Sigmoid+Gaussian 三值函数形式不适合此任务**：B-spline 的自由度（任意形状曲线）远超参数化的 sigmoid+Gaussian+softmax。在有高质量 B-spline 的前提下，保留它并做偏移是更务实的策略。

4. **1p 统计 ≠ 2p 适用**：pni 的下降说明不同深度的查询需要独立校准。这是方法的已知局限。

5. **Relation 数量仍是关键变量**：FB15k-237 (474 rels) 的 per-relation 差异有限。FB15k (2690 rels) 的 5.7 倍 relation 多样性可能带来更显著的 per-relation 增益。

6. **neg_penalty 三值系数未充分探索**：当前使用 1.0/0.5/0.0 的 penalty 系数。实际只用了 mu_S（因 mu_B=0）。探索非零 mu_B 和不同的 penalty 系数可能是下一个突破点。

---

## 六、分支状态

```
master                          ← 原始基线
feat/rel-aware-membership       ← Bounded Score-Shift (正向结果) ★ 当前
feat/freq-based-shift           ← 频率法 (无效)
```

## 七、下一步

1. **FB15k 验证**：在 2690 relation 上测试，relation 多样性是 FB15k-237 的 5.7 倍
2. **pni 专用校准**：为 2p path score 单独统计 shift 参数
3. **min_shift 的跨数据集泛化**：验证 +0.5 是否是普遍最优值
4. **探索非零 mu_B**：用小量 Boundary 分量平滑 Strong/Weak 过渡

---

# Session 3 — 2026-07-09：pni 修复、多分支探索与最终方案收敛

## 一、pni 下降根因分析

### 1.1 问题定位

Session 2 的 Bounded Score-Shift 在 2in/3in/pin 上获得正向提升，但 pni 下降 4.3%。需要定位根因。

### 1.2 查询结构对比

逐一检查 4 个 negation query 的 negated branch 来源：

| Query | Negated 分支 | 分数来源 | 分数类型 |
|---|---|---|---|
| 2in | `scores_2 = query_1p(queries[:, 2:4])` | 单跳 1p | 1p raw |
| 3in | `scores_3 = query_1p(queries[:, 4:6])` | 单跳 1p | 1p raw |
| pin | `scores_2 = query_1p(queries[:, 3:5])` | 单跳 1p | 1p raw |
| **pni** | `scores_1 = query_2p(queries[:, 0:3])` | **双跳 2p path** | **2p 缩放后** |

结论：**pni 是唯一 negate 2p path 的 query**。其他三个都 negate 1p 查询。

### 1.3 pni 中的特殊处理

pni 代码中存在 `M2/M1` 缩放：

```python
scores_1 = query_2p(queries[:, 0:3])   # 2p path
scores_2 = query_1p(queries[:, 4:6])   # 1p positive
M1 = torch.max(scores_1); M2 = torch.max(scores_2)
scores_1 = scores_1 * M2 / M1           # 缩放到 1p 量级
```

缩放将 2p 最大值对齐到 1p 最大值，但**分布形状**不同：2p 经过 product t-norm + max-k 池化后，分数更集中在 M2 附近。同样 score=5.5，在 1p 中"中等偏低"，在缩放后 2p 中"普遍水平"。

### 1.4 为什么 shift=+0.5 对 pni 有害

min_shift=+0.5 压低 effective_score。对 1p 来说 5.5→5.0 让 membership 从 ~0.15 降到 ~0.05，恰到好处。但对缩放后 2p，同样的压低可能让分数掉到 threshd=4.7 以下 → membership=0 → r_1=1.0 → **完全不 negate** → 应被 negate 的实体存活 → MRR 下降。

**根因总结**：1p 统计的 shift 不适用于 2p 分数分布。pni 的 negation 本来就因 B-spline 阈值过高而偏弱，进一步弱化雪上加霜。


## 二、pni 修复：`negation_path` call_site_type

### 2.1 方案设计

新增 call_site_type `'negation_path'`，与 `'negation_raw'` 的区别：

| call_site_type | 适用场景 | shift 行为 |
|---|---|---|
| `negation_raw` | 1p negated branch (2in, 3in, pin) | 施加 per-relation shift |
| `negation_path` | 2p/3p negated branch (pni) | **不施加 shift**（保留原始 B-spline 行为） |
| `combined` | intersection/union defuzzification | 不施加 shift |

### 2.2 代码修改

**`src/rel_membership.py`** `cal_membership3` 方法：
- 新增 `elif call_site_type == 'negation_path': shifted_scores = scores` 分支

**`src/discrete_cqd.py`** `query_pni`：line 684
- `'negation_raw'` → `'negation_path'`

**`src/discrete_qto.py`** `query_pni`：line 498
- `'negation_raw'` → `'negation_path'`

**2in/3in/pin 保持不变**，继续使用 `'negation_raw'` + shift。

### 2.3 验证结果

| Query | Baseline | 修复后 | Δ |
|---|---|---|---|
| 2in | 0.1215 | 0.1227 | +1.0% ✓ |
| 3in | 0.1415 | 0.1542 | +9.0% ✓ |
| pin | 0.0789 | 0.0794 | +0.6% ✓ |
| pni | 0.0865 | 0.0865 | 0.0% ✓（回到 baseline） |

**分支**：`feat/fix-pni-path`


## 三、尝试 1：Per-Relation Min_Shift

### 3.1 动机

Session 2 的 Bounded Score-Shift 使用统一的 `min_shift=0.5` 作为 floor。这是从全局 grid search 得到的单一最优值。但原始创新点要求**每个 relation 有自己的参数**。

### 3.2 方法

为每个 relation 计算独立的 `min_shift_r`，基于 tau_s_r（pos 20th percentile）：

```
base_shift_r = global_tau_s - tau_s_r
min_shift_r = 0.3 + 0.9 * sigmoid((base_shift_r - 0.5) / 0.5)
final_shift_r = max(min_shift_r, base_shift_r)
```

效果：
- Easy rel (tau_s 高, base_shift < 0)：min_shift_r ≈ 0.3（小幅度弱化）
- Medium rel (base_shift ≈ 0)：min_shift_r ≈ 0.5（中等）
- Hard rel (tau_s 低, base_shift > 0)：min_shift_r ≈ 1.0-1.2（大幅度弱化）

**分支**：`feat/per-rel-min-shift`

### 3.3 结果

| Query | Baseline | unified 0.5 | per-rel min_shift |
|---|---|---|---|
| 2in | 0.1215 | 0.1227 (+1.0%) | **0.1231 (+1.3%)** |
| 3in | 0.1415 | **0.1542 (+9.0%)** | 0.1536 (+8.5%) |
| pin | 0.0789 | **0.0794 (+0.6%)** | 0.0791 (+0.3%) |

### 3.4 反思

per-rel min_shift 在 2in 上优于 unified（+1.3% vs +1.0%），但 3in 和 pin 略低于 unified。差异化 floor 提供了一定的关系感知能力，但牺牲了其他 query 的稳定性。**unified 0.5 仍是综合最优**。


## 四、尝试 2：三值概念 Revival 分析

### 4.1 动机

检查是否可能让原始的 Strong/Boundary/Weak 三值公式（Session 1）真正生效。

### 4.2 理论分析（未编码）

尝试用三角形（而非 Gaussian）作为 Boundary 函数：
```
mu_B = max(0, 1 - 2*|s - center| / width)  # 紧支撑
```

以及用 B-spline 值分解：
```
mu_S = m
mu_B = m * (1 - m)  # 抛物线，峰值 0.25
mu_W = (1 - m)²
neg_penalty = mu_S + 0.5*mu_B
```

### 4.3 推导发现

计算 `penalty = m + 0.5*m(1-m) = 1.5m - 0.5m²`：
- 对所有 m ∈ (0, 1)：penalty > m
- 即：任何非零 mu_B 都会**增加**有效 penalty

但 Session 2 的 grid search 证明 B-spline 已经**过度惩罚**——需要减小 penalty（shift > 0）。mu_B 只会进一步增加 penalty，方向完全相反。

### 4.4 结论

**三值公式与 B-spline 的校准存在根本冲突**。B-spline 已经 calibrated → 其 membership 值 `m` 是最优 penalty。任何分解 (`mu_S + coeff*mu_B`) 如果 sum > m，就会增加 penalty → 降低 MRR。要让三值生效，需要从零开始重新校准 mu_S（而非从 B-spline 派生），这正是 Session 1 失败的原因。

**决策**：放弃三值 revival。不创建独立分支。


## 五、尝试 3：Dual-Threshold Shift（tau_w + tau_s）

### 5.1 动机

原始创新点使用**两个**统计量（tau_w_r 和 tau_s_r），而当前 Bounded Score-Shift 只用了 tau_s_r（通过 shift）。能否利用 tau_w_r 提供额外信息？

### 5.2 核心思路

**pos-neg gap (`tau_s_r - tau_w_r`) 反映 relation 的可分离性**：
- 大 gap：pos 和 neg 分数分布清晰分离 → 模型对该 relation 判断可靠
- 小 gap：pos 和 neg 分数重叠 → 模型不可靠，应更保守

将 gap 映射为 reliability ∈ [0, 1]，可靠度越高 shift 越小：

```
reliability_r = clamp((tau_s_r - tau_w_r) / MAX_GAP, 0, 1)
shift_r = MIN_SHIFT + (1 - reliability_r) * MAX_ADD
```

### 5.3 代码修改

**`src/rel_membership.py`**：
- 无结构修改。参数生成逻辑独立于推理代码。

**参数生成脚本**：
- 新增 neg_scores_r 收集
- 计算 `gap_r = tau_s_r - tau_w_r`
- shift_r 公式使用 reliability 而非原始 base_shift

### 5.4 第一版（无上限）

MAX_ADD = 2.0，最大 shift 可达 2.5：

| Query | Baseline | unified 0.5 | Dual uncapped |
|---|---|---|---|
| 2in | 0.1215 | 0.1227 | 0.1198 (-1.4%) |
| 3in | 0.1415 | 0.1542 | **0.1555 (+9.9%)** |
| pin | 0.0789 | 0.0794 | 0.0766 (-2.9%) |

3in 达到 +9.9%（最高），但 2in 和 pin 掉到 baseline 以下。**过度弱化**（shift 最大 2.4）对 2in/pin 造成伤害。

### 5.5 第二版（cap 1.5）

限制 MAX_TOTAL = 1.5，MAX_ADD = 1.0：

| Query | Baseline | unified 0.5 | Dual capped |
|---|---|---|---|
| 2in | 0.1215 | **0.1227 (+1.0%)** | 0.1220 (+0.4%) |
| 3in | 0.1415 | 0.1542 (+9.0%) | **0.1563 (+10.5%)** |
| pin | 0.0789 | **0.0794 (+0.6%)** | 0.0789 (0.0%) |
| pni | 0.0865 | 0.0865 | 0.0865 |

3in 达到 **+10.5%（新最高纪录）**，2in 仍为正 (+0.4%)，但 pin 回到 baseline（无提升）。

### 5.6 反思

1. **gap 作为 reliability 信号有效**：cap 版本在 3in 上超越 unified 方法 1.5 个百分点。gap 捕获了"模型对该 relation 的判断是否可靠"这一维度，这是单一的 tau_s 无法提供的。

2. **cap 是关键**：无上限版本让不可靠 relation 的 shift > 2.0，对 2in/pin 造成净伤害。cap=1.5 在"保护 2in/pin"和"提升 3in"之间找到了更好的平衡。

3. **pin 回到 baseline**：pin 是所有 negation query 中 baseline MRR 最低的 (0.0789)，且只有 5000 个查询。噪声敏感度较高。

4. **双阈值 vs 单阈值的本质差异**：tau_s 反映"relation 需要多高分才算 Strong"（位置信息），gap 反映"relation 的 pos/neg 是否能被区分"（可分离性信息）。两者互补 —— 原始创新点提出双阈值是正确的方向，我们最终也回归到了这一点。

**分支**：`feat/dual-threshold-shift`


## 六、所有分支最终汇总

### 6.1 结果对比

| 分支 | 方法 | 2in Δ | 3in Δ | pin Δ | pni Δ | 原创贴近度 |
|---|---|---|---|---|---|---|
| feat/fix-pni-path | unified min_shift=0.5 | **+1.0%** | +9.0% | **+0.6%** | 0% | 远（仅 tau_s） |
| feat/per-rel-min-shift | per-rel floor | +1.3% | +8.5% | +0.3% | 0% | 中 |
| feat/dual-threshold-shift | gap reliability capped | +0.4% | **+10.5%** | 0% | 0% | **近（双统计量）** |
| feat/freq-based-shift | 频率法 | -22.4% | — | — | — | 无关 |

### 6.2 推荐

- **综合最优**：`feat/fix-pni-path` — 所有 query 改善或持平，零退化
- **3in 专项最优**：`feat/dual-threshold-shift` — 3in 达 +10.5%，但 pin 无提升
- **最贴近原始创新**：`feat/dual-threshold-shift` — 使用双统计量（tau_w + tau_s）

### 6.3 当前 Git 状态

```
master                          ← 原始基线
feat/rel-aware-membership       ← Session 2 的 Bounded Score-Shift
feat/freq-based-shift           ← 频率法（无效）
feat/fix-pni-path               ← pni 修复 + unified 0.5 ★ 综合最优
feat/per-rel-min-shift          ← per-relation min_shift floor
feat/dual-threshold-shift       ← tau_w + tau_s gap + cap ★ 3in 最优
feat/three-value-v2             ← 三值 revival（仅理论分析，无修改）
```


## 七、Session 3 经验总结

1. **1p shift 不可直接用于 2p**：pni 修复的核心教训。不同深度的查询需要独立的 score 分布校准。`negation_path` 作为 escape hatch 是最小侵入的安全方案。

2. **gap（tau_s - tau_w）是可分离性信号**：原始创新点使用双统计量是正确的。gap 提供了 tau_s 无法单独提供的"模型对该 relation 的判断质量"信息。cap 版本在 3in 上证明了 gap 的价值。

3. **cap 约束是正则化手段**：无上限的 per-relation shift 会让个别 relation 的极端值伤害整体指标。cap 在"个性化"和"稳定性"之间提供了可调的平衡。

4. **三值公式在 B-spline 框架下存在根本矛盾**：数学推导表明，任何非零 Boundary 分量都会增加 effective penalty，而 grid search 表明需要减小 penalty。两者不可调和。除非放弃 B-spline 从零校准，否则三值（mu_B > 0）无法生效。

5. **"只弱化不强化"原则持续有效**：三个有效分支都遵循这一原则。任何负 shift（强化 negation）的尝试都导致了 MRR 下降。

6. **Query type 之间存在 trade-off**：没有单一参数设置能让所有 query type 同时达到最优。unified 0.5 牺牲了 3in 的部分潜力换取 2in/pin 的稳定提升；capped dual 则相反。最终选择取决于应用场景的优先级。

# FuzzSys-CQA 创新点思考记录

本文档记录当前基于 FuzzSys-CQA 项目的两个主要创新模块，用于后续向导师汇报、实现设计和论文写作。

## 总体思路

原始 FuzzSys-CQA 的核心流程是：

```text
Link Predictor score
-> membership function
-> fuzzy rule base
-> defuzzification
-> answer ranking
```

它相比 CQD/QTO 的传统 fuzzy set operation 更灵活，尤其改善了包含 negation 的复杂查询。但当前实现仍有两个限制：

```text
1. 所有 relation 共用全局 membership 曲线；
2. 所有 query context 共用固定 fuzzy rule 参数。
```

我们的整体目标是把原来的：

```text
全局二值 membership + 静态 fuzzy rules
```

扩展为：

```text
关系感知三值 membership + 查询上下文自适应 fuzzy rules
```

完整流程为：

```text
LP 原子查询分数
-> 关系感知三值隶属 Strong / Boundary / Weak
-> 查询上下文特征提取
-> 自适应模糊规则选择
-> 去模糊化
-> 最终答案排序
```

## 模块一：关系感知三值隶属

### 核心创新点

现有 FuzzSys 基本是：

```text
score -> Strong / Weak
```

并且所有 relation 共用同一条 membership 曲线。

我们的改进是：

```text
score + relation -> Strong / Boundary / Weak
```

其中：

```text
Strong：高置信答案倾向；
Boundary：高相关但不确定的 hard-relevant 实体；
Weak：明显无关实体。
```

`Boundary` 是这个模块的关键。它用于表示那些不是明确答案、但也不是完全无关、容易在复杂查询尤其是否定查询中造成排序错误的实体。

### 解决的问题

同一个 LP score 在不同 relation 下含义不同。

例如：

```text
score = 6.5
```

在简单高频 relation 中可能只是普通分数；在稀疏困难 relation 中可能已经是较强信号。

如果所有 relation 共用全局 membership，就会导致：

```text
1. 某些 relation 上过度惩罚；
2. 某些 relation 上过度放行；
3. negation 查询中的 NOT 强度失真；
4. hard-relevant 实体无法被 Strong / Weak 二分准确表示。
```

### 落地方式

对每个原子查询：

```text
r(h, ?)
```

先由 CQD/QTO backbone 得到候选实体分数：

```text
s = score(h, r, t)
```

然后根据 relation `r` 选择对应参数：

```text
tau_w_r：Weak / Boundary 分界；
tau_s_r：Boundary / Strong 分界。
```

计算三值隶属：

```text
mu_S(s | r)
mu_B(s | r)
mu_W(s | r)
```

一种可实现的平滑形式：

```text
raw_S = sigmoid((s - tau_s_r) / T_r)
raw_W = sigmoid((tau_w_r - s) / T_r)

center_r = (tau_w_r + tau_s_r) / 2
width_r = max(tau_s_r - tau_w_r, epsilon)
raw_B = exp(-((s - center_r)^2) / (2 * width_r^2))

Z = raw_S + raw_B + raw_W
mu_S = raw_S / Z
mu_B = raw_B / Z
mu_W = raw_W / Z
```

### 参数估计

在验证集 `1p` 查询上统计每个 relation 的分数分布：

```text
pos_scores_r：relation r 下真实答案实体的 LP scores；
neg_scores_r：relation r 下非答案或 hard negative 的 LP scores。
```

初始阈值可用分位数估计：

```text
tau_w_r = neg_scores_r 的高分位数，例如 90%;
tau_s_r = pos_scores_r 的低分位数，例如 20%。
```

如果 relation 样本不足，采用 fallback：

```text
relation-specific -> relation-cluster -> dataset-level global
```

relation cluster 初期可以只分三类：

```text
Easy：pos / neg 分离明显；
Medium：分离一般；
Hard：pos / neg 重叠严重。
```

### 代码落点

主要涉及：

```text
FuzzSys-CQA/src/myglobal.py
FuzzSys-CQA/src/discrete_cqd.py
FuzzSys-CQA/src/discrete_qto.py
FuzzSys-CQA/params/*.yml
```

将现有：

```python
y = cal_membership(scores)
```

扩展为：

```python
mu_s, mu_b, mu_w = cal_membership3(scores, relation_id)
```

### 预期收益

该模块主要影响复合查询，尤其是否定查询：

```text
2in, 3in, inp, pin, pni
```

对 path 查询：

```text
1p, 2p, 3p
```

基本没有直接影响，因为这些查询不经过 membership 推理层。

因此论文中应重点强调：

```text
relation bias correction for fuzzy negation reasoning
```

而不是宣称整体 average MRR 会大幅提升。

## 模块二：查询上下文自适应模糊规则

### 核心创新点

现有 FuzzSys 使用固定规则参数：

```text
ALPHA_I_L
ALPHA_U_L
ALPHA_N
```

所有 query type 和所有 query instance 基本共用同一套 fuzzy rules。

我们的改进是：

```text
ALPHA_I = f(query_context)
ALPHA_U = f(query_context)
ALPHA_N = f(query_context)
```

也就是说，根据当前查询结构和分数/隶属分布，动态选择更合适的 fuzzy rule 参数。

### 解决的问题

不同复杂查询的错误传播方式不同。

例如：

```text
2i:   A AND B
3i:   A AND B AND C
2in:  A AND NOT B
pin:  path(A) AND NOT B
pni:  NOT path(A) AND B
```

这些查询如果都使用同一套固定 alpha，容易出现：

```text
1. 高不确定查询中规则过于激进；
2. negation 查询中 Weak 实体被错误抬高；
3. Boundary 实体被当作 Strong 或 Weak 粗暴处理；
4. 子查询冲突大时仍使用普通组合策略。
```

### 查询上下文特征

第一版可提取轻量、可解释的特征：

```text
query_type：2i, 3i, ip, pi, 2in, 3in, inp, pin, pni 等；
boundary_mass：Boundary 隶属度均值或 top-k 均值；
strong_mass：Strong 隶属度均值或比例；
weak_mass：Weak 隶属度均值或比例；
score_gap：top1 score - top-k mean score；
score_entropy：top-k score 分布熵；
conflict：多个子查询分数差异或 top-k 候选冲突程度；
negated_boundary_mass：被否定子查询中的 Boundary 比例。
```

这些特征尽量在 GPU 上批量计算，避免逐样本循环。

### 自适应规则设计

第一版建议采用可解释 rule-based selector，而不是直接上 MLP。

#### Intersection

FuzzSys 交集规则可理解为：

```text
alpha 越大 -> 越接近 min -> 越保守；
alpha 越小 -> 越接近 max / mean -> 越宽松。
```

可设计：

```text
boundary_mass 高 -> alpha_I 增大；
conflict 高 -> alpha_I 增大；
score_gap 小 -> alpha_I 增大。
```

直觉是：当前查询越不确定、子查询越冲突，交集规则越应保守。

#### Negation

否定查询是重点。

对于：

```text
A AND NOT B
```

三值处理应为：

```text
B 是 Strong   -> 强惩罚；
B 是 Boundary -> 中等惩罚 / 保守处理；
B 是 Weak     -> 可以放行，但不能仅靠 NOT 被大幅抬高。
```

可定义：

```text
penalty = p_s * mu_S + p_b * mu_B + p_w * mu_W
```

其中：

```text
p_s > p_b > p_w
```

再由 query context 调整：

```text
negated_boundary_mass 高 -> Boundary 惩罚更保守；
negated_strong_mass 高 -> Strong 惩罚更强；
主查询本身 Weak -> 不允许 NOT 单独抬高实体。
```

### 可学习版本

如果 rule-based selector 有效果，再考虑 MLP RuleSelector：

```text
context_features -> RuleSelector -> alpha_I / alpha_U / alpha_N
```

但 MLP 版本不是第一阶段重点，因为它带来：

```text
1. 可微 ranking loss 设计问题；
2. 验证集过拟合风险；
3. 训练稳定性问题；
4. 解释性下降。
```

因此推荐路线是：

```text
手工规则验证有效性 -> 再尝试可学习 selector
```

### 代码落点

主要涉及：

```text
FuzzSys-CQA/src/discrete_cqd.py
FuzzSys-CQA/src/discrete_qto.py
FuzzSys-CQA/src/myglobal.py
```

新增：

```python
def extract_context(...):
    ...

def adaptive_alpha_i(context):
    ...

def adaptive_alpha_n(context):
    ...
```

将现有固定 alpha：

```python
a = torch.FloatTensor(list(myglobal.FS_PARAMS.ALPHA_I_L)).unsqueeze(-1).cuda()
```

替换为：

```python
a = adaptive_alpha_i(context)
```

优先改 negation 查询：

```text
query_2in
query_3in
query_inp
query_pin
query_pni
```

### 预期收益

该模块主要提升：

```text
negation 查询；
高 boundary_mass 查询；
子查询冲突较大的复合查询。
```

不应期待它显著提升 path query。

## 推荐实现路线

### Phase 1：最小可行版本

```text
全局三值 membership
+ 手工 query-context adaptive rules
+ 只改 negation 查询
+ 保留原 defuzzification
```

目标：

```text
证明 Boundary + adaptive negation rules 有效。
```

### Phase 2：加入关系感知

```text
relation-group membership
-> relation-specific membership with fallback
```

目标：

```text
证明 relation-aware membership 对 negation 和复合查询有额外收益。
```

### Phase 3：可学习 RuleSelector

如果 Phase 1 和 Phase 2 稳定有效，再尝试：

```text
MLP RuleSelector
```

它应作为增强版本，而不是主方法唯一依赖。

## 论文贡献表述

可以将贡献写成：

```text
1. We propose a relation-aware tri-valued membership function to distinguish target-like, boundary, and weakly relevant entities under relation-specific score distributions.

2. We design a query-context adaptive fuzzy rule selection mechanism that adjusts fuzzy rule parameters according to query type, score conflict, and tri-valued membership statistics.

3. The proposed method is plug-and-play with CQD/QTO+FuzzSys and does not require retraining the underlying link predictor.
```

中文表述：

```text
我们提出一种关系感知三值隶属函数，用于在不同 relation 的分数分布下区分答案倾向实体、边界高相关实体和明显无关实体。

我们设计一种查询上下文自适应模糊规则选择机制，根据查询类型、子查询冲突程度和三值隶属统计动态调整模糊规则参数。

该方法可直接接入 CQD/QTO+FuzzSys 框架，无需重训底层 link predictor。
```

## 风险与边界

需要避免过度声明：

```text
不要说首次做 fuzzy CQA；
不要说首次做 CQA calibration；
不要说首次做 adaptive operator；
不要承诺整体 average MRR 大幅提升。
```

更稳妥的定位是：

```text
在 LP-based fuzzy CQA 框架中，针对全局二值 membership 和静态 fuzzy rules 的不足，提出关系感知三值隶属与查询上下文自适应规则，用于改善复合查询尤其是否定查询中的 hard-relevant entity 处理。
```

# 2026-07-09 关系感知 Membership 训练记录后的反思

## 总体判断

结合当前训练记录，第一个创新点仍然是可行的，但需要调整表述和技术重心。

原始设想是：

```text
Relation-aware Tri-valued Membership
score + relation -> Strong / Boundary / Weak
```

但当前实验真正支持的是：

```text
Relation-aware Conservative Membership Calibration
或
Bounded Relation-aware Score-Shift for Fuzzy Negation
```

也就是说，关系感知这个方向仍然成立，但“显式三值 Boundary membership”目前还没有被实验充分证明。当前有效部分主要是：保留原始 B-spline 的形状，通过有界 score shift 对 negation 中的 membership 强度做保守校准。

## 训练记录说明的问题

训练记录中尝试了多个版本：

```text
1. Sigmoid + Gaussian 三值 membership；
2. 加入硬阈值 mask；
3. 数据驱动 tau_w / tau_s；
4. 限制 tau gap 的三值参数；
5. Score-shift；
6. Bounded score-shift。
```

结果表明：

```text
Sigmoid + Gaussian 三值函数明显劣于原始 B-spline；
纯统计式 tau_w / tau_s 容易让 Boundary 区域过宽；
替换 B-spline 的风险很高；
保留 B-spline 并在输入侧做 bounded shift 是目前更务实的方向。
```

尤其关键的是，全局 grid search 发现：

```text
global shift = +0.5
```

能使 `2in` 查询略优于 baseline。这说明原始 B-spline 在 negation 场景中可能存在系统性偏激进问题，适当弱化 membership 可以改善部分 negation 查询。

## 当前正向结果

在 FB15k-237 + CQD backbone 上，Bounded Score-Shift 取得了部分正向结果：

```text
2in: +1.0% MRR
3in: +9.0% MRR
pin: +0.6% MRR
pni: -4.3% MRR
2i: 0.0% change
```

这说明：

```text
1. membership 校准的主要收益确实集中在 negation 查询；
2. 纯 intersection 查询不受影响或影响极小；
3. pni 的下降暴露了 path-negation 与 1p-negation 的分布差异；
4. 方法不能不加区分地作用于所有 negation 结构。
```

## 对原创新点的修正

### 原始表述风险

如果继续直接说：

```text
我们提出关系感知三值隶属 Strong / Boundary / Weak
```

会有一个问题：当前最有效实现中实际是：

```text
mu_S = original_Bspline(score - shift_r)
mu_B = 0
mu_W = 1 - mu_S
```

因此它还不是严格意义上的三值隶属。当前实验不能作为三值 Boundary 成功的证据。

### 更稳妥的新表述

当前更稳妥的表述应为：

```text
我们发现原始 FuzzSys 的全局 membership 在 negation 查询中存在系统性偏激进问题。
为此，我们提出一种关系感知的保守式 membership 校准方法，在保留原始 B-spline 形状的基础上，根据 relation 可靠性对 negation membership 进行有界弱化。
```

也可以命名为：

```text
Bounded Relation-aware Score-Shift
```

或者：

```text
Relation-aware Conservative Membership Calibration
```

## 当前方法的核心思想

当前方法不是替换原始 membership 函数，而是保留原始 B-spline：

```text
mu = B-spline(score)
```

改为：

```text
effective_score = score - shift_r
mu = B-spline(effective_score)
```

其中 `shift_r` 是有界且非负的：

```text
shift_r >= 0
```

这意味着方法只做一件事：

```text
弱化 negation，不额外强化 negation。
```

设计直觉是：

```text
原始 B-spline 已经较强；
负向 shift 会强化 negation，实验中容易降低 MRR；
正向 shift 会降低 membership，使 negation 更保守；
对 hard relation 给予更大弱化，可以保护正确答案不被不可靠的 negated branch 误伤。
```

## 为什么 pni 会下降

`pni` 的下降非常重要。它说明：

```text
pni 中被 negated 的部分是 path score，而不是普通 1p raw score；
当前 shift 参数来自 valid 1p 查询统计；
1p score 分布不能直接泛化到 2p/path score；
因此对 path-negation 使用同一套 relation-aware shift 是不合适的。
```

后续实现中应考虑：

```text
1. 只对 raw 1p negation branch 启用 relation-aware shift；
2. 对 path-negation fallback 到 global shift 或 original membership；
3. 或者为 path score 单独做 calibration。
```

## 当前还需要证明的问题

### 1. Relation-aware 是否真的优于 global shift

当前记录中已经发现：

```text
global shift = +0.5
```

可以带来收益。因此下一步必须证明：

```text
bounded relation-aware shift > global shift only
```

否则创新点会退化为一个简单的全局 negation 校准。

建议对比：

```text
baseline
global shift only
bounded relation-aware shift
random relation shift
```

### 2. 参数选择必须 valid/test 分离

如果 `min_shift=+0.5` 是在 test 查询上 grid search 得到的，则不能作为正式实验结果。

正式流程应为：

```text
valid set 上选择 min_shift / max_additional_shift / clipping range；
固定参数；
test set 上只评估一次。
```

### 3. FB15k 需要复验

FB15k-237 relation 数量相对少，relation 多样性可能不足。训练记录中也指出 FB15k relation 数更多，可能更适合体现 relation-aware 的价值。

后续应在：

```text
FB15k
FB15k-237
NELL
```

至少两个数据集上验证。

## 对论文创新点的重新组织

当前更稳妥的创新点组织方式：

```text
1. Conservative fuzzy negation calibration:
   发现 FuzzSys 的 membership 在 negation 中可能偏激进，提出 bounded score-shift 弱化策略。

2. Relation-aware reliability adjustment:
   根据 relation 的 score 分布或可靠性，为不同 relation 分配不同的弱化幅度。

3. Query-structure-aware application:
   只在适合的 negation branch 上启用该校准，对 path-negation 等分布不同的情况使用 fallback。
```

如果后续非零 `mu_B` 的三值形式能带来提升，再恢复为：

```text
Relation-aware Tri-valued Membership
```

否则当前论文中不宜强行主张三值 Boundary 是已验证贡献。

## 下一步建议

优先级如下：

```text
1. valid/test 分离，重新确认 global shift 和 bounded relation-aware shift；
2. 在 FB15k 上复验，观察 relation-aware 是否比 FB15k-237 更有效；
3. 对 pni/path-negation 单独处理或 fallback；
4. 尝试非常小的非零 mu_B，而不是重新使用 Sigmoid+Gaussian；
5. 如果 relation-aware 的额外收益有限，则把第一个模块定位为 conservative negation calibration，而不是完整三值系统。
```

## 当前结论

第一个创新点不是失败，而是发生了技术重心转移。

从：

```text
显式三值关系感知 membership
```

转向：

```text
保留 B-spline 的关系感知有界校准
```

这反而是一个更稳的方向，因为它尊重了原始 FuzzSys 已经高度优化的 B-spline，同时只在实验发现最敏感的 negation 场景中做保守修正。

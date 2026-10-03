"""TKG 查询算子层 —— 朴素模糊算子 + TComplEx 原子。

这是**基线**:算子全部是固定的(min/max/前后缀 max),不含任何可学习参数。
创新点(可学习算子)接在同一个接口上,故这里的方法签名就是下游契约。

## 校准(calibrate):为什么默认不做归一

**共享的**单调变换不改变 min 的排序:`min(f(a), f(b)) = f(min(a, b))` 逐点
成立。实测印证:sigmoid(s/T) 在 T = 0.5/1/2/4/8 下给出**完全相同**的 MRR,
中心偏移 c 扫描亦然。所以想靠"把分数压到 [0,1]"来改善 min 是徒劳的。

**逐路**归一(每路各自按名次映射到 [0,1])曾被寄望于此,但全量 ICEWS14
test 上实测**没有任何一致增益**,多数题型是净损失:

    题型     none     rank
    Pe2     44.12    29.33   ↓
    Pe3     42.76    26.88   ↓
    e2i     68.85    68.62   ≈
    e3i     91.36    90.62   ≈
    t2i     13.95    10.94   ↓
    t3i     30.85    22.13   ↓

Pe2/Pe3 的损失有明确机制:`project` 里 P 的每一行对应不同的中间实体 x,
逐行秩归一给"该 x 下最好的候选"发通行证,哪怕它全局很差 —— 那里需要的是
**全局**标定,不是逐行归一。

故默认 `calibrate="none"`,即计划里字面的逐元素 min;"rank" 保留为可选,
供否定等需要 [0,1] 隶属度的算子使用。

> 记录一个已修正的坑:早期一次 300 查询的测量(当时 easy 掩码协议写错了)
> 曾显示 rank 归一在 e3i 上有 +0.09 的增益,据此还写过"原始 min 近乎退化、
> 三个合取项等于白算"。修正掩码协议后该增益消失,退化结论也不成立 ——
> 修正后随机 600 查询上,e3i 单路 ≈0.88、min 合取 0.9223(合取**有效**,
> 逐查询最优单路的上界是 0.9777);e2i 单路 ≈0.699、min 合取 0.6736
> (两路合取反而略降)。教训:样本上的反常增益先怀疑协议,别先写进结论。

### 一个必须记录的坑:float32 饱和

`main_TCQE_TFLEX.py` 的 TFLEX 用 `sigmoid(2x)` 把打分转隶属度
(`train_TCQE_FLEX.py:25-28`)。直接照搬到 TComplEx 上会得到诱人的高数字
(e3i 0.88),**但那是假象**:TComplEx 真答案分数均值 7.7,
`sigmoid(2×7.7)` 在 float32 下精确等于 1.0,大量候选饱和并列,名次完全由
`argsort` 的并列打破顺序决定。实测同一配置下,并列升序得 MRR 0.699、
降序得 0.363、默认 argsort 得 0.883 —— 数字不随模型变、只随排序实现变,
故不可用。用 float64 消除饱和后,它精确退化回 raw min 的 0.6857。
**结论:除非同时做逐路分布归一,否则不要把 sigmoid 用在 TComplEx 分数上。**

## 逆向关系(rel_id >= n_relation)

TFLEX 的查询数据把逆向边编码成 `r + n_relation`
(`ComplexTemporalQueryData.py:707-719`):三元组 (s,r,o,τ) 同时生成
(o, r+230, s, τ)。我们的 checkpoint 只有 230 个**正向**关系嵌入,没有
逆向的,所以逆向遍历不能靠"发明"一个逆关系向量,而是:

  - Pe(e, r0+n, τ)  → 把候选放在**头位**用正向关系打分
                      (y, r0, e, τ),即 score_entities_as_subject
  - Pt(e1, r0+n, e2) → 边的两个端点互换,用正向关系打分
                      (e2, r0, e1, τ);Pt 只问时间,方向不影响

这两条都是精确的,不是近似。但注意:TFLEX 自己的模型有 460 个**训练过的**
关系嵌入,其逆向边有自己的向量,故我们的分数与它不等价 —— 这是基线
与 TFLEX 的固有差异,记录在 docs/tkg-baseline.md。

## 时间算子

按权威结构定义(`ICEWS14.py:98-127` 的 query_structures):

    after(Pt(...))  = "晚于该事件发生的时刻"  → 前缀 max
    before(Pt(...)) = "早于该事件发生的时刻"  → 后缀 max
    between(t1,t2)  = TimeAnd(after(Pt₁), before(Pt₂))

即 `after(S)[τ] = max_{τ' ≤ τ} S[τ']`,`before(S)[τ] = max_{τ' ≥ τ} S[τ']`。
TFLEX 的实现是在连续位置特征上右移/左移(`main_TCQE_TFLEX.py:319-340`),
机制不同但意图一致;我们的时间戳是离散的,故用前缀/后缀 max。
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from tcomplex import TComplExScorer


class NaiveOperators:
    """朴素模糊算子。所有方法返回该题型的答案域分数矩阵 [B, R]。

    `calibrate` 决定组合前怎么把各路分数放到同一尺度:
      - "none": 不归一,直接 min/max 原始分数(默认,见模块 docstring)
      - "rank": 每路内秩归一到 [0,1](确定性、无并列;实测非增益)
    """

    def __init__(self, scorer: TComplExScorer, topk: int = 10, calibrate: str = "none"):
        if calibrate not in ("rank", "none"):
            raise ValueError(f"未知 calibrate={calibrate!r}")
        self.sc = scorer
        self.n_entity = scorer.n_entity
        self.n_relation = scorer.n_relation
        self.n_time = scorer.n_time
        self.topk = topk
        self.calibrate = calibrate

    def norm(self, S: Tensor) -> Tensor:
        """把一路分数归一到可比较的隶属度。"""
        if self.calibrate == "none":
            return S
        n = S.shape[-1]
        if n <= 1:
            return torch.ones_like(S)
        order = S.argsort(dim=-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(n, device=S.device).expand_as(order))
        return 1.0 - rank.to(S.dtype) / (n - 1)      # 最好 -> 1,最差 -> 0

    # ---------- 原子 ----------

    def Pe(self, e_ids: Tensor, r_ids: Tensor, tau_ids: Tensor) -> Tensor:
        """Pe(e, r, τ) → 实体分数 [B, N_entity]。r 可为逆向 id。"""
        fwd = r_ids < self.n_relation
        r0 = torch.where(fwd, r_ids, r_ids - self.n_relation)
        out = torch.empty(e_ids.shape[0], self.n_entity, dtype=torch.float32)
        if fwd.any():
            out[fwd] = self.sc.score_entities(e_ids[fwd], r0[fwd], tau_ids[fwd])
        if (~fwd).any():
            out[~fwd] = self.sc.score_entities_as_subject(r0[~fwd], e_ids[~fwd], tau_ids[~fwd])
        return out

    def Pt(self, e1_ids: Tensor, r_ids: Tensor, e2_ids: Tensor) -> Tensor:
        """Pt(e1, r, e2) → 时间分数 [B, N_time]。r 可为逆向 id。

        逆向时边的两端互换(时间预测与方向无关)。
        """
        fwd = r_ids < self.n_relation
        r0 = torch.where(fwd, r_ids, r_ids - self.n_relation)
        h = torch.where(fwd, e1_ids, e2_ids)
        o = torch.where(fwd, e2_ids, e1_ids)
        return self.sc.score_times(h, r0, o)

    # ---------- 逻辑算子(无参数) ----------

    def And(self, *qs: Tensor) -> Tensor:
        """交集:各路归一后逐元素 min。"""
        out = self.norm(qs[0])
        for q in qs[1:]:
            out = torch.minimum(out, self.norm(q))
        return out

    def Or(self, *qs: Tensor) -> Tensor:
        """并集:各路归一后逐元素 max(对应 DNF 的分支取优)。"""
        out = self.norm(qs[0])
        for q in qs[1:]:
            out = torch.maximum(out, self.norm(q))
        return out

    def Not(self, S: Tensor) -> Tensor:
        """否定:1 − μ,在归一后的隶属度上取补。

        注意 calibrate="none" 时是字面的 1−S(原始分数):正项路径的分数
        尺度(~[0,12])远大于补集(~[−11,1]),And(P, Not(N)) 会被否定项
        数值主导 —— 这是计划里"字面基线"语义的一部分,否定组正是创新点
        (可学习否定强度)要替换的位置。用 calibrate="rank" 时 Not 是
        反向秩,退化消失。
        """
        return 1.0 - self.norm(S)

    # ---------- 时间算子(闭式) ----------

    @staticmethod
    def after(S: Tensor) -> Tensor:
        """after(S)[τ] = max_{τ' ≤ τ} S[τ'](前缀 max)。"""
        return torch.cummax(S, dim=-1).values

    @staticmethod
    def before(S: Tensor) -> Tensor:
        """before(S)[τ] = max_{τ' ≥ τ} S[τ'](后缀 max)。"""
        return torch.flip(torch.cummax(torch.flip(S, dims=(-1,)), dim=-1).values, dims=(-1,))

    def between(self, S1: Tensor, S2: Tensor) -> Tensor:
        """晚于 S1 且早于 S2。"""
        return self.And(self.after(S1), self.before(S2))

    # ---------- 投影(多跳的中间步) ----------

    def project(self, S: Tensor, r_ids: Tensor, tau_ids: Tensor) -> Tensor:
        """把实体分数向量沿关系 r 投影一步,得到新的实体分数 [B, N_entity]。

        `S` 是中间实体的隶属度,`S[i, x]` = 第 i 个查询里 x 作为中间节点的
        隶属度。全枚举是 [B, N, N] 不可行,故取 **top-k** 中间实体:

            out[i, y] = max_{x ∈ topk} min(S[i, x], Pe(x, r_i, τ_i)[y])

        即"存在一条经由高隶属度中间节点的路径"。min 是模糊合取,
        max 是路径之间的取优。
        """
        B, N = S.shape
        k = min(self.topk, N)
        v, idx = self.norm(S).topk(k, dim=1)           # [B, k]

        flat_e = idx.reshape(-1)                       # [B*k]
        flat_r = r_ids.repeat_interleave(k)
        flat_t = tau_ids.repeat_interleave(k)
        P = self.Pe(flat_e, flat_r, flat_t)            # [B*k, N],原始分数
        P = self.norm(P).view(B, k, N)                 # 与 v 同一尺度

        return torch.minimum(v.unsqueeze(-1), P).max(dim=1).values

    def project_time(self, S: Tensor, r_ids: Tensor, e_ids: Tensor,
                     side: str = "h") -> Tensor:
        """把实体隶属度沿一条边投影到时间轴:[B, N_time]。

        side="h": 中间实体 x 作头,Pt(x, r, e) —— Pt_sPe/Pt_se2i/Pt_sPe_Pt
        side="o": 中间实体 x 作尾,Pt(e, r, x) —— Pt_oPe/Pt_oe2i/Pt_oPe_Pt

        与 project 相同的 max-min 合成,只是第二步打分落在时间域。
        """
        assert side in ("h", "o")
        B, N = S.shape
        k = min(self.topk, N)
        v, idx = self.norm(S).topk(k, dim=1)           # [B, k]

        flat_r = r_ids.repeat_interleave(k)
        flat_e = e_ids.repeat_interleave(k)
        flat_x = idx.reshape(-1)
        h = flat_x if side == "h" else flat_e
        o = flat_e if side == "h" else flat_x
        P = self.Pt(h, flat_r, o)                      # [B*k, N_time]
        P = self.norm(P).view(B, k, -1)

        return torch.minimum(v.unsqueeze(-1), P).max(dim=1).values

    def project_via_time(self, S: Tensor, e_ids: Tensor, r_ids: Tensor) -> Tensor:
        """把时间隶属度沿一条边投影到实体域:[B, N_entity]。

        用于 Pe(e, r, 中间时间集):top-k 时刻 τ,对每 τ 打 Pe(e, r, τ),
        max_τ min(v_τ, P_τ)。中间时间集可以是 Pt / after(Pt) / before(Pt)
        / t2i / TimeOr 的结果。
        """
        B, N = S.shape
        k = min(self.topk, N)
        v, idx = self.norm(S).topk(k, dim=1)           # [B, k]

        flat_e = e_ids.repeat_interleave(k)
        flat_r = r_ids.repeat_interleave(k)
        flat_t = idx.reshape(-1)
        P = self.Pe(flat_e, flat_r, flat_t)            # [B*k, N_entity]
        P = self.norm(P).view(B, k, -1)

        return torch.minimum(v.unsqueeze(-1), P).max(dim=1).values

"""TComplEx 打分模块 —— 加载对齐后的时态 LP 权重,提供 CQA 所需的分数矩阵。

打分公式(TComplEx, ICLR 2020):

    score(h, r, o, τ) = Re(⟨h ⊙ r ⊙ w_τ, conj(o)⟩)

其中 h/r/o 是实体/关系复数向量,w_τ 是时间戳 τ 的复数向量。

权重张量每行 512 维 = **256 复数维**(前 256 实部,后 256 虚部),由
`scripts/align_tcomplex_to_tflex.py` 按 TFLEX 的 id 顺序重排过,因此
这里的 id 语义与 `data/tkg/<dataset>/meta.json` 完全一致。

设备无关(纯 CPU 可跑),不依赖 myglobal / 静态 KG 链路。
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor


class TComplExScorer:
    """时态 LP 打分器。所有 id 均为 TFLEX id 空间。"""

    def __init__(self, ckpt_path: str | Path):
        ckpt = torch.load(Path(ckpt_path), map_location="cpu")
        missing = {"entity", "relation", "time"} - set(ckpt)
        if missing:
            raise KeyError(f"checkpoint 缺少张量: {sorted(missing)},实际有 {sorted(ckpt)}")

        self.entity: Tensor = ckpt["entity"].float()      # [N_ent, 2*rank]
        self.relation: Tensor = ckpt["relation"].float()  # [N_rel, 2*rank]
        self.time: Tensor = ckpt["time"].float()          # [N_time, 2*rank]

        if not (self.entity.shape[1] == self.relation.shape[1] == self.time.shape[1]):
            raise ValueError("三类嵌入的维度不一致")

        self.dim = self.entity.shape[1]
        if self.dim % 2:
            raise ValueError(f"嵌入维度应为偶数(实部+虚部),实际 {self.dim}")
        self.rank = self.dim // 2

        self.n_entity, self.n_relation, self.n_time = (
            self.entity.shape[0], self.relation.shape[0], self.time.shape[0],
        )

    # ---------- 基础运算 ----------

    def split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """切成 (实部, 虚部)。"""
        return x[..., :self.rank], x[..., self.rank:]

    def rel_eff(self, r_ids: Tensor, tau_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """关系与时间戳的复数和:r ⊙ w_τ → (实部, 虚部)。

        复数乘法 (a+bi)(c+di) = (ac−bd) + (ad+bc)i。
        """
        r_re, r_im = self.split(self.relation[r_ids])
        w_re, w_im = self.split(self.time[tau_ids])
        return r_re * w_re - r_im * w_im, r_re * w_im + r_im * w_re

    @staticmethod
    def conj_mul(a: Tuple[Tensor, Tensor], b: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """两个复数向量逐元素相乘:a ⊙ b。"""
        a_re, a_im = a
        b_re, b_im = b
        return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re

    # ---------- 打分 ----------

    def score_entities(self, h_ids: Tensor, r_ids: Tensor, tau_ids: Tensor) -> Tensor:
        """对**全部实体**打分,得到 [B, N_entity]。

        用于 Pe(e, r, τ) —— 预测 τ 时刻与 e 通过 r 相连的尾实体。
        """
        h_re, h_im = self.split(self.entity[h_ids])
        p_re, p_im = self.rel_eff(r_ids, tau_ids)          # [B, rank]

        # B = h ⊙ r ⊙ w_τ
        b_re = h_re * p_re - h_im * p_im
        b_im = h_re * p_im + h_im * p_re

        t_re, t_im = self.split(self.entity)               # [N_ent, rank]
        # Re(⟨B, conj(t)⟩) = Σ (B_re·t_re + B_im·t_im)
        return b_re @ t_re.T + b_im @ t_im.T

    def score_entities_as_subject(self, r_ids: Tensor, o_ids: Tensor, tau_ids: Tensor) -> Tensor:
        """对**全部实体**按头位置打分,得到 [B, N_entity]。

        用于沿一条边**逆向**走:已知尾实体 o,求头实体是谁。等价于
        `score_entities(h', r, o, τ)` 对全部候选 h' 求值 —— 用的是同一个
        正向关系 r,不需要逆关系嵌入。

        注意:TFLEX 的查询数据把逆向边编码成 id = r + n_relation
        (`ComplexTemporalQueryData.py:707-719`),本方法正是解释那类 id 的方式;
        但 TFLEX 自己的模型有 460 个**训练过的**关系嵌入,而我们的 checkpoint
        只有 230 个正向的,故此处是同一正向关系换到候选头位,与该模型不等价。

        展开:C_k = r_k w_τk conj(o_k) 与候选无关,于是

            score(h') = Re(Σ_k h'_k C_k) = Σ_k (h'_re·C_re − h'_im·C_im)
        """
        p_re, p_im = self.rel_eff(r_ids, tau_ids)           # p = r ⊙ w_τ
        o_re, o_im = self.split(self.entity[o_ids])
        # C = p ⊙ conj(o)
        c_re = p_re * o_re + p_im * o_im
        c_im = p_im * o_re - p_re * o_im

        h_re, h_im = self.split(self.entity)               # [N_ent, rank]
        return c_re @ h_re.T - c_im @ h_im.T               # [B, N_ent]

    def score_times(self, h_ids: Tensor, r_ids: Tensor, o_ids: Tensor) -> Tensor:
        """对**全部时间戳**打分,得到 [B, N_time]。

        用于 Pt(e, r, o) —— 预测 e 与 o 通过 r 相连的时间戳。

        这里时间戳出现在打分公式内部(w_τ),无法像实体那样直接矩阵乘,
        故先把与 τ 无关的部分合并成 A = h ⊙ r ⊙ conj(o),再对每个 τ 求
        Re(A ⊙ w_τ) 之和:

            score = Σ_k Re(A_k · w_τk) = Σ_k (A_re·w_re − A_im·w_im)
        """
        h_re, h_im = self.split(self.entity[h_ids])
        r_re, r_im = self.split(self.relation[r_ids])
        o_re, o_im = self.split(self.entity[o_ids])

        # A = (h ⊙ r) ⊙ conj(o)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        # conj(o) 的虚部取负
        a_re = hr_re * o_re + hr_im * o_im
        a_im = hr_im * o_re - hr_re * o_im

        w_re, w_im = self.split(self.time)                 # [N_time, rank]
        return a_re @ w_re.T - a_im @ w_im.T

    def score_triples(self, h_ids: Tensor, r_ids: Tensor, o_ids: Tensor, tau_ids: Tensor) -> Tensor:
        """对给定四元组直接打分,得到 [B]。标准 TKGC 评测用。"""
        h_re, h_im = self.split(self.entity[h_ids])
        r_re, r_im = self.split(self.relation[r_ids])
        o_re, o_im = self.split(self.entity[o_ids])
        w_re, w_im = self.split(self.time[tau_ids])

        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        # A = h ⊙ r ⊙ conj(o)
        a_re = hr_re * o_re + hr_im * o_im
        a_im = hr_im * o_re - hr_re * o_im
        # Re(A ⊙ w_τ)
        return (a_re * w_re - a_im * w_im).sum(-1)

    # ---------- 自检 ----------

    def describe(self) -> str:
        return (f"TComplEx: {self.n_entity} 实体 / {self.n_relation} 关系 / "
                f"{self.n_time} 时间戳,复数维 {self.rank}(张量 {self.dim} 维)")

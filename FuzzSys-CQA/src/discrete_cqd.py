# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor
import numpy as np

from typing import Callable, Tuple, Optional

import src.myglobal as myglobal

from scipy.interpolate import BSpline, splrep, splev

# godel
def mem_tnorm(mem_list):
    return torch.min(torch.vstack(mem_list), dim=0, keepdim=True)[0]

def mem_tconorm(mem_list):
    return torch.max(torch.vstack(mem_list), dim=0, keepdim=True)[0]

# product
# def mem_tnorm(mem_list):
#     return torch.prod(torch.vstack(mem_list), dim=0, keepdim=True)

# def mem_tconorm(mem_list):
#     return 1 - torch.prod(1 - torch.vstack(mem_list), dim=0, keepdim=True)

def cal_membership(scores):
    if myglobal.FS_PARAMS.MEMB_TYPE == 'symbolic':
        m = torch.sigmoid(myglobal.FS_PARAMS.LR_b0 + myglobal.FS_PARAMS.LR_b1 * scores)
    else:
        tck = (myglobal.FS_PARAMS.Bspline_t, myglobal.FS_PARAMS.Bspline_c, myglobal.FS_PARAMS.Bspline_k)
        m = splev(scores.detach().cpu().numpy(), tck)
        m = torch.FloatTensor(m).cuda().clamp(min=0., max=1.)

        pos_mask = (scores >= tck[0][-1]).float()
        m = torch.maximum(m, pos_mask)

        neg_mask = (scores >= myglobal.FS_PARAMS.threshd).float()
        m = neg_mask * m

    return m


def cal_membership3(scores, rel_ids=None, call_site_type='combined',
                    query_type=None):
    """
    Three-value relation-aware membership function.

    When use_rel_memb is enabled and rel_ids are provided, uses
    per-relation Strong/Boundary/Weak thresholds. Otherwise falls
    back to the original global two-value membership.

    Args:
        scores: tensor of any shape — LP scores
        rel_ids: tensor of relation IDs, optional
        call_site_type: 'negation_raw', 'combined', or 'union'
        query_type: '2in','3in','pin','pni' — scales offset per query structure
    """
    if myglobal.FS_PARAMS.use_rel_memb and rel_ids is not None:
        return myglobal.FS_PARAMS.REL_MEMB.cal_membership3(
            scores, rel_ids, call_site_type, query_type=query_type)
    else:
        # Fallback: original global two-value membership
        m = cal_membership(scores)
        # Degrade to binary: Strong = m, Boundary = 0, Weak = 1-m
        zero = torch.zeros_like(m)
        return m, zero, 1.0 - m


def defuzzification(w_tensor, res_tensor):
    if myglobal.DEFUZZ == 'mean':
        sum_w = torch.sum(w_tensor, dim=0)
        res = torch.sum(w_tensor * res_tensor, dim=0, keepdim=True)
        res = res / sum_w
    else:
        max_rind = torch.argmax(w_tensor, dim=0).unsqueeze(1)
        cind = torch.arange(res_tensor.shape[1]).unsqueeze(1)
        res = res_tensor[max_rind, cind].T
    return res

def score_candidates(s_emb: Tensor,
                     p_emb: Tensor,
                     candidates_emb: Tensor,
                     k: Optional[int],
                     entity_embeddings: nn.Module,
                     scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tuple[Tensor, Optional[Tensor]]:

    batch_size = max(s_emb.shape[0], p_emb.shape[0])
    embedding_size = s_emb.shape[1]

    def reshape(emb: Tensor) -> Tensor:
        if emb.shape[0] < batch_size:
            n_copies = batch_size // emb.shape[0]
            emb = emb.reshape(-1, 1, embedding_size).repeat(1, n_copies, 1).reshape(-1, embedding_size)
        return emb

    s_emb = reshape(s_emb)
    p_emb = reshape(p_emb)
    nb_entities = candidates_emb.shape[0]

    x_k_emb_3d = None

    # [B, N]
    atom_scores_2d = scoring_function(s_emb, p_emb, candidates_emb)
    atom_k_scores_2d = atom_scores_2d

    if k is not None:
        k_ = min(k, nb_entities)

        # [B, K], [B, K]
        atom_k_scores_2d, atom_k_indices = torch.topk(atom_scores_2d, k=k_, dim=1)

        # [B, K, E]
        x_k_emb_3d = entity_embeddings(atom_k_indices)

    return atom_k_scores_2d, x_k_emb_3d

def query_1p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tensor:
    s_emb = entity_embeddings(queries[:, 0])
    p_emb = predicate_embeddings(queries[:, 1])
    candidates_emb = entity_embeddings.weight

    assert queries.shape[1] == 2

    res, _ = score_candidates(s_emb=s_emb, p_emb=p_emb,
                              candidates_emb=candidates_emb, k=None,
                              entity_embeddings=entity_embeddings,
                              scoring_function=scoring_function)

    if myglobal.GRID:
        noise = torch.rand_like(res) - 0.5
        res = (res + noise).clamp(min=0.)

    return res


def query_2p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, K], [B, K, E]
    atom1_k_scores_2d, x1_k_emb_3d = score_candidates(s_emb=s_emb, p_emb=p1_emb,
                                                      candidates_emb=candidates_emb, k=k,
                                                      entity_embeddings=entity_embeddings,
                                                      scoring_function=scoring_function)

    # [B * K, E]
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)

    # [B * K, N]
    atom2_scores_2d, _ = score_candidates(s_emb=x1_k_emb_2d, p_emb=p2_emb,
                                          candidates_emb=candidates_emb, k=None,
                                          entity_embeddings=entity_embeddings,
                                          scoring_function=scoring_function)

    # [B, K] -> [B, K, N]
    atom1_scores_3d = atom1_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)
    # [B * K, N] -> [B, K, N]
    atom2_scores_3d = atom2_scores_2d.reshape(batch_size, -1, nb_entities)

    atom1_scores_3d = (atom1_scores_3d > 0.).float() * atom1_scores_3d

    res = t_norm(atom1_scores_3d, atom2_scores_3d)

    # [B, K, N] -> [B, N]
    res, _ = torch.max(res, dim=1)
    return res


def query_3p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])
    p3_emb = predicate_embeddings(queries[:, 3])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, K], [B, K, E]
    atom1_k_scores_2d, x1_k_emb_3d = score_candidates(s_emb=s_emb, p_emb=p1_emb,
                                                      candidates_emb=candidates_emb, k=k,
                                                      entity_embeddings=entity_embeddings,
                                                      scoring_function=scoring_function)

    # [B * K, E]
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)

    # [B * K, K], [B * K, K, E]
    atom2_k_scores_2d, x2_k_emb_3d = score_candidates(s_emb=x1_k_emb_2d, p_emb=p2_emb,
                                                      candidates_emb=candidates_emb, k=k,
                                                      entity_embeddings=entity_embeddings,
                                                      scoring_function=scoring_function)


    # [B * K * K, E]
    x2_k_emb_2d = x2_k_emb_3d.reshape(-1, emb_size)

    # [B * K * K, N]
    atom3_scores_2d, _ = score_candidates(s_emb=x2_k_emb_2d, p_emb=p3_emb,
                                          candidates_emb=candidates_emb, k=None,
                                          entity_embeddings=entity_embeddings,
                                          scoring_function=scoring_function)


    # [B, K] -> [B, K, N]
    atom1_scores_3d = atom1_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)

    # [B * K, K] -> [B, K * K, N]
    atom2_scores_3d = atom2_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)

    # [B * K * K, N] -> [B, K * K, N]
    atom3_scores_3d = atom3_scores_2d.reshape(batch_size, -1, nb_entities)

    atom1_scores_3d = atom1_scores_3d.repeat(1, atom3_scores_3d.shape[1] // atom1_scores_3d.shape[1], 1)

    atom1_scores_3d = (atom1_scores_3d > 0.).float() * atom1_scores_3d
    atom2_scores_3d = (atom2_scores_3d > 0.).float() * atom2_scores_3d
    # atom3_scores_3d = (atom3_scores_3d > 0.).float() * atom3_scores_3d

    res = t_norm(atom1_scores_3d, atom2_scores_3d)
    res = t_norm(res, atom3_scores_3d)

    # [B, K, N] -> [B, N]
    res, _ = torch.max(res, dim=1)
    return res

def query_2i(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)

    # Branch-level per-relation shift: make scores comparable across relations
    # before max/min combination (Module 1: cross-relation score comparability).
    if myglobal.FS_PARAMS.use_rel_memb:
        rel_memb = myglobal.FS_PARAMS.REL_MEMB
        shift_1 = rel_memb.get_shift_for_rel(queries[:, 1])  # [B]
        shift_2 = rel_memb.get_shift_for_rel(queries[:, 3])  # [B]
        scores_1 = scores_1 - shift_1.unsqueeze(1)
        scores_2 = scores_2 - shift_2.unsqueeze(1)

    M = torch.maximum(scores_1, scores_2)
    m = torch.minimum(scores_1, scores_2)

    # Alpha-adaptation (Module 2): adjust soft-min alpha per query based
    # on relation pair reliability. Unreliable pairs get smaller alpha
    # (more weight on max → more lenient intersection).
    alpha_base = torch.FloatTensor(list(myglobal.FS_PARAMS.ALPHA_I_L))  # [3]
    if myglobal.FS_PARAMS.use_rel_memb:
        rel_r1 = rel_memb.get_reliability_for_rel(queries[:, 1])  # [B]
        rel_r2 = rel_memb.get_reliability_for_rel(queries[:, 3])  # [B]
        rel_pair = torch.minimum(rel_r1, rel_r2)  # weakest link
        alpha_shift = (rel_pair - 0.5) * 0.3  # [B]
        # Preserve original [3,1] shape by averaging shift across batch
        alpha_shift_mean = alpha_shift.mean()  # scalar on CUDA
        a = alpha_base.cuda() + alpha_shift_mean  # [3] + scalar → [3]
        a = torch.clamp(a, 0.0, 1.0).unsqueeze(-1)  # [3, 1]
    else:
        a = alpha_base.unsqueeze(-1).cuda()  # [3, 1]

    res_tensor = (1. - a) * M + a * m

    # Relation-aware: use first branch relation for combined scores
    mu_S1, mu_B1, mu_W1 = cal_membership3(res_tensor[0], queries[:, 1], 'combined')
    mu_S3, mu_B3, mu_W3 = cal_membership3(res_tensor[2], queries[:, 1], 'combined')
    w_1 = mu_S1
    w_2 = torch.ones_like(w_1)
    w_3 = 1.0 - mu_S3

    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res


def query_3i(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)
    scores_3 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 4:6], scoring_function=scoring_function)

    # Branch-level per-relation shift: make scores comparable across relations
    # before sort/max/min combination.
    if myglobal.FS_PARAMS.use_rel_memb:
        rel_memb = myglobal.FS_PARAMS.REL_MEMB
        shift_1 = rel_memb.get_shift_for_rel(queries[:, 1])  # [B]
        shift_2 = rel_memb.get_shift_for_rel(queries[:, 3])  # [B]
        shift_3 = rel_memb.get_shift_for_rel(queries[:, 5])  # [B]
        scores_1 = scores_1 - shift_1.unsqueeze(1)
        scores_2 = scores_2 - shift_2.unsqueeze(1)
        scores_3 = scores_3 - shift_3.unsqueeze(1)

    scores_tensor = torch.vstack([scores_1, scores_2, scores_3])
    sorted_scores = torch.sort(scores_tensor, dim=0, descending=True).values

    M = sorted_scores[0]
    A = sorted_scores[1]
    m = sorted_scores[2]

    (a_1, a_2, a_3) = myglobal.FS_PARAMS.ALPHA_I_L

    # Alpha-adaptation (Module 2): per-query alpha shift for 3-branch intersection.
    if myglobal.FS_PARAMS.use_rel_memb:
        rel_r1 = rel_memb.get_reliability_for_rel(queries[:, 1])
        rel_r3 = rel_memb.get_reliability_for_rel(queries[:, 3])
        rel_r5 = rel_memb.get_reliability_for_rel(queries[:, 5])
        rel_avg = (rel_r1 + rel_r3 + rel_r5) / 3.0  # [B]
        alpha_shift = (rel_avg - 0.5) * 0.3  # [B]
        a_1 = (a_1 + alpha_shift).clamp(0.0, 1.0)  # [B]
        a_2 = (a_2 + alpha_shift).clamp(0.0, 1.0)  # [B]
        a_3 = (a_3 + alpha_shift).clamp(0.0, 1.0)  # [B]

    res_1 = (((1. - a_1) * M + a_1 * A) + ((1. - a_1) * M + a_1 * m) + ((1. - a_1) * A + a_1 * m))/3

    res_2 = (((1. - a_2) * M + a_2 * A) + ((1. - a_2) * M + a_2 * m) + ((1. - a_2) * A + a_2 * m))/3

    res_3 = (((1. - a_3) * M + a_3 * A) + ((1. - a_3) * M + a_3 * m) + ((1. - a_3) * A + a_3 * m))/3

    # Relation-aware: use first branch relation for combined scores
    mu_S1, mu_B1, mu_W1 = cal_membership3(res_1, queries[:, 1], 'combined')
    mu_S3, mu_B3, mu_W3 = cal_membership3(res_3, queries[:, 1], 'combined')
    w_1 = mu_S1
    w_2 = torch.ones_like(w_1)
    w_3 = 1.0 - mu_S3

    res_tensor = torch.vstack([res_1, res_2, res_3])
    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res


def query_ip(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    # [B, N]
    scores_1 = query_2i(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:4], scoring_function=scoring_function, t_norm=t_norm)

    # [B, E]
    p_emb = predicate_embeddings(queries[:, 4])

    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]

    k_ = min(k, nb_entities)

    # [B, K], [B, K]
    scores_1_k, scores_1_k_indices = torch.topk(scores_1, k=k_, dim=1)

    # [B, K, E]
    scores_1_k_emb = entity_embeddings(scores_1_k_indices)

    # [B * K, E]
    scores_1_k_emb_2d = scores_1_k_emb.reshape(batch_size * k_, emb_size)

    # [B * K, N]
    scores_2, _ = score_candidates(s_emb=scores_1_k_emb_2d, p_emb=p_emb, candidates_emb=e_emb, k=None,
                                   entity_embeddings=entity_embeddings, scoring_function=scoring_function)

    # [B * K, N]
    scores_1_k = scores_1_k.reshape(batch_size, k_, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, k_, nb_entities)

    res = t_norm(scores_1_k, scores_2)

    res, _ = torch.max(res, dim=1)

    return res


def query_pi(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_2p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:3], scoring_function=scoring_function, k=k, t_norm=t_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 3:5], scoring_function=scoring_function)

    M1 = torch.max(scores_1)
    M2 = torch.max(scores_2)
    scores_1 = scores_1*M2/M1

    # Branch-level per-relation shift for the 1p branch only.
    # Path branch (scores_1) is skipped: 1p-derived shifts don't apply to
    # 2p path score distributions (same rationale as negation_path in pni).
    if myglobal.FS_PARAMS.use_rel_memb:
        shift_2 = myglobal.FS_PARAMS.REL_MEMB.get_shift_for_rel(queries[:, 4])
        scores_2 = scores_2 - shift_2.unsqueeze(1)

    M = torch.maximum(scores_1, scores_2)
    m = torch.minimum(scores_1, scores_2)

    # Alpha-adaptation (Module 2): per-query alpha based on relation reliability.
    alpha_base = torch.FloatTensor(list(myglobal.FS_PARAMS.ALPHA_I_L))  # [3]
    if myglobal.FS_PARAMS.use_rel_memb:
        rel_r1 = myglobal.FS_PARAMS.REL_MEMB.get_reliability_for_rel(queries[:, 1])
        rel_r4 = myglobal.FS_PARAMS.REL_MEMB.get_reliability_for_rel(queries[:, 4])
        rel_pair = torch.minimum(rel_r1, rel_r4)
        alpha_shift = (rel_pair - 0.5) * 0.3  # [B]
        a = alpha_base.unsqueeze(1).cuda() + alpha_shift.unsqueeze(0)  # [3, B]
        a = torch.clamp(a, 0.0, 1.0).unsqueeze(2)  # [3, B, 1]
    else:
        a = alpha_base.unsqueeze(1).unsqueeze(1).cuda()  # [3, 1, 1]

    res_tensor = (1. - a) * M + a * m

    # Relation-aware: use first path relation for combined scores
    mu_S1, mu_B1, mu_W1 = cal_membership3(res_tensor[0], queries[:, 1], 'combined')
    mu_S3, mu_B3, mu_W3 = cal_membership3(res_tensor[2], queries[:, 1], 'combined')
    w_1 = mu_S1
    w_2 = torch.ones_like(w_1)
    w_3 = 1.0 - mu_S3

    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res


def query_2u_dnf(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 t_conorm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)

    # Branch-level per-relation shift: make scores comparable across relations
    # before max/min/union combination.
    if myglobal.FS_PARAMS.use_rel_memb:
        rel_memb = myglobal.FS_PARAMS.REL_MEMB
        shift_1 = rel_memb.get_shift_for_rel(queries[:, 1])  # [B]
        shift_2 = rel_memb.get_shift_for_rel(queries[:, 3])  # [B]
        scores_1 = scores_1 - shift_1.unsqueeze(1)
        scores_2 = scores_2 - shift_2.unsqueeze(1)

    M = torch.maximum(scores_1, scores_2)
    m = torch.minimum(scores_1, scores_2)
    MM = torch.max(M)

    a = torch.FloatTensor(list(myglobal.FS_PARAMS.ALPHA_U_L)).unsqueeze(-1).cuda()

    res_tensor = (1. - a) * MM + a * M

    # Relation-aware: use first branch relation for union
    mu_S, mu_B, mu_W = cal_membership3(M, queries[:, 1], 'union')
    w_1 = mu_S
    w_2 = w_1
    w_3 = torch.ones_like(w_1)

    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res


def query_up_dnf(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 k: int,
                 t_norm: Callable[[Tensor, Tensor], Tensor],
                 t_conorm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    # [B, N]
    scores_1 = query_2u_dnf(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                            queries=queries[:, 0:4], scoring_function=scoring_function, t_conorm=t_conorm)

    # [B, E]
    p_emb = predicate_embeddings(queries[:, 5])

    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]

    k_ = min(k, nb_entities)

    # [B, K], [B, K]
    scores_1_k, scores_1_k_indices = torch.topk(scores_1, k=k_, dim=1)

    # [B, K, E]
    scores_1_k_emb = entity_embeddings(scores_1_k_indices)

    # [B * K, E]
    scores_1_k_emb_2d = scores_1_k_emb.reshape(batch_size * k_, emb_size)

    # [B * K, N]
    scores_2, _ = score_candidates(s_emb=scores_1_k_emb_2d, p_emb=p_emb, candidates_emb=e_emb, k=None,
                                   entity_embeddings=entity_embeddings, scoring_function=scoring_function)

    # [B * K, N]
    scores_1_k = scores_1_k.reshape(batch_size, k_, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, k_, nb_entities)

    res = t_norm(scores_1_k, scores_2)

    res, _ = torch.max(res, dim=1)

    return res


# (('e', ('r',)), ('e', ('r', 'n'))): '2in'
def query_2in(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 t_norm: Callable[[Tensor, Tensor], Tensor],
                 negation: Callable[[Tensor], Tensor]) -> Tensor:
    
    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)

    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)

    # Three-value relation-aware negation:
    #   Strong -> full penalty (1.0), Boundary -> half penalty (0.5), Weak -> no penalty (0.0)
    mu_S, mu_B, mu_W = cal_membership3(scores_2, queries[:, 3], 'negation_raw',
                                        query_type='2in')
    neg_penalty = 1.0 * mu_S + 0.5 * mu_B + 0.0 * mu_W
    r_2 = 1.0 - neg_penalty

    (a_1, a_2, a_3) = myglobal.FS_PARAMS.ALPHA_N

    res_1 = ((1 - a_1) + a_1 * r_2) * scores_1
    res_2 = ((1 - a_2) + a_2 * r_2) * scores_1
    res_3 = ((1 - a_3) + a_3 * r_2) * scores_1

    w_1 = torch.ones_like(res_1)

    # Defuzzification weights: use Strong component of combined scores
    mu_S21, mu_B21, mu_W21 = cal_membership3(res_2/(1 - a_2), queries[:, 1], 'combined')
    mu_S22, mu_B22, mu_W22 = cal_membership3(res_2, queries[:, 1], 'combined')
    w_2 = mu_S21 + (1.0 - mu_S22)

    w_3 = torch.ones_like(res_3)

    res_tensor = torch.vstack([res_1, res_2, res_3])
    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res


# (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in'
def query_3in(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor],
             negation: Callable[[Tensor], Tensor]) -> Tensor:

    scores_12 = query_2i(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:4], scoring_function=scoring_function, t_norm=t_norm)

    scores_3 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 4:6], scoring_function=scoring_function)


    # Three-value relation-aware negation for third branch
    mu_S, mu_B, mu_W = cal_membership3(scores_3, queries[:, 5], 'negation_raw',
                                        query_type='3in')
    neg_penalty = 1.0 * mu_S + 0.5 * mu_B + 0.0 * mu_W
    r_3 = 1.0 - neg_penalty

    (a_1, a_2, a_3) = myglobal.FS_PARAMS.ALPHA_N

    res_1 = ((1 - a_1) + a_1 * r_3) * scores_12
    res_2 = ((1 - a_2) + a_2 * r_3) * scores_12
    res_3 = ((1 - a_3) + a_3 * r_3) * scores_12

    w_1 = torch.ones_like(res_1)

    # Defuzzification weights: use Strong component of combined scores
    mu_S21, mu_B21, mu_W21 = cal_membership3(res_2/(1 - a_2), queries[:, 1], 'combined')
    mu_S22, mu_B22, mu_W22 = cal_membership3(res_2, queries[:, 1], 'combined')
    w_2 = mu_S21 + (1.0 - mu_S22)

    w_3 = torch.ones_like(res_3)

    res_tensor = torch.vstack([res_1, res_2, res_3])
    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res

# ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp'
def query_inp(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    # [B, N]
    scores_1 = query_2in(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:5], scoring_function=scoring_function, t_norm=t_norm,
                        negation=None)

    # [B, E]
    p_emb = predicate_embeddings(queries[:, 5])

    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]

    k_ = min(k, nb_entities)

    # [B, K], [B, K]
    scores_1_k, scores_1_k_indices = torch.topk(scores_1, k=k_, dim=1)

    # [B, K, E]
    scores_1_k_emb = entity_embeddings(scores_1_k_indices)

    # [B * K, E]
    scores_1_k_emb_2d = scores_1_k_emb.reshape(batch_size * k_, emb_size)

    # [B * K, N]
    scores_2, _ = score_candidates(s_emb=scores_1_k_emb_2d, p_emb=p_emb, candidates_emb=e_emb, k=None,
                                   entity_embeddings=entity_embeddings, scoring_function=scoring_function)

    # [B * K, N]
    scores_1_k = scores_1_k.reshape(batch_size, k_, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, k_, nb_entities)

    res = t_norm(scores_1_k, scores_2)

    res, _ = torch.max(res, dim=1)

    return res

# (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin'
def query_pin(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_2p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:3], scoring_function=scoring_function, k=k, t_norm=t_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 3:5], scoring_function=scoring_function)

    # Three-value relation-aware negation for second (negated) branch
    mu_S, mu_B, mu_W = cal_membership3(scores_2, queries[:, 4], 'negation_raw',
                                        query_type='pin')
    neg_penalty = 1.0 * mu_S + 0.5 * mu_B + 0.0 * mu_W
    r_2 = 1.0 - neg_penalty

    (a_1, a_2, a_3) = myglobal.FS_PARAMS.ALPHA_N

    res_1 = ((1 - a_1) + a_1 * r_2) * scores_1
    res_2 = ((1 - a_2) + a_2 * r_2) * scores_1
    res_3 = ((1 - a_3) + a_3 * r_2) * scores_1

    w_1 = torch.ones_like(res_1)

    # Defuzzification weights
    mu_S21, mu_B21, mu_W21 = cal_membership3(res_2/(1 - a_2), queries[:, 1], 'combined')
    mu_S22, mu_B22, mu_W22 = cal_membership3(res_2, queries[:, 1], 'combined')
    w_2 = mu_S21 + (1.0 - mu_S22)

    w_3 = torch.ones_like(res_3)

    res_tensor = torch.vstack([res_1, res_2, res_3])
    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res

# (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni'
def query_pni(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_2p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:3], scoring_function=scoring_function, k=k, t_norm=t_norm)
    
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 4:6], scoring_function=scoring_function)

    M1 = torch.max(scores_1)
    M2 = torch.max(scores_2)
    scores_1 = scores_1*M2/M1

    # Three-value relation-aware negation for path (first) branch
    # Use the first path relation for negation check
    mu_S, mu_B, mu_W = cal_membership3(scores_1, queries[:, 1], 'negation_path',
                                        query_type='pni')
    neg_penalty = 1.0 * mu_S + 0.5 * mu_B + 0.0 * mu_W
    r_1 = 1.0 - neg_penalty

    (a_1, a_2, a_3) = myglobal.FS_PARAMS.ALPHA_N

    res_1 = ((1 - a_1) + a_1 * r_1) * scores_2
    res_2 = ((1 - a_2) + a_2 * r_1) * scores_2
    res_3 = ((1 - a_3) + a_3 * r_1) * scores_2

    w_1 = torch.ones_like(res_1)

    # Defuzzification weights
    mu_S21, mu_B21, mu_W21 = cal_membership3(res_2/(1 - a_2), queries[:, 1], 'combined')
    mu_S22, mu_B22, mu_W22 = cal_membership3(res_2, queries[:, 1], 'combined')
    w_2 = mu_S21 + (1.0 - mu_S22)

    w_3 = torch.ones_like(res_3)

    res_tensor = torch.vstack([res_1, res_2, res_3])
    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res
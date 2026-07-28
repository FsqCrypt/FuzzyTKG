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
        m = myglobal.FS_PARAMS.EXP_2 * torch.exp(myglobal.FS_PARAMS.EXP_0 * scores + myglobal.FS_PARAMS.EXP_1) + myglobal.FS_PARAMS.EXP_3
        m = m.clamp(min=0., max=1.)
        max_score = myglobal.FS_PARAMS.Max_score

        max_mask = (scores >= max_score).float()
        m = torch.maximum(m, max_mask)

        min_mask = (scores >= myglobal.FS_PARAMS.threshd).float()
        m = m * min_mask
    else:
        t = myglobal.FS_PARAMS.Bspline_t
        c = myglobal.FS_PARAMS.Bspline_c
        k = myglobal.FS_PARAMS.Bspline_k
        max_score = myglobal.FS_PARAMS.Max_score

        tck = (t, c, k)
        m = splev(scores.detach().cpu().numpy(), tck)
        m = torch.FloatTensor(m).cuda().clamp(min=0., max=1.)

        max_mask = (scores >= max_score).float()
        m = torch.maximum(m, max_mask)

        min_mask = (scores >= myglobal.FS_PARAMS.threshd).float()
        m = m * min_mask

    return m


def cal_membership3(scores, rel_ids=None, call_site_type='combined',
                    query_type=None):
    """
    Three-value relation-aware membership function.

    When use_rel_memb is enabled and rel_ids are provided, uses
    per-relation Strong/Boundary/Weak thresholds. Otherwise falls
    back to the original global two-value membership.

    Args:
        scores: tensor of any shape -- LP scores
        rel_ids: tensor of relation IDs, optional
        call_site_type: 'negation_raw', 'combined', or 'union'
        query_type: '2in','3in','pin','pni' -- scales offset per query structure
    """
    if myglobal.FS_PARAMS.use_rel_memb and rel_ids is not None:
        return myglobal.FS_PARAMS.REL_MEMB.cal_membership3(
            scores, rel_ids, call_site_type, query_type=query_type)
    else:
        # Fallback: original global two-value membership
        m = cal_membership(scores)
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

def relation_projection(embedding, r_embedding, nentity, fraction=10):
    dim = nentity // fraction
    rest = nentity - fraction * dim
    new_embedding = torch.zeros_like(embedding)
    for i in range(fraction):
        s = i * dim
        t = (i+1) * dim
        if i == fraction - 1:
            t += rest
        fraction_embedding = embedding[:, s:t]
        if fraction_embedding.sum().item() == 0:
            continue
        nonzero = torch.nonzero(fraction_embedding, as_tuple=True)[1]
        fraction_embedding = fraction_embedding[:, nonzero]
        fraction_r_embedding = r_embedding[i].to_dense()[nonzero, :].unsqueeze(0)

        # fraction_r_embedding = torch.log(fraction_r_embedding + myglobal.FS_PARAMS.Epi) - torch.log(torch.ones_like(fraction_r_embedding) * myglobal.FS_PARAMS.Epi)
        fraction_r_embedding = fraction_r_embedding.to_sparse()
        ind = fraction_r_embedding._indices()
        val = fraction_r_embedding._values()
        val = torch.log(val + myglobal.FS_PARAMS.Epi) - torch.log(torch.ones_like(val) * myglobal.FS_PARAMS.Epi)
        fraction_r_embedding = torch.sparse_coo_tensor(ind, val, fraction_r_embedding.shape,
                                                       device=fraction_r_embedding.device).to_dense()

        fraction_embedding_premax = fraction_r_embedding * fraction_embedding.unsqueeze(-1)

        fraction_embedding, _ = torch.max(fraction_embedding_premax, dim=1)
        new_embedding = torch.maximum(new_embedding, fraction_embedding)

    return new_embedding

def query_1p(entity_embeddings: nn.Module,
             queries: Tensor) -> Tensor:
    s_emb = entity_embeddings(queries[:, 0])
    candidates_emb = entity_embeddings.weight

    assert queries.shape[1] == 2

    r = queries[0, 1].detach().cpu().item()

    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]

    embedding = torch.zeros(batch_size, nb_entities).to(torch.float).cuda()
    embedding.scatter_(-1, queries[:, 0].unsqueeze(-1), 1)

    atom1_scores_2d = relation_projection(embedding=embedding, r_embedding=myglobal.ADJ[r], nentity=nb_entities)

    res = atom1_scores_2d

    if myglobal.GRID:
        noise = torch.rand_like(res) - 0.5
        res = (res + noise).clamp(min=0., max=myglobal.FS_PARAMS.Max_score)

    return res


def query_2p(entity_embeddings: nn.Module,
             queries: Tensor) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]

    r1 = queries[0, 1].detach().cpu().item()
    r2 = queries[0, 2].detach().cpu().item()

    embedding = torch.zeros(batch_size, nb_entities).to(torch.float).cuda()
    embedding.scatter_(-1, queries[:, 0].unsqueeze(-1), 1)

    atom1_scores_2d = relation_projection(embedding=embedding, r_embedding=myglobal.ADJ[r1], nentity=nb_entities)

    atom2_scores_2d = relation_projection(embedding=atom1_scores_2d, r_embedding=myglobal.ADJ[r2], nentity=nb_entities)

    res = atom2_scores_2d

    return res

def query_3p(entity_embeddings: nn.Module,
             queries: Tensor) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]

    r1 = queries[0, 1].detach().cpu().item()
    r2 = queries[0, 2].detach().cpu().item()
    r3 = queries[0, 3].detach().cpu().item()

    embedding = torch.zeros(batch_size, nb_entities).to(torch.float).cuda()
    embedding.scatter_(-1, queries[:, 0].unsqueeze(-1), 1)

    atom1_scores_2d = relation_projection(embedding=embedding, r_embedding=myglobal.ADJ[r1], nentity=nb_entities)

    atom2_scores_2d = relation_projection(embedding=atom1_scores_2d, r_embedding=myglobal.ADJ[r2], nentity=nb_entities)

    atom3_scores_2d = relation_projection(embedding=atom2_scores_2d, r_embedding=myglobal.ADJ[r3], nentity=nb_entities)

    res = atom3_scores_2d

    return res


def query_2i(entity_embeddings: nn.Module,
             queries: Tensor) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 0:2])
    scores_2 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 2:4])

    # Branch-level per-relation shift
    if myglobal.FS_PARAMS.use_rel_memb:
        rel_memb = myglobal.FS_PARAMS.REL_MEMB
        shift_1 = rel_memb.get_shift_for_rel(queries[:, 1])
        shift_2 = rel_memb.get_shift_for_rel(queries[:, 3])
        scores_1 = scores_1 - shift_1.unsqueeze(1)
        scores_2 = scores_2 - shift_2.unsqueeze(1)

    M = torch.maximum(scores_1, scores_2)
    m = torch.minimum(scores_1, scores_2)

    a = torch.FloatTensor(list(myglobal.FS_PARAMS.ALPHA_I_L)).unsqueeze(-1).cuda()

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
             queries: Tensor) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 0:2])
    scores_2 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 2:4])
    scores_3 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 4:6])

    # Branch-level per-relation shift
    if myglobal.FS_PARAMS.use_rel_memb:
        rel_memb = myglobal.FS_PARAMS.REL_MEMB
        shift_1 = rel_memb.get_shift_for_rel(queries[:, 1])
        shift_2 = rel_memb.get_shift_for_rel(queries[:, 3])
        shift_3 = rel_memb.get_shift_for_rel(queries[:, 5])
        scores_1 = scores_1 - shift_1.unsqueeze(1)
        scores_2 = scores_2 - shift_2.unsqueeze(1)
        scores_3 = scores_3 - shift_3.unsqueeze(1)

    scores_tensor = torch.vstack([scores_1, scores_2, scores_3])
    sorted_scores = torch.sort(scores_tensor, dim=0, descending=True).values

    M = sorted_scores[0]
    A = sorted_scores[1]
    m = sorted_scores[2]

    (a_1, a_2, a_3) = myglobal.FS_PARAMS.ALPHA_I_L

    res_1 = (((1. - a_1) * M + a_1 * A) + ((1. - a_1) * M + a_1 * m) + ((1. - a_1) * A + a_1 * m))/3
    
    res_2 = (((1. - a_2) * M + a_2 * A) + ((1. - a_2) * M + a_2 * m) + ((1. - a_2) * A + a_2 * m))/3
    
    res_3 = (((1. - a_3) * M + a_3 * A) + ((1. - a_3) * M + a_3 * m) + ((1. - a_3) * A + a_3 * m))/3

    res_tensor = torch.vstack([res_1, res_2, res_3])

    # Relation-aware: use first branch relation for combined scores
    mu_S1, mu_B1, mu_W1 = cal_membership3(res_tensor[0], queries[:, 1], 'combined')
    mu_S3, mu_B3, mu_W3 = cal_membership3(res_tensor[2], queries[:, 1], 'combined')
    w_1 = mu_S1
    w_2 = torch.ones_like(w_1)
    w_3 = 1.0 - mu_S3

    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res


def query_ip(entity_embeddings: nn.Module,
             queries: Tensor) -> Tensor:
    
    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    # [B, N]
    scores_1 = query_2i(entity_embeddings=entity_embeddings, queries=queries[:, 0:4])

    r = queries[0, 4].detach().cpu().item()

    scores_2 = relation_projection(embedding=scores_1, r_embedding=myglobal.ADJ[r], nentity=nb_entities)

    res = scores_2

    return res


def query_pi(entity_embeddings: nn.Module,
             queries: Tensor) -> Tensor:

    scores_1 = query_2p(entity_embeddings=entity_embeddings, queries=queries[:, 0:3])
    scores_2 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 3:5])

    M1 = torch.max(scores_1)
    M2 = torch.max(scores_2)
    scores_1 = scores_1 * M2/M1

    # Branch-level per-relation shift for 1p branch only (skip path branch)
    if myglobal.FS_PARAMS.use_rel_memb:
        shift_2 = myglobal.FS_PARAMS.REL_MEMB.get_shift_for_rel(queries[:, 4])
        scores_2 = scores_2 - shift_2.unsqueeze(1)

    M = torch.maximum(scores_1, scores_2)
    m = torch.minimum(scores_1, scores_2)

    a = torch.FloatTensor(list(myglobal.FS_PARAMS.ALPHA_I_L)).unsqueeze(-1).cuda()

    res_tensor = (1. - a) * M + a * m

    # Relation-aware: use first branch relation
    mu_S1, mu_B1, mu_W1 = cal_membership3(res_tensor[0], queries[:, 1], 'combined')
    mu_S3, mu_B3, mu_W3 = cal_membership3(res_tensor[2], queries[:, 1], 'combined')
    w_1 = mu_S1
    w_2 = torch.ones_like(w_1)
    w_3 = 1.0 - mu_S3

    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res


def query_2u_dnf(entity_embeddings: nn.Module,
                 queries: Tensor) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 0:2])
    scores_2 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 2:4])

    # Branch-level per-relation shift
    if myglobal.FS_PARAMS.use_rel_memb:
        rel_memb = myglobal.FS_PARAMS.REL_MEMB
        shift_1 = rel_memb.get_shift_for_rel(queries[:, 1])
        shift_2 = rel_memb.get_shift_for_rel(queries[:, 3])
        scores_1 = scores_1 - shift_1.unsqueeze(1)
        scores_2 = scores_2 - shift_2.unsqueeze(1)

    MM = myglobal.FS_PARAMS.Max_score
    M = torch.maximum(scores_1, scores_2)
    m = torch.minimum(scores_1, scores_2)

    a = torch.FloatTensor(list(myglobal.FS_PARAMS.ALPHA_U_L)).unsqueeze(-1).cuda()

    res_tensor = (1. - a) * MM + a * M

    # Relation-aware: use first branch relation for union
    mu_S, mu_B, mu_W = cal_membership3(M, queries[:, 1], 'union')
    w_1 = mu_S
    w_2 = w_1
    w_3 = torch.ones_like(w_1)

    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    if myglobal.GRID:
        mask = 1. - (scores_1 < 4.7).float() * (scores_2 < 4.7).float()
        res = res * mask

    return res


def query_up_dnf(entity_embeddings: nn.Module,
                 queries: Tensor) -> Tensor:
    # [B, N]
    scores_1 = query_2u_dnf(entity_embeddings=entity_embeddings, queries=queries[:, 0:4])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    r = queries[0, 5].detach().cpu().item()

    scores_2 = relation_projection(embedding=scores_1, r_embedding=myglobal.ADJ[r], nentity=nb_entities)

    res = scores_2

    return res


# (('e', ('r',)), ('e', ('r', 'n'))): '2in'
def query_2in(entity_embeddings: nn.Module,
                 queries: Tensor) -> Tensor:
    
    scores_1 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 0:2])

    scores_2 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 2:4])

    # Three-value relation-aware negation
    mu_S, mu_B, mu_W = cal_membership3(scores_2, queries[:, 3], 'negation_raw', query_type='2in')
    neg_penalty = 1.0 * mu_S + 0.5 * mu_B + 0.0 * mu_W
    r_2 = 1.0 - neg_penalty

    (a_1, a_2, a_3) = myglobal.FS_PARAMS.ALPHA_N

    res_1 = ((1 - a_1) + a_1 * r_2) * scores_1
    res_2 = ((1 - a_2) + a_2 * r_2) * scores_1
    res_3 = ((1 - a_3) + a_3 * r_2) * scores_1

    w_1 = torch.ones_like(res_1)

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
             queries: Tensor) -> Tensor:

    scores_12 = query_2i(entity_embeddings=entity_embeddings, queries=queries[:, 0:4])

    scores_3 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 4:6])

    # Three-value relation-aware negation for third branch
    mu_S, mu_B, mu_W = cal_membership3(scores_3, queries[:, 5], 'negation_raw', query_type='3in')
    neg_penalty = 1.0 * mu_S + 0.5 * mu_B + 0.0 * mu_W
    r_3 = 1.0 - neg_penalty

    (a_1, a_2, a_3) = myglobal.FS_PARAMS.ALPHA_N

    res_1 = ((1 - a_1) + a_1 * r_3) * scores_12
    res_2 = ((1 - a_2) + a_2 * r_3) * scores_12
    res_3 = ((1 - a_3) + a_3 * r_3) * scores_12

    w_1 = torch.ones_like(res_1)

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
             queries: Tensor) -> Tensor:

    scores_1 = query_2in(entity_embeddings=entity_embeddings, queries=queries[:, 0:5])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    r = queries[0, 5].detach().cpu().item()

    scores_2 = relation_projection(embedding=scores_1, r_embedding=myglobal.ADJ[r], nentity=nb_entities)

    res = scores_2

    return res

# (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin'
def query_pin(entity_embeddings: nn.Module,
             queries: Tensor) -> Tensor:

    scores_1 = query_2p(entity_embeddings=entity_embeddings, queries=queries[:, 0:3])
    scores_2 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 3:5])

    # Three-value relation-aware negation
    mu_S, mu_B, mu_W = cal_membership3(scores_2, queries[:, 4], 'negation_raw', query_type='pin')
    neg_penalty = 1.0 * mu_S + 0.5 * mu_B + 0.0 * mu_W
    r_2 = 1.0 - neg_penalty

    (a_1, a_2, a_3) = myglobal.FS_PARAMS.ALPHA_N

    res_1 = ((1 - a_1) + a_1 * r_2) * scores_1
    res_2 = ((1 - a_2) + a_2 * r_2) * scores_1
    res_3 = ((1 - a_3) + a_3 * r_2) * scores_1

    w_1 = torch.ones_like(res_1)

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
             queries: Tensor) -> Tensor:

    scores_1 = query_2p(entity_embeddings=entity_embeddings, queries=queries[:, 0:3])
    
    scores_2 = query_1p(entity_embeddings=entity_embeddings, queries=queries[:, 4:6])

    M1 = torch.max(scores_1)
    M2 = torch.max(scores_2)
    scores_1 = scores_1 * M2/M1 

    # Three-value relation-aware negation for path branch
    mu_S, mu_B, mu_W = cal_membership3(scores_1, queries[:, 1], 'negation_path', query_type='pni')
    neg_penalty = 1.0 * mu_S + 0.5 * mu_B + 0.0 * mu_W
    r_1 = 1.0 - neg_penalty

    (a_1, a_2, a_3) = myglobal.FS_PARAMS.ALPHA_N

    res_1 = ((1 - a_1) + a_1 * r_1) * scores_2
    res_2 = ((1 - a_2) + a_2 * r_1) * scores_2
    res_3 = ((1 - a_3) + a_3 * r_1) * scores_2

    w_1 = torch.ones_like(res_1)

    mu_S21, mu_B21, mu_W21 = cal_membership3(res_2/(1 - a_2), queries[:, 1], 'combined')
    mu_S22, mu_B22, mu_W22 = cal_membership3(res_2, queries[:, 1], 'combined')
    w_2 = mu_S21 + (1.0 - mu_S22)

    w_3 = torch.ones_like(res_3)

    res_tensor = torch.vstack([res_1, res_2, res_3])
    w_tensor = torch.vstack([w_1, w_2, w_3])

    res = defuzzification(w_tensor, res_tensor)

    return res
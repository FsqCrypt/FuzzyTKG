# -*- coding: utf-8 -*-
"""
Relation-Aware Membership Module (Score-Shift Approach).

Core idea: different relations have different score distributions.
Instead of replacing the well-calibrated global B-spline/sigmoid membership
function, we apply a per-relation score SHIFT before evaluating membership.

For each relation r, we compute from validation 1p queries:
  tau_s_r = 20th percentile of positive (answer) scores

Then:
  shift_r = clamp(tau_s_r - global_tau_s, -MAX_SHIFT, +MAX_SHIFT)

During inference for negation queries:
  effective_score = score - shift_r
  mu = original_cal_membership(effective_score)

This means:
  - Hard relations (low tau_s_r, negative shift): scores shifted up →
    membership increases → negation penalty increases → more conservative
  - Easy relations (high tau_s_r, positive shift): scores shifted down →
    membership decreases → negation penalty decreases → more aggressive

Fallback: relation-specific -> cluster (Easy/Medium/Hard) -> global (shift=0)
"""

import torch
import numpy as np
import yaml
import os
import pickle
import logging
from collections import defaultdict
from tqdm import tqdm

import src.myglobal as myglobal


class RelationMembership:
    """Per-relation score-shift and three-value offset for relation-aware membership.

    Two complementary mechanisms:
      Shift  (BEFORE B-spline): effective_score = score - shift_r
             Adjusts for per-relation score-scale differences using tau_s_r.
      Offset (AFTER B-spline):  mu_B = min(offset_r, m), mu_S = m - mu_B
             Fractional penalty: reduces effective penalty for unreliable relations
             using gap_r = tau_s_r - tau_w_r (pos/neg separability).
    """

    def __init__(self, nrelation, params_dict):
        self.nrelation = nrelation

        # ---- Shift params (before B-spline) ----
        self.global_shift = 0.0

        self.cluster_shifts = []
        for c in params_dict.get('clusters', []):
            self.cluster_shifts.append(float(c.get('shift', 0.0)))

        self.shifts = np.full(nrelation, np.nan, dtype=np.float32)
        self.cluster_id = np.full(nrelation, -1, dtype=np.int32)

        # ---- Offset params (after B-spline, three-value) ----
        self.global_offset = float(params_dict.get('global_offset', 0.0))

        self.cluster_offsets = []
        for c in params_dict.get('clusters', []):
            self.cluster_offsets.append(float(c.get('offset', 0.0)))

        self.offsets = np.full(nrelation, np.nan, dtype=np.float32)

        for rel_id_str, rparams in params_dict.get('relations', {}).items():
            rid = int(rel_id_str)
            if rid < nrelation:
                if rparams.get('shift') is not None:
                    self.shifts[rid] = float(rparams['shift'])
                if rparams.get('offset') is not None:
                    self.offsets[rid] = float(rparams['offset'])
                if rparams.get('cluster') is not None:
                    self.cluster_id[rid] = int(rparams['cluster'])

        # GPU tensors (lazy init)
        self._shifts_t = None
        self._offsets_t = None
        self._cluster_shifts_t = None
        self._cluster_offsets_t = None
        self._cluster_id_t = None
        self._device = None

    def _ensure_tensors(self, device):
        if self._device == device:
            return
        self._device = device
        self._shifts_t = torch.from_numpy(self.shifts).to(device)
        self._offsets_t = torch.from_numpy(self.offsets).to(device)
        self._cluster_shifts_t = torch.tensor(self.cluster_shifts, device=device)
        self._cluster_offsets_t = torch.tensor(self.cluster_offsets, device=device)
        self._cluster_id_t = torch.from_numpy(self.cluster_id).long().to(device)
        self._global_shift_t = torch.tensor(self.global_shift, device=device)
        self._global_offset_t = torch.tensor(self.global_offset, device=device)

    def _get_shift_for_rel(self, rel_ids):
        """Return per-relation shift [B], with cluster/global fallback."""
        device = rel_ids.device
        self._ensure_tensors(device)

        shift = self._shifts_t[rel_ids]  # [B]

        # Fallback 1: cluster
        nan_mask = torch.isnan(shift)
        if nan_mask.any():
            cid = self._cluster_id_t[rel_ids]
            valid = (cid >= 0) & (cid < len(self.cluster_shifts))
            fb_mask = nan_mask & valid
            if fb_mask.any():
                shift[fb_mask] = self._cluster_shifts_t[cid[fb_mask]]

        # Fallback 2: global (shift=0)
        still_nan = torch.isnan(shift)
        if still_nan.any():
            shift[still_nan] = self._global_shift_t

        return shift

    def get_shift_for_rel(self, rel_ids):
        """Public wrapper for per-relation shift lookup.

        Returns shift [B] for each relation, with cluster/global fallback.
        Used by query functions for branch-level score calibration before
        fuzzy operators (max/min/product).
        """
        return self._get_shift_for_rel(rel_ids)

    def _get_offset_for_rel(self, rel_ids):
        """Return per-relation offset [B], with cluster/global fallback.

        Offset controls the fractional-penalty three-value decomposition:
          mu_B = min(offset_r, m)   — uncertain portion, half-penalty
          mu_S = m - mu_B           — confident portion, full penalty
          penalty = m - 0.5*mu_B < m  (when offset_r > 0)

        Fallback: per-relation → cluster → global (0 = two-value mode).
        A cluster offset of 0 means "undefined for this cluster" and falls
        through to global, preventing cluster defaults from blocking three-value.
        """
        device = rel_ids.device
        self._ensure_tensors(device)

        offset = self._offsets_t[rel_ids]  # [B]

        # Fallback 1: cluster
        nan_mask = torch.isnan(offset)
        if nan_mask.any():
            cid = self._cluster_id_t[rel_ids]
            valid = (cid >= 0) & (cid < len(self.cluster_offsets))
            fb_mask = nan_mask & valid
            if fb_mask.any():
                cluster_vals = self._cluster_offsets_t[cid[fb_mask]]
                offset[fb_mask] = cluster_vals
                # cluster offset=0 means undefined → fall through to global
                zero_cluster = (cluster_vals == 0.0)
                if zero_cluster.any():
                    fb_indices = fb_mask.nonzero(as_tuple=True)[0]
                    zi = fb_indices[zero_cluster]
                    offset[zi] = float('nan')

        # Fallback 2: global (default 0 = no three-value decomposition)
        still_nan = torch.isnan(offset)
        if still_nan.any():
            offset[still_nan] = self._global_offset_t

        return offset

    # Per-query-type offset scaling factors.
    # Same relation may need different offset strength depending on how
    # many positive branches back up the negation in the query structure.
    #  3in (2 pos + 1 neg): full offset → scale 1.0
    #  2in (1 pos + 1 neg): reduced offset → scale 0.3
    #  pin (path pos + 1 neg): minimal offset → scale 0.2
    #  pni (negated path): no offset (uses negation_path, not reached)
    QUERY_TYPE_SCALES = {
        '3in': 1.0,
        '2in': 0.2,
        'pin': 0.1,
        'pni': 0.0,
        None: 1.0,   # backward compat when query_type not provided
    }

    # Context-dependent beta coefficient for three-value decomposition.
    # Beta controls how strongly the uncertain portion (mu_B) is discounted
    # from the confident portion (mu_S) in different call-site contexts.
    #   beta = 1.0: no discount — mu_S = m           (original binary)
    #   beta = 0.5: half discount — mu_S = m - 0.5*mu_B  (negation default)
    #   beta = 0.0: full discount — mu_S = m - mu_B  (mu_B fully stripped)
    CONTEXT_BETA = {
        'negation_raw':   0.5,   # caller uses mu_S + 0.5*mu_B → penalty
        'negation_path':  0.0,   # no three-value for path negation
        'combined':       0.3,   # conservative discount for defuzz weights
        'union':          0.4,   # moderate discount for union defuzz weights
    }

    def cal_membership3(self, scores, rel_ids, call_site_type='combined',
                        query_type=None):
        """
        Compute three-value membership with per-relation score shift
        and per-query-type offset scaling.

        For 'negation_raw': applies per-relation shift before evaluating
        the original global membership, then decomposes into three values
        with offset scaled by query_type factor.

        For 'combined'/'union': returns original membership without shift.
        """
        device = scores.device

        # Normalize rel_ids
        if not isinstance(rel_ids, torch.Tensor):
            rel_ids = torch.tensor([rel_ids], device=device)
        if rel_ids.dim() == 0:
            rel_ids = rel_ids.unsqueeze(0)

        original_shape = scores.shape
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        B_score = scores.shape[0]
        B_rel = rel_ids.shape[0]
        if B_rel == 1 and B_score > 1:
            rel_ids = rel_ids.expand(B_score)

        # Apply per-relation score shift.
        # 'negation_raw': full shift + three-value offset + query-type scaling.
        #   The negated branch's raw 1p scores are shifted here.
        # 'combined'/'union': NO shift — branch-level shift already applied
        #   in the query function before score combination (max/min/product).
        # 'negation_path': no shift — path scores have different distributions.
        if call_site_type == 'negation_raw':
            shift = self._get_shift_for_rel(rel_ids)
            shift = torch.clamp(shift, min=-3.0, max=3.0)
            if shift.dim() == 1:
                shift = shift.unsqueeze(1)
            shifted_scores = scores - shift
        elif call_site_type == 'negation_path':
            shifted_scores = scores  # no shift for path-based negation
        else:
            # combined / union: branch-level shift already applied
            shifted_scores = scores

        # Evaluate original global membership function on shifted scores
        # Import here to avoid circular import
        from src.discrete_cqd import cal_membership as cal_mem_cqd
        from src.discrete_qto import cal_membership as cal_mem_qto

        if myglobal.FS_PARAMS.BACKBONE == 'QTO':
            m = cal_mem_qto(shifted_scores)
        else:
            m = cal_mem_cqd(shifted_scores)

        # Three-value decomposition:
        #   negation_raw:  fractional penalty via offset → true 3-value
        #     mu_S = m - mu_B, mu_B = min(offset, m)
        #     caller computes: penalty = 1.0*mu_S + 0.5*mu_B = m - 0.5*mu_B
        #   combined/union: uncertainty-aware weight discount via context beta
        #     mu_B_raw = min(offset*0.5, m), mu_S = m - (1-beta)*mu_B_raw
        #     caller uses mu_S as defuzzification weight (mu_B informational)
        #   negation_path: binary decomposition (mu_B=0)
        beta = self.CONTEXT_BETA.get(call_site_type, 0.0)

        if call_site_type == 'negation_raw':
            # Fractional Penalty: reclassify up to offset_r of membership
            # from Strong (full penalty) to Boundary (half penalty).
            # penalty = mu_S + 0.5*mu_B = m - 0.5*mu_B < m  (when offset>0)
            #
            # Per-query-type scaling: same relation needs different offset
            # strength depending on query structure (Module 2 entry point).
            offset_r = self._get_offset_for_rel(rel_ids)
            scale_q = self.QUERY_TYPE_SCALES.get(query_type, 1.0)
            offset_r = offset_r * scale_q
            offset_r = torch.clamp(offset_r, min=0.0, max=0.5)
            if offset_r.dim() == 1 and m.dim() > 1:
                offset_r = offset_r.unsqueeze(1)
            mu_B = torch.minimum(offset_r, m)
            mu_S = m - mu_B
            mu_W = 1.0 - m
        elif call_site_type in ('combined', 'union') and beta > 0:
            # Uncertainty-aware weight discount: for unreliable relations,
            # discount the membership used as defuzzification weight.
            #   mu_S = m - (1-beta)*mu_B_raw  (beta=0.3 → discount 0.7*mu_B_raw)
            # Smaller offset than negation (×0.5) — defuzz weights are more
            # sensitive to membership changes than negation penalties.
            offset_r = self._get_offset_for_rel(rel_ids)
            offset_r = offset_r * 0.5  # reduced magnitude for defuzz context
            offset_r = torch.clamp(offset_r, min=0.0, max=0.5)
            if offset_r.dim() == 1 and m.dim() > 1:
                offset_r = offset_r.unsqueeze(1)
            mu_B_raw = torch.minimum(offset_r, m)
            mu_S = m - (1.0 - beta) * mu_B_raw   # e.g. m - 0.7*mu_B_raw
            mu_B = mu_B_raw                       # informational
            mu_W = 1.0 - m
        else:
            # Binary decomposition (negation_path, or beta=0)
            zero = torch.zeros_like(m)
            mu_S = m
            mu_B = zero
            mu_W = 1.0 - m

        # Restore original shape
        if len(original_shape) == 1:
            mu_S = mu_S.squeeze(0)
            mu_B = mu_B.squeeze(0)
            mu_W = mu_W.squeeze(0)

        return mu_S, mu_B, mu_W


# ---------------------------------------------------------------------------
# Statistics computation and param generation
# ---------------------------------------------------------------------------

MAX_SHIFT = 3.0  # Maximum absolute score shift per relation


def compute_per_relation_stats(data_path, checkpoint_path, dataname,
                                backbone_type='CQD', memb_type='bspline',
                                cuda=True, output_dir='./params',
                                neg_sample_size=100):
    """
    Compute per-relation score shifts from validation 1p queries.
    Saves params YAML with per-relation shift values.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from src.base import CQD
    import src.discrete as d1

    logging.info(f'Computing per-relation stats for {dataname} ({backbone_type})...')

    # Load dataset stats
    with open(os.path.join(data_path, 'stats.txt')) as f:
        lines = f.readlines()
        nentity = int(lines[0].split(' ')[-1])
        nrelation = int(lines[1].split(' ')[-1])

    # Load validation 1p queries
    valid_queries = pickle.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
    valid_answers = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
    query_structure_1p = ('e', ('r',))
    if query_structure_1p not in valid_queries:
        logging.error('No 1p queries in validation set!')
        return None
    queries_1p = list(valid_queries[query_structure_1p])
    logging.info(f'Found {len(queries_1p)} validation 1p queries')

    # Setup model
    query_name_dict = {query_structure_1p: '1p'}
    model = CQD(nentity, nrelation, rank=1000, test_batch_size=1,
                method='discrete', t_norm_name='prod', k=1024,
                query_name_dict=query_name_dict, wo_fl=False,
                backbone_type=backbone_type, use_cuda=cuda)
    myglobal.FS_PARAMS.set_static_fs_params(
        backbone=backbone_type, memb_type=memb_type, dataname=dataname.lower())
    myglobal.FS_PARAMS.load_membership_params(dataname)

    checkpoint = torch.load(os.path.join(checkpoint_path, 'checkpoint'),
                            map_location='cpu' if not cuda else None)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if cuda:
        model = model.cuda()
    model.eval()

    entity_embeddings = model.embeddings[0]
    predicate_embeddings = model.embeddings[1]

    # Group queries by relation (handle BetaE format: query is (entity, (rel,)))
    rel_queries = defaultdict(list)
    for q in queries_1p:
        r = q[1][0] if isinstance(q[1], tuple) else q[1]
        rel_queries[r].append(q)

    # Collect per-relation pos and neg score statistics
    pos_scores_r = defaultdict(list)
    neg_scores_r = defaultdict(list)
    MAX_PER_REL = 100  # max queries per relation

    # Build global answer lookup for neg sampling
    answer_sets = {}
    for q, ans in valid_answers.items():
        answer_sets[q] = set(ans) if not isinstance(ans, set) else ans

    with torch.no_grad():
        for r, r_queries in tqdm(rel_queries.items(), desc='Collecting scores'):
            if len(r_queries) > MAX_PER_REL:
                indices = np.random.choice(len(r_queries), MAX_PER_REL, replace=False)
                r_queries = [r_queries[i] for i in indices]

            for q in r_queries:
                h = q[0]
                rid = q[1][0] if isinstance(q[1], tuple) else q[1]
                answers = answer_sets.get(q, set())

                query_t = torch.LongTensor([[h, rid]])
                if cuda:
                    query_t = query_t.cuda()

                raw_scores = d1.query_1p(
                    entity_embeddings=entity_embeddings,
                    predicate_embeddings=predicate_embeddings,
                    queries=query_t,
                    scoring_function=lambda rel, lhs, rhs: model.score_o(lhs, rel, rhs)[0]
                ).squeeze(0)

                # Pos scores: answer entities
                for ans in answers:
                    if ans < raw_scores.shape[0]:
                        pos_scores_r[r].append(raw_scores[ans].item())

                # Neg scores: sample non-answer entities
                answers_set = set(int(a) for a in answers if a < raw_scores.shape[0])
                neg_pool = [i for i in range(raw_scores.shape[0])
                            if i not in answers_set]
                if len(neg_pool) > neg_sample_size:
                    neg_samples = np.random.choice(neg_pool, neg_sample_size, replace=False)
                else:
                    neg_samples = neg_pool
                for neg_e in neg_samples:
                    neg_scores_r[r].append(raw_scores[int(neg_e)].item())

    # Compute global tau_s (20th percentile of all pos scores)
    all_pos = np.concatenate([np.array(v) for v in pos_scores_r.values() if len(v) > 0])
    global_tau_s = float(np.percentile(all_pos, 20))
    logging.info(f'Global tau_s (pos 20pct): {global_tau_s:.4f}')

    # Per-relation shift
    rel_params = {}
    all_shifts = []
    for r in range(nrelation):
        pos_list = pos_scores_r.get(r, [])
        n_pos = len(pos_list)
        if n_pos >= 10:
            tau_s_r = float(np.percentile(pos_list, 20))
            shift_r = np.clip(tau_s_r - global_tau_s, -MAX_SHIFT, MAX_SHIFT)
            rel_params[r] = {
                'shift': round(float(shift_r), 4),
                'tau_s': round(tau_s_r, 4),
                'n_samples': n_pos,
            }
            all_shifts.append(shift_r)
        else:
            rel_params[r] = {
                'shift': None,
                'tau_s': None,
                'n_samples': n_pos,
            }

    # Compute per-relation tau_w (neg score 90th pct) and gap
    all_neg = np.concatenate([np.array(v) for v in neg_scores_r.values() if len(v) > 0])
    global_tau_w = float(np.percentile(all_neg, 90))
    logging.info(f'Global tau_w (neg 90pct): {global_tau_w:.4f}')

    gaps = []
    for r in range(nrelation):
        neg_list = neg_scores_r.get(r, [])
        n_neg = len(neg_list)
        if n_neg >= 10:
            tau_w_r = float(np.percentile(neg_list, 90))
            rel_params[r]['tau_w'] = round(tau_w_r, 4)
            # gap = tau_s - tau_w: how well pos/neg scores separate
            if rel_params[r]['tau_s'] is not None:
                gap_r = max(rel_params[r]['tau_s'] - tau_w_r, 0.01)
                rel_params[r]['gap'] = round(float(gap_r), 4)
                gaps.append(gap_r)
        else:
            rel_params[r]['tau_w'] = None
            rel_params[r]['gap'] = None

    # Compute per-relation offset from gap (three-value Fractional Penalty)
    if len(gaps) > 0:
        gaps_arr = np.array(gaps)
        GAP_REF = float(np.percentile(gaps_arr, 90))
        OFFSET_MIN = 0.01
        OFFSET_MAX = 0.08
        for r in range(nrelation):
            gap_r = rel_params[r].get('gap')
            if gap_r is not None and GAP_REF > 0:
                reliability = np.clip(gap_r / GAP_REF, 0, 1)
                offset_r = OFFSET_MIN + (OFFSET_MAX - OFFSET_MIN) * (1 - reliability)
                rel_params[r]['offset'] = round(float(offset_r), 4)
            else:
                rel_params[r]['offset'] = None
        logging.info(f'GAP_REF={GAP_REF:.4f}, offsets: [{OFFSET_MIN}, {OFFSET_MAX}]')

    # Cluster relations
    clusters = []
    rel_ids_with_data = [r for r in range(nrelation) if rel_params[r]['shift'] is not None]

    if len(rel_ids_with_data) >= 9:
        try:
            from sklearn.cluster import KMeans
            feats = np.array([[rel_params[r]['shift'], rel_params[r]['tau_s']]
                              for r in rel_ids_with_data])
            from sklearn.preprocessing import StandardScaler
            feats = StandardScaler().fit_transform(feats)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(feats)

            cluster_shifts = {0: [], 1: [], 2: []}
            for i, rid in enumerate(rel_ids_with_data):
                cid = int(labels[i])
                cluster_shifts[cid].append(rel_params[rid]['shift'])

            sorted_clusters = sorted(cluster_shifts.items(),
                                     key=lambda x: np.mean(x[1]), reverse=True)
            name_map = {old_cid: new_idx for new_idx, (old_cid, _) in enumerate(sorted_clusters)}

            for new_cid in range(3):
                old_cid = sorted_clusters[new_cid][0]
                clusters.append({
                    'id': new_cid,
                    'shift': round(float(np.mean(cluster_shifts[old_cid])), 4),
                    'n_relations': len(cluster_shifts[old_cid]),
                })

            for i, rid in enumerate(rel_ids_with_data):
                rel_params[rid]['cluster'] = name_map[int(labels[i])]
        except ImportError:
            logging.warning('sklearn not available, skipping clustering')
    else:
        logging.warning(f'Not enough relations for clustering ({len(rel_ids_with_data)})')

    if not clusters:
        clusters = [{'id': i, 'shift': 0.0, 'n_relations': 0} for i in range(3)]

    for r in range(nrelation):
        if 'cluster' not in rel_params[r]:
            rel_params[r]['cluster'] = 1  # default: Medium

    # Save YAML
    os.makedirs(output_dir, exist_ok=True)
    output_fn = os.path.join(output_dir, f'{dataname}_{backbone_type}_rel_memb_params.yml')

    with open(output_fn, 'w') as f:
        f.write(f'# Relation-aware membership params for {dataname} ({backbone_type})\n')
        f.write(f'# Dual-layer: Score-Shift + Fractional Penalty Offset\n')
        f.write(f'# Generated from validation 1p queries\n\n')
        f.write(f'nrelation: {nrelation}\n')
        f.write(f'global_tau_s: {global_tau_s:.4f}\n')
        f.write(f'global_tau_w: {global_tau_w:.4f}\n')
        f.write(f'global_offset: 0.0\n\n')
        f.write(f'clusters:\n')
        for c in clusters:
            f.write(f'  - id: {c["id"]}\n')
            f.write(f'    shift: {c["shift"]}\n')
            f.write(f'    offset: 0.0\n')
            f.write(f'    n_relations: {c["n_relations"]}\n')
        f.write(f'\nrelations:\n')
        for r in range(nrelation):
            rp = rel_params[r]
            s_s = 'null' if rp['shift'] is None else rp['shift']
            ts_s = 'null' if rp['tau_s'] is None else rp['tau_s']
            tw_s = 'null' if rp.get('tau_w') is None else rp['tau_w']
            gap_s = 'null' if rp.get('gap') is None else rp['gap']
            off_s = 'null' if rp.get('offset') is None else rp['offset']
            f.write(f'  {r}: {{shift: {s_s}, tau_s: {ts_s}, tau_w: {tw_s}, '
                    f'gap: {gap_s}, offset: {off_s}, '
                    f'cluster: {rp["cluster"]}, n_samples: {rp["n_samples"]}}}\n')

    n_with = sum(1 for rp in rel_params.values() if rp['shift'] is not None)
    logging.info(f'Saved to {output_fn} ({n_with}/{nrelation} relations with data)')
    return rel_params


def create_default_params(nrelation, backbone_type, dataname, output_dir='./params'):
    """Create default params with shift=0 for all relations (no relation awareness)."""
    os.makedirs(output_dir, exist_ok=True)
    output_fn = os.path.join(output_dir, f'{dataname}_{backbone_type}_rel_memb_params.yml')

    with open(output_fn, 'w') as f:
        f.write(f'# Default relation-aware membership params for {dataname} ({backbone_type})\n')
        f.write(f'# All shifts = 0 (no relation awareness) — run compute_rel_stats.py\n\n')
        f.write(f'nrelation: {nrelation}\n')
        f.write(f'global_tau_s: 6.0\n\n')
        f.write(f'clusters:\n')
        for i in range(3):
            f.write(f'  - id: {i}\n    shift: 0.0\n    n_relations: 0\n')
        f.write(f'\nrelations:\n')
        for r in range(nrelation):
            f.write(f'  {r}: {{shift: null, tau_s: null, cluster: 1, n_samples: 0}}\n')

    logging.info(f'Default params: {output_fn}')

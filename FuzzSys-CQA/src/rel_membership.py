# -*- coding: utf-8 -*-
"""
Relation-Aware Three-Value Membership Module.

For each relation r, we estimate two thresholds from validation 1p query scores:
  tau_w_r: boundary between Weak and Boundary (90th percentile of neg scores)
  tau_s_r: boundary between Boundary and Strong (20th percentile of pos scores)

Then for any score s and relation r:
  raw_S = sigmoid((s - tau_s_r) / T_r)
  raw_W = sigmoid((tau_w_r - s) / T_r)
  raw_B = exp(-(s - center_r)^2 / (2 * width_r^2))
  Z = raw_S + raw_B + raw_W
  mu_S, mu_B, mu_W = raw_S/Z, raw_B/Z, raw_W/Z

Fallback chain: relation-specific -> cluster (Easy/Medium/Hard) -> global
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
    """Per-relation three-value (Strong/Boundary/Weak) membership parameters."""

    def __init__(self, nrelation, params_dict):
        self.nrelation = nrelation
        self.epsilon = 0.1  # minimum width to avoid division by zero

        # Global fallback
        g = params_dict['global']
        self.global_tau_w = float(g['tau_w'])
        self.global_tau_s = float(g['tau_s'])
        self.global_T = float(g['T'])

        # Cluster fallbacks
        self.cluster_tau_w = []
        self.cluster_tau_s = []
        self.cluster_T = []
        for c in params_dict['clusters']:
            self.cluster_tau_w.append(float(c['tau_w']))
            self.cluster_tau_s.append(float(c['tau_s']))
            self.cluster_T.append(float(c['T']))

        # Per-relation parameters
        self.tau_w = np.full(nrelation, np.nan, dtype=np.float32)
        self.tau_s = np.full(nrelation, np.nan, dtype=np.float32)
        self.T = np.full(nrelation, np.nan, dtype=np.float32)
        self.cluster_id = np.full(nrelation, -1, dtype=np.int32)

        for rel_id_str, rparams in params_dict['relations'].items():
            rid = int(rel_id_str)
            if rid < nrelation:
                if rparams.get('tau_w') is not None:
                    self.tau_w[rid] = float(rparams['tau_w'])
                    self.tau_s[rid] = float(rparams['tau_s'])
                    self.T[rid] = float(rparams['T'])
                if rparams.get('cluster') is not None:
                    self.cluster_id[rid] = int(rparams['cluster'])

        # Convert to tensors for fast GPU lookup
        self._tau_w_t = None
        self._tau_s_t = None
        self._T_t = None
        self._cluster_tau_w_t = None
        self._cluster_tau_s_t = None
        self._cluster_T_t = None
        self._cluster_id_t = None
        self._device = None

    def _ensure_tensors(self, device):
        """Lazily create GPU tensors for parameter lookup."""
        if self._device == device:
            return
        self._device = device
        self._tau_w_t = torch.from_numpy(self.tau_w).to(device)
        self._tau_s_t = torch.from_numpy(self.tau_s).to(device)
        self._T_t = torch.from_numpy(self.T).to(device)
        self._cluster_id_t = torch.from_numpy(self.cluster_id).long().to(device)
        self._cluster_tau_w_t = torch.tensor(self.cluster_tau_w, device=device)
        self._cluster_tau_s_t = torch.tensor(self.cluster_tau_s, device=device)
        self._cluster_T_t = torch.tensor(self.cluster_T, device=device)
        self._global_tau_w_t = torch.tensor(self.global_tau_w, device=device)
        self._global_tau_s_t = torch.tensor(self.global_tau_s, device=device)
        self._global_T_t = torch.tensor(self.global_T, device=device)

    def _get_params_for_rel(self, rel_ids):
        """
        Given relation IDs [B], return (tau_w, tau_s, T) per batch element,
        applying the fallback chain: specific -> cluster -> global.
        """
        device = rel_ids.device
        self._ensure_tensors(device)
        B = rel_ids.shape[0]

        # Gather per-relation params
        tau_w_r = self._tau_w_t[rel_ids]  # [B]
        tau_s_r = self._tau_s_t[rel_ids]  # [B]
        T_r = self._T_t[rel_ids]          # [B]

        # Fallback 1: cluster-level for relations without specific params
        nan_mask = torch.isnan(tau_w_r)
        if nan_mask.any():
            cid = self._cluster_id_t[rel_ids]  # [B]
            # Clamp to valid cluster indices
            valid_cluster = (cid >= 0) & (cid < len(self.cluster_tau_w))
            fallback_mask = nan_mask & valid_cluster
            if fallback_mask.any():
                tau_w_r[fallback_mask] = self._cluster_tau_w_t[cid[fallback_mask]]
                tau_s_r[fallback_mask] = self._cluster_tau_s_t[cid[fallback_mask]]
                T_r[fallback_mask] = self._cluster_T_t[cid[fallback_mask]]

        # Fallback 2: global for anything still NaN
        still_nan = torch.isnan(tau_w_r)
        if still_nan.any():
            tau_w_r[still_nan] = self._global_tau_w_t
            tau_s_r[still_nan] = self._global_tau_s_t
            T_r[still_nan] = self._global_T_t

        # Ensure T_r is at least epsilon to avoid division by zero
        T_r = torch.clamp(T_r, min=self.epsilon)

        return tau_w_r, tau_s_r, T_r

    def cal_membership3(self, scores, rel_ids, call_site_type='combined'):
        """
        Compute three-value membership (mu_S, mu_B, mu_W) for given scores and relations.

        Args:
            scores: tensor of shape [B, N] or [B] or [N] — LP scores
            rel_ids: tensor of shape [B] or [1] or scalar — relation IDs
            call_site_type: 'negation_raw', 'combined', or 'union'

        Returns:
            (mu_S, mu_B, mu_W): three tensors of same shape as scores
        """
        device = scores.device

        # Normalize rel_ids to [B] tensor
        if not isinstance(rel_ids, torch.Tensor):
            rel_ids = torch.tensor([rel_ids], device=device)
        if rel_ids.dim() == 0:
            rel_ids = rel_ids.unsqueeze(0)

        B_rel = rel_ids.shape[0]
        original_shape = scores.shape

        # Ensure scores is [B, N] for consistent indexing
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)  # [1, N]

        B_score = scores.shape[0]

        # Broadcast rel_ids if needed: one relation per batch of scores
        if B_rel == 1 and B_score > 1:
            rel_ids = rel_ids.expand(B_score)
            B_rel = B_score
        elif B_rel != B_score:
            # If mismatch, expand the smaller to match
            if B_rel == 1:
                rel_ids = rel_ids.expand(B_score)
            else:
                raise ValueError(f"Cannot broadcast rel_ids shape {rel_ids.shape} "
                                 f"with scores shape {scores.shape}")

        tau_w, tau_s, T_r = self._get_params_for_rel(rel_ids)  # [B]
        center = (tau_w + tau_s) / 2.0
        width = torch.clamp(tau_s - tau_w, min=self.epsilon)

        # Reshape for broadcasting: [B] -> [B, 1]
        tau_w = tau_w.unsqueeze(1)
        tau_s = tau_s.unsqueeze(1)
        T_r = T_r.unsqueeze(1)
        center = center.unsqueeze(1)
        width = width.unsqueeze(1)

        # Compute raw components
        raw_S = torch.sigmoid((scores - tau_s) / T_r)
        raw_W = torch.sigmoid((tau_w - scores) / T_r)
        raw_B = torch.exp(-((scores - center) ** 2) / (2.0 * width ** 2))

        # Softmax normalization
        Z = raw_S + raw_B + raw_W
        Z = torch.clamp(Z, min=1e-8)

        mu_S = raw_S / Z
        mu_B = raw_B / Z
        mu_W = raw_W / Z

        # Restore original shape if scores was 1D
        if len(original_shape) == 1:
            mu_S = mu_S.squeeze(0)
            mu_B = mu_B.squeeze(0)
            mu_W = mu_W.squeeze(0)

        return mu_S, mu_B, mu_W


def compute_per_relation_stats(data_path, checkpoint_path, dataname,
                                backbone_type='CQD', memb_type='bspline',
                                cuda=True, output_dir='./params',
                                neg_sample_size=100):
    """
    Compute per-relation statistics from validation 1p queries.

    Steps:
    1. Load model checkpoint
    2. Iterate validation 1p queries, run inference
    3. Collect per-relation pos_scores and neg_scores
    4. Compute tau_w_r, tau_s_r, T_r for each relation
    5. Cluster relations (Easy/Medium/Hard)
    6. Compute global fallback
    7. Save params YAML

    Args:
        data_path: path to dataset (e.g., '../KG_data/FB15k-betae')
        checkpoint_path: path to model checkpoint directory
        dataname: dataset name (e.g., 'FB15k')
        backbone_type: 'CQD' or 'QTO'
        memb_type: 'bspline' or 'symbolic'
        cuda: use GPU
        output_dir: directory to save params YAML
        neg_sample_size: number of negative samples per query for statistics
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from models import CQD, KGReasoning
    from dataloader import TestDataset
    from torch.utils.data import DataLoader
    from util import flatten_query

    logging.info(f'Computing per-relation stats for {dataname} ({backbone_type})...')

    # Load dataset stats
    with open(os.path.join(data_path, 'stats.txt')) as f:
        lines = f.readlines()
        nentity = int(lines[0].split(' ')[-1])
        nrelation = int(lines[1].split(' ')[-1])

    # Load validation 1p queries and answers
    valid_queries = pickle.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))

    # Only use 1p queries
    query_structure_1p = ('e', ('r',))
    if query_structure_1p not in valid_queries:
        logging.error(f'No 1p queries found in validation set!')
        return

    queries_1p = list(valid_queries[query_structure_1p])
    logging.info(f'Found {len(queries_1p)} validation 1p queries')

    # Create model
    query_name_dict = {query_structure_1p: '1p'}
    if backbone_type == 'QTO':
        # For QTO we need the adjacency matrices
        model = CQD(nentity, nrelation, rank=1000,
                    test_batch_size=1, method='discrete',
                    t_norm_name='prod', k=1024,
                    query_name_dict=query_name_dict,
                    wo_fl=False, backbone_type='QTO',
                    use_cuda=cuda)
        myglobal.FS_PARAMS.set_static_fs_params(
            backbone='QTO', memb_type=memb_type, dataname=dataname.lower())
        # Load adjacency matrices
        adj_path = os.path.join(checkpoint_path.replace('-betae', '-q2b'), '..',
                                dataname.lower() + '-q2b', 'neural_adj.pt')
        if os.path.exists(adj_path):
            myglobal.ADJ = torch.load(adj_path)
    else:
        model = CQD(nentity, nrelation, rank=1000,
                    test_batch_size=1, method='discrete',
                    t_norm_name='prod', k=1024,
                    query_name_dict=query_name_dict,
                    wo_fl=False, backbone_type='CQD',
                    use_cuda=cuda)
        myglobal.FS_PARAMS.set_static_fs_params(
            backbone='CQD', memb_type=memb_type, dataname=dataname.lower())

    # Load membership params (needed for scoring function setup)
    myglobal.FS_PARAMS.load_membership_params(dataname)

    # Load checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_path, 'checkpoint'),
                            map_location=torch.device('cpu') if not cuda else None)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if cuda:
        model = model.cuda()
    model.eval()

    # Collect per-relation scores using the model's scoring function
    # We'll use query_1p from discrete.py (wo_fl path) to get raw scores
    # without fuzzy membership
    import src.discrete as d1

    entity_embeddings = model.embeddings[0]
    predicate_embeddings = model.embeddings[1]

    # Per-relation score accumulators
    pos_scores_r = defaultdict(list)
    neg_scores_r = defaultdict(list)

    with torch.no_grad():
        for query_tuple in tqdm(queries_1p, desc='Collecting per-relation scores'):
            h, r = query_tuple
            answers = valid_hard_answers.get(query_tuple, set())

            # Build query tensor [1, 2]
            query_t = torch.LongTensor([[h, r]])
            if cuda:
                query_t = query_t.cuda()

            # Run 1p inference (raw scores, no membership)
            raw_scores = d1.query_1p(
                entity_embeddings=entity_embeddings,
                predicate_embeddings=predicate_embeddings,
                queries=query_t,
                scoring_function=lambda rel, lhs, rhs: model.score_o(lhs, rel, rhs)[0]
            )  # [1, nentity]

            raw_scores = raw_scores.squeeze(0)  # [nentity]

            # Positive scores
            for ans in answers:
                if ans < raw_scores.shape[0]:
                    pos_scores_r[r].append(raw_scores[ans].item())

            # Negative scores: sample entities not in answers
            all_entities = set(range(nentity))
            neg_candidates = list(all_entities - answers)
            if len(neg_candidates) > neg_sample_size:
                neg_samples = np.random.choice(neg_candidates, size=neg_sample_size, replace=False)
            else:
                neg_samples = neg_candidates
            for neg_e in neg_samples:
                neg_scores_r[r].append(raw_scores[neg_e].item())

    # Compute statistics
    logging.info(f'Computing thresholds for {nrelation} relations...')
    rel_params = {}
    all_pos_means = []
    all_pos_stds = []
    all_neg_means = []
    all_pos_neg_gaps = []
    all_cardinalities = []
    relation_ids_with_samples = []

    for r in range(nrelation):
        pos_list = pos_scores_r.get(r, [])
        neg_list = neg_scores_r.get(r, [])

        n_pos = len(pos_list)

        if n_pos >= 10 and len(neg_list) >= 10:
            pos_arr = np.array(pos_list)
            neg_arr = np.array(neg_list)

            tau_w_r = float(np.percentile(neg_arr, 90))
            tau_s_r = float(np.percentile(pos_arr, 20))
            T_r = float(0.5 * max(tau_s_r - tau_w_r, 0.1))

            rel_params[r] = {
                'tau_w': round(tau_w_r, 4),
                'tau_s': round(tau_s_r, 4),
                'T': round(T_r, 4),
                'n_samples': n_pos,
            }

            pos_mean = float(np.mean(pos_arr))
            pos_std = float(np.std(pos_arr))
            neg_mean = float(np.mean(neg_arr))
            pos_neg_gap = pos_mean - neg_mean

            all_pos_means.append(pos_mean)
            all_pos_stds.append(pos_std)
            all_neg_means.append(neg_mean)
            all_pos_neg_gaps.append(pos_neg_gap)
            all_cardinalities.append(n_pos)
            relation_ids_with_samples.append(r)
        else:
            rel_params[r] = {
                'tau_w': None,
                'tau_s': None,
                'T': None,
                'n_samples': n_pos,
            }

    # Cluster relations into Easy/Medium/Hard
    num_clusters = 3
    cluster_labels = np.full(nrelation, -1, dtype=np.int32)
    cluster_params = []

    if len(relation_ids_with_samples) >= num_clusters * 3:
        try:
            from sklearn.cluster import KMeans
            features = np.column_stack([
                all_pos_means,
                all_pos_stds,
                all_neg_means,
                all_pos_neg_gaps,
                np.log1p(all_cardinalities),
            ])
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            features = StandardScaler().fit_transform(features)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)

            for i, rid in enumerate(relation_ids_with_samples):
                cluster_labels[rid] = int(labels[i])

            # Compute cluster-level parameters
            # Sort clusters by tau_s (higher = easier relation)
            cluster_stats = defaultdict(lambda: {'tau_w': [], 'tau_s': [], 'T': []})
            for i, rid in enumerate(relation_ids_with_samples):
                cid = int(labels[i])
                cluster_stats[cid]['tau_w'].append(rel_params[rid]['tau_w'])
                cluster_stats[cid]['tau_s'].append(rel_params[rid]['tau_s'])
                cluster_stats[cid]['T'].append(rel_params[rid]['T'])

            # Assign cluster names: highest tau_s -> Easy, lowest -> Hard
            cluster_avg_tau_s = {}
            for cid, stats in cluster_stats.items():
                cluster_avg_tau_s[cid] = np.mean(stats['tau_s'])
            sorted_clusters = sorted(cluster_avg_tau_s.items(), key=lambda x: x[1], reverse=True)
            cluster_name_map = {}
            for idx, (cid, _) in enumerate(sorted_clusters):
                cluster_name_map[cid] = idx  # 0=Easy, 1=Medium, 2=Hard

            # Remap cluster labels
            for i, rid in enumerate(relation_ids_with_samples):
                cluster_labels[rid] = cluster_name_map[int(labels[i])]

            # Build cluster params
            for new_cid in range(num_clusters):
                old_cid = sorted_clusters[new_cid][0]
                cstats = cluster_stats[old_cid]
                cluster_params.append({
                    'id': new_cid,
                    'tau_w': round(float(np.mean(cstats['tau_w'])), 4),
                    'tau_s': round(float(np.mean(cstats['tau_s'])), 4),
                    'T': round(float(np.mean(cstats['T'])), 4),
                    'n_relations': len(cstats['tau_w']),
                })
        except ImportError:
            logging.warning('sklearn not available, skipping clustering. '
                            'All relations without specific params will use global fallback.')
    else:
        logging.warning(f'Not enough relations with samples ({len(relation_ids_with_samples)}) '
                        f'for clustering. Using global fallback only.')

    # Assign cluster IDs to rel_params entries
    for r in range(nrelation):
        if rel_params[r].get('tau_w') is None:
            rel_params[r]['cluster'] = int(cluster_labels[r]) if cluster_labels[r] >= 0 else 0
        else:
            rel_params[r]['cluster'] = int(cluster_labels[r]) if cluster_labels[r] >= 0 else 0

    # Compute global fallback
    all_global_tau_w = [rp['tau_w'] for rp in rel_params.values() if rp['tau_w'] is not None]
    all_global_tau_s = [rp['tau_s'] for rp in rel_params.values() if rp['tau_s'] is not None]
    all_global_T = [rp['T'] for rp in rel_params.values() if rp['T'] is not None]

    if all_global_tau_w:
        global_params = {
            'tau_w': round(float(np.mean(all_global_tau_w)), 4),
            'tau_s': round(float(np.mean(all_global_tau_s)), 4),
            'T': round(float(np.mean(all_global_T)), 4),
        }
    else:
        # Sensible defaults if no per-relation stats available
        global_params = {'tau_w': 4.7, 'tau_s': 6.0, 'T': 0.65}

    # Build output params dict
    params_dict = {
        'nrelation': nrelation,
        'global': global_params,
        'clusters': cluster_params,
        'relations': rel_params,
    }

    # Save to YAML
    os.makedirs(output_dir, exist_ok=True)
    output_fn = os.path.join(output_dir, f'{dataname}_{backbone_type}_rel_memb_params.yml')

    # Use simple YAML format (no numpy serialization)
    with open(output_fn, 'w') as fout:
        fout.write(f"# Relation-aware membership parameters for {dataname} ({backbone_type})\n")
        fout.write(f"# Generated from validation 1p queries\n\n")
        fout.write(f"nrelation: {nrelation}\n\n")
        fout.write(f"global:\n")
        fout.write(f"  tau_w: {global_params['tau_w']}\n")
        fout.write(f"  tau_s: {global_params['tau_s']}\n")
        fout.write(f"  T: {global_params['T']}\n\n")
        fout.write(f"clusters:\n")
        for c in cluster_params:
            fout.write(f"  - id: {c['id']}\n")
            fout.write(f"    tau_w: {c['tau_w']}\n")
            fout.write(f"    tau_s: {c['tau_s']}\n")
            fout.write(f"    T: {c['T']}\n")
            fout.write(f"    n_relations: {c['n_relations']}\n")
        fout.write(f"\n")
        fout.write(f"relations:\n")
        for r in range(nrelation):
            rp = rel_params[r]
            tw = 'null' if rp['tau_w'] is None else rp['tau_w']
            ts = 'null' if rp['tau_s'] is None else rp['tau_s']
            t = 'null' if rp['T'] is None else rp['T']
            fout.write(f"  {r}: {{tau_w: {tw}, tau_s: {ts}, T: {t}, "
                       f"cluster: {rp['cluster']}, n_samples: {rp['n_samples']}}}\n")

    logging.info(f'Saved relation membership params to {output_fn}')
    logging.info(f'  Relations with specific params: {len(all_global_tau_w)}/{nrelation}')
    logging.info(f'  Clusters: {len(cluster_params)}')
    logging.info(f'  Global: tau_w={global_params["tau_w"]}, tau_s={global_params["tau_s"]}, '
                 f'T={global_params["T"]}')
    return params_dict


def create_default_params(nrelation, backbone_type, dataname, output_dir='./params'):
    """
    Create a default (global-only) relation membership parameter file.
    Uses sensible defaults based on existing threshd values. Useful for testing
    before running full statistics computation.
    """
    # Sensible defaults based on existing threshold=4.7
    params_dict = {
        'nrelation': nrelation,
        'global': {
            'tau_w': 5.2,
            'tau_s': 6.5,
            'T': 0.65,
        },
        'clusters': [
            {'id': 0, 'tau_w': 5.5, 'tau_s': 7.2, 'T': 0.85, 'n_relations': 0},
            {'id': 1, 'tau_w': 4.8, 'tau_s': 6.5, 'T': 0.85, 'n_relations': 0},
            {'id': 2, 'tau_w': 4.0, 'tau_s': 5.5, 'T': 0.75, 'n_relations': 0},
        ],
        'relations': {},
    }
    for r in range(nrelation):
        params_dict['relations'][r] = {
            'tau_w': None,
            'tau_s': None,
            'T': None,
            'cluster': 1,  # Default to Medium cluster
            'n_samples': 0,
        }

    os.makedirs(output_dir, exist_ok=True)
    output_fn = os.path.join(output_dir, f'{dataname}_{backbone_type}_rel_memb_params.yml')

    with open(output_fn, 'w') as fout:
        fout.write(f"# Default relation-aware membership parameters for {dataname} ({backbone_type})\n")
        fout.write(f"# Generated with create_default_params() — run compute_rel_stats.py for real values\n\n")
        fout.write(f"nrelation: {nrelation}\n\n")
        fout.write(f"global:\n")
        fout.write(f"  tau_w: {params_dict['global']['tau_w']}\n")
        fout.write(f"  tau_s: {params_dict['global']['tau_s']}\n")
        fout.write(f"  T: {params_dict['global']['T']}\n\n")
        fout.write(f"clusters:\n")
        for c in params_dict['clusters']:
            fout.write(f"  - id: {c['id']}\n")
            fout.write(f"    tau_w: {c['tau_w']}\n")
            fout.write(f"    tau_s: {c['tau_s']}\n")
            fout.write(f"    T: {c['T']}\n")
            fout.write(f"    n_relations: {c['n_relations']}\n")
        fout.write(f"\n")
        fout.write(f"relations:\n")
        for r in range(nrelation):
            rp = params_dict['relations'][r]
            fout.write(f"  {r}: {{tau_w: null, tau_s: null, T: null, "
                       f"cluster: {rp['cluster']}, n_samples: 0}}\n")

    logging.info(f'Saved default relation membership params to {output_fn}')
    return params_dict

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit per-cluster B-spline membership curves for relation-aware fuzzy CQA.

Splits validation 1p queries by relation cluster (Easy/Medium/Hard from
rel_memb_params) and fits a separate B-spline membership curve for each
cluster. Hard-relation clusters get lower thresholds to prevent mid-range
scores from being zeroed out — breaking the monotonicity that makes linear
calibration ineffective for terminal intersection/union queries.

Usage:
    python scripts/fit_cluster_bspline.py \
        --data_path ../KG_data/FB15k-237-betae \
        --checkpoint_path ./models/fb15k-237-betae \
        --dataname FB15k-237 --backbone_type CQD
"""

import os, sys, pickle, argparse, logging
import numpy as np
import yaml
from collections import defaultdict
from tqdm import tqdm
from scipy.interpolate import splrep

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.myglobal as myglobal
from src.base import CQD
import src.discrete as d1

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

N_GRID = 50  # Number of score bins (matches original visual_bspline_cqd.py)
N_INTERIOR_KNOTS = 5  # Interior knots for cubic B-spline


def fit_bspline_for_scores(pos_scores_list, neg_scores_list, manual_threshd=None):
    """Fit a cubic B-spline membership curve from pos/neg score distributions.

    Returns tck, threshd. If manual_threshd is given, uses it instead of
    the grid-search-derived value.
    """
    all_pos = np.concatenate([np.array(v) for v in pos_scores_list if len(v) > 0])
    all_neg = np.concatenate([np.array(v) for v in neg_scores_list if len(v) > 0])

    if len(all_pos) == 0:
        return None, None

    # Determine score range
    score_min = float(np.percentile(all_pos, 0.1))
    score_max = float(np.percentile(all_pos, 99.9))
    score_range = score_max - score_min

    # Create grid bins
    bin_edges = np.linspace(score_min, score_max, N_GRID + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Count pos/neg per bin
    pos_counts = np.zeros(N_GRID)
    neg_counts = np.zeros(N_GRID)
    for i in range(N_GRID):
        pos_counts[i] = np.sum((all_pos >= bin_edges[i]) & (all_pos < bin_edges[i + 1]))
        neg_counts[i] = np.sum((all_neg >= bin_edges[i]) & (all_neg < bin_edges[i + 1]))

    # Target entity frequency per bin
    ys = np.zeros(N_GRID)
    for i in range(N_GRID):
        total = pos_counts[i] + neg_counts[i]
        ys[i] = pos_counts[i] / total if total > 0 else 0.0

    # Force 1.0 from first bin with frequency >= 0.999
    for i in range(N_GRID):
        if ys[i] >= 0.999:
            ys[i:] = 1.0
            break

    # Build extended arrays with zero-plateau below min
    pad_below = max(5, N_GRID // 10)
    xs_pad = np.linspace(score_min - score_range * 0.5, score_min - 1e-6, pad_below)
    ts_all = np.concatenate([xs_pad, bin_centers])
    ys_all = np.concatenate([np.zeros(pad_below), ys])

    # Interior knots at quantiles
    knots = np.quantile(ts_all, np.linspace(0, 1, N_INTERIOR_KNOTS + 2)[1:-1])

    try:
        tck = splrep(ts_all, ys_all, t=knots, k=3)
    except Exception:
        logging.warning('splrep failed, trying with default knots')
        tck = splrep(ts_all, ys_all, k=3)

    # Determine threshold: score below which membership is ~0
    if manual_threshd is not None:
        threshd = manual_threshd
    else:
        # Use the bin where positive frequency first exceeds noise level
        for i in range(N_GRID):
            if ys[i] > 0.02:
                threshd = float(bin_edges[i])
                break
        else:
            threshd = 4.7

    return tck, threshd


def main():
    parser = argparse.ArgumentParser(description='Fit per-cluster B-spline membership')
    parser.add_argument('--data_path', required=True, help='Path to KG data')
    parser.add_argument('--checkpoint_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--dataname', required=True, help='Dataset name')
    parser.add_argument('--backbone_type', default='CQD', help='CQD or QTO')
    parser.add_argument('--output_dir', default='./params', help='Output directory')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--no_cuda', action='store_false', dest='cuda')
    args = parser.parse_args()

    # Load rel_memb_params to get cluster assignments
    rel_params_fn = os.path.join(
        args.output_dir,
        f'{args.dataname}_{args.backbone_type}_rel_memb_params.yml')
    if not os.path.exists(rel_params_fn):
        logging.error(f'rel_memb_params not found: {rel_params_fn}')
        logging.error('Run compute_rel_stats.py first to generate cluster assignments.')
        sys.exit(1)

    with open(rel_params_fn) as f:
        rel_params = yaml.load(f, Loader=yaml.Loader)

    nrelation = rel_params['nrelation']
    logging.info(f'Loaded {nrelation} relations from {rel_params_fn}')

    # Build cluster_id per relation
    cluster_id = np.full(nrelation, 1, dtype=np.int32)
    for rel_id_str, rp in rel_params['relations'].items():
        rid = int(rel_id_str)
        if rid < nrelation and rp.get('cluster') is not None:
            cluster_id[rid] = int(rp['cluster'])

    n_per_cluster = [int(np.sum(cluster_id == i)) for i in range(3)]
    logging.info(f'Cluster sizes: Easy(0)={n_per_cluster[0]}, '
                 f'Medium(1)={n_per_cluster[1]}, Hard(2)={n_per_cluster[2]}')

    # Load validation 1p queries
    with open(os.path.join(args.data_path, 'stats.txt')) as f:
        lines = f.readlines()
        nentity = int(lines[0].split(' ')[-1])

    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    query_structure_1p = ('e', ('r',))

    if query_structure_1p not in valid_queries:
        logging.error('No 1p queries in validation set!')
        sys.exit(1)

    queries_1p = list(valid_queries[query_structure_1p])
    logging.info(f'Found {len(queries_1p)} validation 1p queries')

    # Setup model
    query_name_dict = {query_structure_1p: '1p'}
    model = CQD(nentity, nrelation, rank=1000, test_batch_size=1,
                method='discrete', t_norm_name='prod', k=1024,
                query_name_dict=query_name_dict, wo_fl=False,
                backbone_type=args.backbone_type, use_cuda=args.cuda)
    myglobal.FS_PARAMS.set_static_fs_params(
        backbone=args.backbone_type, memb_type='bspline',
        dataname=args.dataname.lower())
    myglobal.FS_PARAMS.load_membership_params(args.dataname)

    checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint'),
                            map_location='cpu' if not args.cuda else None)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if args.cuda:
        model = model.cuda()
    model.eval()

    entity_embeddings = model.embeddings[0]
    predicate_embeddings = model.embeddings[1]

    # Build answer sets
    answer_sets = {}
    for q, ans in valid_answers.items():
        answer_sets[q] = set(ans) if not isinstance(ans, set) else ans

    # Collect pos/neg scores per cluster
    MAX_PER_REL = 100
    NEG_SAMPLE = 100

    pos_scores_c = [defaultdict(list) for _ in range(3)]
    neg_scores_c = [defaultdict(list) for _ in range(3)]

    # Group queries by relation
    rel_queries = defaultdict(list)
    for q in queries_1p:
        r = q[1][0] if isinstance(q[1], tuple) else q[1]
        rel_queries[r].append(q)

    import torch
    with torch.no_grad():
        for r, r_queries in tqdm(rel_queries.items(), desc='Collecting scores'):
            if len(r_queries) > MAX_PER_REL:
                indices = np.random.choice(len(r_queries), MAX_PER_REL, replace=False)
                r_queries = [r_queries[i] for i in indices]

            cid = int(cluster_id[r])

            for q in r_queries:
                h = q[0]
                answers = answer_sets.get(q, set())

                query_t = torch.LongTensor([[h, r]])
                if args.cuda:
                    query_t = query_t.cuda()

                raw_scores = d1.query_1p(
                    entity_embeddings=entity_embeddings,
                    predicate_embeddings=predicate_embeddings,
                    queries=query_t,
                    scoring_function=lambda rel_e, lhs, rhs: model.score_o(lhs, rel_e, rhs)[0]
                ).squeeze(0)

                for ans in answers:
                    if ans < raw_scores.shape[0]:
                        pos_scores_c[cid][r].append(raw_scores[ans].item())

                answers_set = set(int(a) for a in answers if a < raw_scores.shape[0])
                neg_pool = [i for i in range(raw_scores.shape[0]) if i not in answers_set]
                if len(neg_pool) > NEG_SAMPLE:
                    neg_samples = np.random.choice(neg_pool, NEG_SAMPLE, replace=False)
                else:
                    neg_samples = neg_pool
                for neg_e in neg_samples:
                    neg_scores_c[cid][r].append(raw_scores[int(neg_e)].item())

    # Fit per-cluster B-spline
    # Hard cluster (2) gets lower threshd to prevent zeroing mid-range scores
    CLUSTER_THRESHD = [5.0, 4.7, 3.5]  # Easy, Medium, Hard
    CLUSTER_NAMES = ['Easy', 'Medium', 'Hard']

    cluster_params = []
    for cid in range(3):
        logging.info(f'Fitting B-spline for Cluster {cid} ({CLUSTER_NAMES[cid]})...')

        pos_lists = [v for v in pos_scores_c[cid].values() if len(v) >= 10]
        neg_lists = [v for v in neg_scores_c[cid].values() if len(v) >= 10]

        if len(pos_lists) == 0:
            logging.warning(f'  Cluster {cid}: no data, using global B-spline')
            tck = (myglobal.FS_PARAMS.Bspline_t,
                   myglobal.FS_PARAMS.Bspline_c,
                   myglobal.FS_PARAMS.Bspline_k)
            threshd = CLUSTER_THRESHD[cid]
        else:
            n_pos_total = sum(len(v) for v in pos_lists)
            logging.info(f'  {len(pos_lists)} relations, {n_pos_total} pos samples')
            tck, threshd_fit = fit_bspline_for_scores(
                pos_lists, neg_lists,
                manual_threshd=CLUSTER_THRESHD[cid])
            threshd = CLUSTER_THRESHD[cid]  # Use manual override
            if tck is None:
                logging.warning(f'  Fit failed, using global B-spline')
                tck = (myglobal.FS_PARAMS.Bspline_t,
                       myglobal.FS_PARAMS.Bspline_c,
                       myglobal.FS_PARAMS.Bspline_k)

        cluster_params.append({
            'id': cid,
            'name': CLUSTER_NAMES[cid],
            't': tck[0].tolist(),
            'c': tck[1].tolist(),
            'k': int(tck[2]),
            'threshd': float(threshd),
        })
        logging.info(f'  threshd={threshd:.1f}, knots={len(tck[0])}')

    # Save YAML
    os.makedirs(args.output_dir, exist_ok=True)
    output_fn = os.path.join(
        args.output_dir,
        f'{args.dataname}_{args.backbone_type}_cluster_bspline_params.yml')

    with open(output_fn, 'w') as f:
        f.write(f'# Per-cluster B-spline membership params for {args.dataname} '
                f'({args.backbone_type})\n')
        f.write(f'# Each cluster has its own B-spline curve with cluster-specific\n')
        f.write(f'# threshold. Hard-relation clusters use lower threshd so mid-range\n')
        f.write(f'# scores are not zeroed — breaking soft-min monotonicity for\n')
        f.write(f'# terminal intersection/union queries.\n\n')
        f.write(f'nrelation: {nrelation}\n')
        f.write(f'n_clusters: 3\n\n')
        f.write(f'clusters:\n')
        for cp in cluster_params:
            f.write(f'  - id: {cp["id"]}\n')
            f.write(f'    name: {cp["name"]}\n')
            f.write(f'    threshd: {cp["threshd"]}\n')
            f.write(f'    k: {cp["k"]}\n')
            f.write(f'    t: {yaml.dump(cp["t"], default_flow_style=True).strip()}\n')
            f.write(f'    c: {yaml.dump(cp["c"], default_flow_style=True).strip()}\n')
            f.write(f'    n_relations: {n_per_cluster[cid]}\n')

    logging.info(f'Saved to {output_fn}')


if __name__ == '__main__':
    main()

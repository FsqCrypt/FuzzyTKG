#!/usr/bin/env python3
"""
Compute per-relation membership statistics from validation 1p queries.

This script:
1. Loads a trained model checkpoint
2. Runs 1p inference on all validation queries
3. Collects per-relation positive and negative score distributions
4. Computes tau_w_r, tau_s_r, T_r thresholds
5. Clusters relations into Easy/Medium/Hard
6. Saves params/{dataname}_{BACKBONE}_rel_memb_params.yml

Usage:
    # For CQD backbone:
    python scripts/compute_rel_stats.py \
        --data_path ../KG_data/FB15k-betae \
        --checkpoint_path ./models/fb15k-betae \
        --dataname FB15k \
        --backbone_type CQD

    # For QTO backbone:
    python scripts/compute_rel_stats.py \
        --data_path ../KG_data/FB15k-betae \
        --checkpoint_path ./models/fb15k-q2b \
        --adj_path ./models/fb15k-q2b/neural_adj.pt \
        --dataname FB15k \
        --backbone_type QTO

    # Generate default (global-only) params without running inference:
    python scripts/compute_rel_stats.py --create_default \
        --dataname FB15k --backbone_type CQD --nrelation 2690
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-relation membership statistics')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to KG data (e.g., ../KG_data/FB15k-betae)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint directory')
    parser.add_argument('--dataname', type=str, required=True,
                        help='Dataset name (FB15k, FB15k-237, NELL)')
    parser.add_argument('--backbone_type', type=str, default='CQD',
                        choices=['CQD', 'QTO'])
    parser.add_argument('--memb_type', type=str, default='bspline',
                        choices=['bspline', 'symbolic'])
    parser.add_argument('--adj_path', type=str, default=None,
                        help='Path to QTO adjacency matrix (.pt file)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use GPU')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable GPU')
    parser.add_argument('--output_dir', type=str, default='./params',
                        help='Output directory for params YAML')
    parser.add_argument('--neg_sample_size', type=int, default=100,
                        help='Number of negative samples per query')
    parser.add_argument('--create_default', action='store_true', default=False,
                        help='Create default params without running inference')
    parser.add_argument('--nrelation', type=int, default=None,
                        help='Number of relations (for --create_default)')

    args = parser.parse_args()

    if args.create_default:
        # Import directly to avoid triggering torch imports in src/__init__.py
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rel_membership",
            os.path.join(os.path.dirname(__file__), '..', 'src', 'rel_membership.py'))
        rel_membership = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rel_membership)

        if args.nrelation is None:
            # Try to read from stats.txt
            stats_path = os.path.join(args.data_path, 'stats.txt') if args.data_path else None
            if stats_path and os.path.exists(stats_path):
                with open(stats_path) as f:
                    args.nrelation = int(f.readlines()[1].split(' ')[-1])
            else:
                print("ERROR: --nrelation required when --create_default and no --data_path")
                sys.exit(1)

        create_default_params(
            nrelation=args.nrelation,
            backbone_type=args.backbone_type,
            dataname=args.dataname,
            output_dir=args.output_dir,
        )
        print(f'Default params created for {args.dataname} ({args.backbone_type})')
        return

    # Full statistics computation
    if args.data_path is None or args.checkpoint_path is None:
        print("ERROR: --data_path and --checkpoint_path required for full computation")
        sys.exit(1)

    from src.rel_membership import compute_per_relation_stats

    use_cuda = args.cuda and not args.no_cuda

    params_dict = compute_per_relation_stats(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        dataname=args.dataname,
        backbone_type=args.backbone_type,
        memb_type=args.memb_type,
        cuda=use_cuda,
        output_dir=args.output_dir,
        neg_sample_size=args.neg_sample_size,
    )

    print(f'\nDone! Stats computed for {args.dataname} ({args.backbone_type})')
    n_with = sum(1 for rp in params_dict['relations'].values() if rp['tau_w'] is not None)
    n_total = params_dict['nrelation']
    print(f'Relations with specific params: {n_with}/{n_total}')
    print(f'Global params: tau_w={params_dict["global"]["tau_w"]}, '
          f'tau_s={params_dict["global"]["tau_s"]}, T={params_dict["global"]["T"]}')
    for c in params_dict['clusters']:
        print(f'  Cluster {c["id"]}: tau_w={c["tau_w"]}, tau_s={c["tau_s"]}, '
              f'T={c["T"]}, n_relations={c["n_relations"]}')


if __name__ == '__main__':
    main()

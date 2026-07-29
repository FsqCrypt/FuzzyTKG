"""Add per-relation mean/std normalization params to existing rel_memb_params."""
import pickle, os, sys, yaml, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict

import torch
from src.base import CQD
import src.discrete as d1
import src.myglobal as myglobal

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
data_path = os.path.join(os.path.dirname(_PROJECT_DIR), 'KG_data/FB15k-237-betae')
params_path = os.path.join(_PROJECT_DIR, 'params', 'FB15k-237_CQD_rel_memb_params.yml')

# Load data stats
with open(os.path.join(data_path, 'stats.txt')) as f:
    nentity = int(f.readline().split()[-1])
    nrelation = int(f.readline().split()[-1])

# Load val queries
valid_queries = pickle.load(open(os.path.join(data_path, 'valid-queries.pkl'), 'rb'))
valid_answers = pickle.load(open(os.path.join(data_path, 'valid-hard-answers.pkl'), 'rb'))
qs_1p = list(valid_queries[('e', ('r',))])
rel_queries = defaultdict(list)
for q in qs_1p:
    r = q[1][0] if isinstance(q[1], tuple) else q[1]
    rel_queries[r].append(q)

answer_sets = {}
for q, a in valid_answers.items():
    answer_sets[q] = set(a) if not isinstance(a, set) else a

# Setup model
model = CQD(nentity, nrelation, rank=1000, test_batch_size=1, method='discrete',
            t_norm_name='prod', k=1024, query_name_dict={('e', ('r',)): '1p'},
            wo_fl=False, backbone_type='CQD', use_cuda=True)
myglobal.FS_PARAMS.set_static_fs_params(backbone='CQD', memb_type='bspline', dataname='fb15k-237')
myglobal.FS_PARAMS.load_membership_params('FB15k-237')
ckpt = torch.load(os.path.join(_PROJECT_DIR, 'models', 'fb15k-237-betae', 'checkpoint'))
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model = model.cuda()
model.eval()

# Load existing params (plain YAML, not safe_load to preserve structure)
with open(params_path) as f:
    params = yaml.safe_load(f)

# Collect per-relation pos scores
pos_scores_r = defaultdict(list)
MAX_Q = 50
with torch.no_grad():
    for r, rqs in rel_queries.items():
        if r >= nrelation:
            continue
        for q in rqs[:MAX_Q]:
            h = q[0]
            rid = q[1][0] if isinstance(q[1], tuple) else q[1]
            answers = answer_sets.get(q, set())
            qt = torch.LongTensor([[h, rid]]).cuda()
            raw = d1.query_1p(
                entity_embeddings=model.embeddings[0],
                predicate_embeddings=model.embeddings[1],
                queries=qt,
                scoring_function=lambda rel, lhs, rhs: model.score_o(lhs, rel, rhs)[0]
            ).squeeze(0)
            for a in answers:
                if a < raw.shape[0]:
                    pos_scores_r[r].append(raw[a].item())

# Compute per-relation mean/std
all_pos = []
n_with = 0
for r in range(nrelation):
    vals = pos_scores_r.get(r, [])
    all_pos.extend(vals)
    rkey = r  # YAML integer keys
    if rkey in params.get('relations', {}):
        if len(vals) >= 5:
            mean_r = float(np.mean(vals))
            std_r = float(np.std(vals))
            if std_r < 0.3:
                std_r = 0.3  # floor to avoid division blowup
            params['relations'][rkey]['mean'] = mean_r
            params['relations'][rkey]['std'] = std_r
            n_with += 1
        else:
            params['relations'][rkey]['mean'] = None
            params['relations'][rkey]['std'] = None

# Global stats
all_arr = np.array(all_pos)
global_mean = float(np.mean(all_arr))
global_std = float(np.std(all_arr))
params['global_mean'] = global_mean
params['global_std'] = global_std
print(f"Global: mean={global_mean:.3f}, std={global_std:.3f}")
print(f"Per-relation mean/std: {n_with}/{nrelation}")

# Update cluster defaults
for c in params.get('clusters', []):
    c['mean'] = global_mean
    c['std'] = global_std

# Write updated params (use custom format to match original style)
output_lines = []
output_lines.append('# Relation-aware membership params for FB15k-237 (CQD)')
output_lines.append('# Dual-layer: Score-Shift + Fractional Penalty Offset')
output_lines.append('# Extended with per-relation mean/std for score normalization')
output_lines.append('')
output_lines.append(f'nrelation: {nrelation}')
output_lines.append(f'global_tau_s: {params["global_tau_s"]:.4f}')
output_lines.append(f'global_tau_w: {params["global_tau_w"]:.4f}')
output_lines.append(f'global_offset: {params.get("global_offset", 0.0)}')
output_lines.append(f'global_mean: {global_mean:.4f}')
output_lines.append(f'global_std: {global_std:.4f}')
output_lines.append('')
output_lines.append('clusters:')
for c in params['clusters']:
    output_lines.append(f'  - id: {c["id"]}')
    output_lines.append(f'    shift: {c["shift"]}')
    output_lines.append(f'    offset: {c.get("offset", 0.0)}')
    output_lines.append(f'    mean: {c["mean"]}')
    output_lines.append(f'    std: {c["std"]}')
    output_lines.append(f'    n_relations: {c.get("n_relations", 0)}')
output_lines.append('')
output_lines.append('relations:')
for r in range(nrelation):
    rp = params['relations'].get(str(r), {})
    s_s = 'null' if rp.get('shift') is None else rp['shift']
    ts_s = 'null' if rp.get('tau_s') is None else rp['tau_s']
    tw_s = 'null' if rp.get('tau_w') is None else rp['tau_w']
    gap_s = 'null' if rp.get('gap') is None else rp['gap']
    off_s = 'null' if rp.get('offset') is None else rp['offset']
    mn_s = 'null' if rp.get('mean') is None else f"{rp['mean']:.4f}"
    sd_s = 'null' if rp.get('std') is None else f"{rp['std']:.4f}"
    cl = rp.get('cluster', 1)
    ns = rp.get('n_samples', 0)
    output_lines.append(
        f'  {r}: {{shift: {s_s}, tau_s: {ts_s}, tau_w: {tw_s}, '
        f'gap: {gap_s}, offset: {off_s}, mean: {mn_s}, std: {sd_s}, '
        f'cluster: {cl}, n_samples: {ns}}}'
    )

with open(params_path, 'w') as f:
    f.write('\n'.join(output_lines) + '\n')

print(f"Updated {params_path}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep, splev
from sklearn.metrics import r2_score

import yaml
import sys

plt.rcParams.update({'font.sans-serif': 'Times New Roman'})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.weight': 'bold'})

dataset = sys.argv[-1]
if dataset not in ['NELL', 'FB15k', 'FB15k-237']:
    print('[ERROR] help: python visual_bspine_qto.py [NELL/FB15k/FB15k-237]')
    exit()

bins_num = 50
fn = f'./params/{dataset}_CQD_1p_4.7_validation.txt'
param_fn = f'./params/{dataset}_CQD_params.yml'

threshd = float(fn.split('_')[-2])


x, y, all_x, ans_x = [], [], [], []
with open(fn, 'r') as fin:
    for line in fin:
        data = line.strip()
        if not data: continue
        data = data.split(',')
        xi = float(data[1])

        x.append([xi])
        y.append(int(data[0]))

        all_x.append(xi)
        if y[-1]: 
            ans_x.append(xi)

min_x_value = np.min(x)
max_x_value = np.max(x)

min_all_x_value = np.min(all_x)
max_all_x_value = np.max(all_x)

print(min_x_value, max_x_value, min_all_x_value, max_all_x_value)

step = (max_x_value-min_x_value)/bins_num
bin_x = np.arange(min_x_value, max_x_value+step, step)[:bins_num]

pos_cnt = np.zeros_like(bin_x)
all_cnt = np.ones_like(bin_x)*0.0001
for xi, yi in zip(x, y):
    ind = np.searchsorted(bin_x, xi) - 1
    ind = ind[0]
    if yi == 1:
        pos_cnt[ind] += 1
    all_cnt[ind] += 1
    
ys = pos_cnt/all_cnt
one_ind = np.argwhere(ys >= 0.999)[0,0]
for i in range(one_ind, len(ys)):
    ys[i] = 1.0

ts_neg = np.arange(0., min_x_value, step)
ys_neg = np.zeros_like(ts_neg)

ts_pos = np.arange(bin_x[-1]+step, max_all_x_value+step, step)
ys_pos = np.ones_like(ts_pos)

ts_all = np.concatenate([ts_neg, bin_x, ts_pos])
ys_all = np.concatenate([ys_neg, ys, ys_pos])

n_interior_knots = 5
qs = np.linspace(0, 1, n_interior_knots+2)[1:-1]
knots = np.quantile(ts_all, qs)
tck = splrep(ts_all, ys_all, t=knots, k=3)
print(tck)

tck_obj = {'t': list(tck[0]), 'c': list(tck[1]),  'k': int(tck[2]), 'threshd': threshd}
params_f = open(param_fn, 'w')
yaml.dump(tck_obj, params_f)
params_f.close()
print(f'Save (t,c,k) in {param_fn}')

ys_smooth = splev(ts_all, tck)
ys_smooth = np.clip(ys_smooth, 0, 1)

pos_mask = (ts_all >= tck[0][-1]).astype(float)
ys_smooth = pos_mask + (1 - pos_mask) * ys_smooth

neg_mask = (ts_all >= threshd).astype(float)
ys_smooth = ys_smooth * neg_mask

all_bins_num = int((max_all_x_value-min_all_x_value)//step)
all_bin_x = np.arange(min_all_x_value, max_all_x_value+step, step)[:all_bins_num]

all_hist = np.histogram(all_x, bins=all_bins_num, range=(min_all_x_value,max_all_x_value))[0]
ans_hist = np.histogram(ans_x, bins=all_bins_num, range=(min_all_x_value,max_all_x_value))[0]

fig = plt.figure(figsize=(7.*0.8, 4.5*0.8))
ax = fig.add_subplot(111)

ax.bar(all_bin_x, ans_hist, width=step*0.9, fc='#F2C06B', label='Target')
ax.bar(all_bin_x, all_hist-ans_hist, width=step*0.9, bottom=ans_hist, fc='#7F7F7F', label='Relevant')

ax.set_ylabel("Number of entities")
ax.set_xlabel("Output of LP ($\lambda$)")
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')

major_xticks_top = np.linspace(0,14,5)
minor_xticks_top = np.linspace(0,14,9)

ax.set_xticks(major_xticks_top)
ax.set_xticks(minor_xticks_top, minor=True)

ax.grid(which="major", alpha=0.6, axis='x')
ax.grid(which="minor", alpha=0.3, axis='x')

ax2 = plt.twinx()

ax2.scatter(ts_all, ys_all, color='#82B0D2')

ax2.plot(ts_all, ys_smooth, color='#CF5F55', linewidth=2.0, linestyle='--', label=r'$\mu_S(\lambda)$')

weak_rel = 1 - ys_smooth
ax2.plot(ts_all, weak_rel, color='#5F9387', linewidth=2.0, linestyle='--', label=r'$\mu_W(\lambda)$')

fig.legend(loc='upper left', bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)

major_xticks_top = np.linspace(0,14,5)
minor_xticks_top = np.linspace(0,14,9)
major_yticks_top = np.linspace(0,1,5)
minor_yticks_top = np.linspace(0,1,9)
 
ax2.set_xticks(major_xticks_top)
ax2.set_yticks(major_yticks_top)
ax2.set_xticks(minor_xticks_top, minor=True)
ax2.set_yticks(minor_yticks_top, minor=True)
ax2.grid(which="major", alpha=0.6, axis='y')
ax2.grid(which="minor", alpha=0.3, axis='y')

ax2.set_ylim(0, 1)
ax2.set_ylabel("Membership degrees")
plt.tight_layout()

plt.title('CQD+FS')

save_fig = f'img/{dataset}_CQD_{bins_num}_4.7_valid.svg'
plt.savefig(save_fig, format='svg', dpi=200)
print(f'Save visualization in {save_fig}')

print('R2 score:', r2_score(ys_all, ys_smooth))
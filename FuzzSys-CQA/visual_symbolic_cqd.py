from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.sans-serif': 'Times New Roman'})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.weight': 'bold'})

import sys
import yaml

dataset = sys.argv[-1]
if dataset not in ['NELL', 'FB15k', 'FB15k-237']:
    print('[ERROR] help: python visual_bspine_qto.py [NELL/FB15k/FB15k-237]')
    exit()

fn = f'./params/{dataset}_CQD_1p_4.7_validation.txt'
param_fn = f'./params/{dataset}_CQD_symbolic_params.yml'

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

clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000).fit(x, y)
b0 = clf.intercept_
b1 = clf.coef_
thresh = -b0[0] / b1[0,0]
print('b0 =', clf.intercept_, ', b1 =', clf.coef_)

tck_obj = {'b0': b0, 'b1': b1, 'threshd': threshd}
params_f = open(param_fn, 'w')
yaml.dump(tck_obj, params_f)
params_f.close()
print(f'Save (b0,b1) in {param_fn}')


def strong_memb_func(input):
    return 1 / (1 + np.exp(-(b0 + b1 * input)))

min_x_value = np.min(all_x)
max_x_value = np.max(all_x)
print(min_x_value, max_x_value)
bins_num = 50
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

bin_x_neg = np.arange(0., min_x_value, step)
ys_neg = np.zeros_like(bin_x_neg)

bin_x_all = np.concatenate([bin_x_neg, bin_x])
ys_all = np.concatenate([ys_neg, ys])

all_hist = np.histogram(all_x, bins=bins_num, range=(min_x_value,max_x_value))[0]
ans_hist = np.histogram(ans_x, bins=bins_num, range=(min_x_value,max_x_value))[0]

fig = plt.figure(figsize=(7.*0.8, 4.5*0.8))
ax = fig.add_subplot(111)

ax.bar(bin_x, ans_hist, width=step*0.9, fc='#F2C06B', label='Target')
ax.bar(bin_x, all_hist-ans_hist, width=step*0.9, bottom=ans_hist, fc='#7F7F7F', label='Relevant')
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

ax2.scatter(bin_x_all, ys_all, color='#82B0D2')

strong_rel = strong_memb_func(bin_x_all).flatten()
ax2.plot(bin_x_all, strong_rel, color='#CF5F55', linewidth=2.0, linestyle='-', label=r'$\bar{\mu}_S(\lambda)$')

weak_rel = 1 - strong_rel
ax2.plot(bin_x_all, weak_rel, color='#5F9387', linewidth=2.0, linestyle='-', label=r'$\bar{\mu}_W(\lambda)$')

# ax2.plot([thresh,thresh], [0,1], color='#7F7F7F', linewidth=2.0, linestyle='--')

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

ax2.set_ylim(0, 1)
ax2.set_ylabel("Membership degrees")
plt.tight_layout()

plt.title('CQD')

save_fig = f'img/{dataset}_CQD_symbolic_4.7_valid.svg'
plt.savefig(save_fig, format='svg', dpi=200)
print(f'Save visualization in {save_fig}')

print('R2 score:', r2_score(ys_all, strong_rel))
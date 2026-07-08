import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

fn = f'./params/{dataset}_QTO_1p_4.7_validation.txt'
param_fn = f'./params/{dataset}_QTO_symbolic_params.yml'

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

min_x_value = np.min(all_x)
max_x_value = np.max(all_x)
print(min_x_value, max_x_value)
bins_num = 50
step = (max_x_value-min_x_value)/bins_num
bin_x = np.arange(min_x_value, max_x_value+step, step)[:bins_num]

extra_size = 5
extra_bin_x = np.arange(min_x_value+bins_num*step, min_x_value+(extra_size+bins_num)*step, step)
bin_x = np.concatenate([bin_x, extra_bin_x])

extra_size = len(bin_x) - bins_num

pos_cnt = np.zeros_like(bin_x)
all_cnt = np.zeros_like(bin_x)
for xi, yi in zip(x, y):
    ind = np.searchsorted(bin_x, xi) - 1
    if yi == 1:
        pos_cnt[ind] += 1
    all_cnt[ind] += 1
    if ind >= (bins_num-1):
        for i in range(1,extra_size+1):
            pos_cnt[ind+i] += 1
            all_cnt[ind+i] += 1
ys = pos_cnt/all_cnt

ts_minus = np.arange(0., min_x_value, step)
ys_minus = np.zeros_like(ts_minus)

ts_all = np.concatenate([ts_minus, bin_x])
ys_all = np.concatenate([ys_minus, ys])


def func(x, a, b, c, d):
    return c * np.exp(a * x + b) + d

popt, pcov = curve_fit(func, ts_all[:-(extra_size+1)], ys_all[:-(extra_size+1)])

print(popt)

ab_obj = {'a': popt[0], 'b': popt[1], 'c': popt[2], 'd': popt[3], 'threshd': threshd}
params_f = open(param_fn, 'w')
yaml.dump(ab_obj, params_f)
params_f.close()
print(f'Save (a,b) in {param_fn}')

ys_smooth = func(bin_x, *popt)
ys_all_smooth = func(ts_all, *popt)

ys_smooth = np.clip(ys_smooth, 0, 1)
ys_all_smooth = np.clip(ys_all_smooth, 0, 1)

pos_mask = (bin_x >= max_x_value).astype(float)
ys_smooth = pos_mask + (1 - pos_mask) * ys_smooth
pos_mask = (ts_all >= max_x_value).astype(float)
ys_all_smooth = pos_mask + (1 - pos_mask) * ys_all_smooth

neg_mask = (bin_x >= threshd).astype(float)
ys_smooth = ys_smooth * neg_mask
neg_mask = (ts_all >= threshd).astype(float)
ys_all_smooth = ys_all_smooth * neg_mask

all_hist = np.histogram(all_x, bins=bins_num, range=(min_x_value,max_x_value))[0]
ans_hist = np.histogram(ans_x, bins=bins_num, range=(min_x_value,max_x_value))[0]

extra_all_hist = np.array([all_hist[-1]]*extra_size)
all_hist = np.concatenate((all_hist, extra_all_hist))

extra_ans_hist = np.array([ans_hist[-1]]*extra_size)
ans_hist = np.concatenate((ans_hist, extra_ans_hist))

fig = plt.figure(figsize=(7.*0.8, 4.5*0.8))
ax = fig.add_subplot(111)

ax.bar(bin_x[:bins_num], ans_hist[:bins_num], width=step*0.9, fc='#F2C06B', label='Target')
ax.bar(bin_x[:bins_num], all_hist[:bins_num]-ans_hist[:bins_num], width=step*0.9, bottom=ans_hist[:bins_num], fc='#7F7F7F', label='Relevant')

ax.set_ylabel("Number of entities")
ax.set_xlabel("Output pf LP ($\lambda$)")
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')

major_xticks_top = np.linspace(4,10,5)
minor_xticks_top = np.linspace(4,10,9)

ax.set_xticks(major_xticks_top)
ax.set_xticks(minor_xticks_top, minor=True)

ax.grid(which="major", alpha=0.6, axis='x')
ax.grid(which="minor", alpha=0.3, axis='x')

ax2 = plt.twinx()

ax2.scatter(bin_x, ys, color='#82B0D2')

ax2.plot(ts_all, ys_all_smooth, color='#CF5F55', linewidth=2.0, linestyle='-', label=r'$\bar{\mu}_S(\lambda)$')

weak_rel = 1 - ys_all_smooth
ax2.plot(ts_all, weak_rel, color='#5F9387', linewidth=2.0, linestyle='-', label=r'$\bar{\mu}_W(\lambda)$')

fig.legend(loc='upper left', bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)

major_xticks_top = np.linspace(4,10,5)
minor_xticks_top = np.linspace(4,10,9)
major_yticks_top = np.linspace(0,1,5)
minor_yticks_top = np.linspace(0,1,9)
 
ax2.set_xticks(major_xticks_top)
ax2.set_yticks(major_yticks_top)
ax2.set_xticks(minor_xticks_top, minor=True)
ax2.set_yticks(minor_yticks_top, minor=True)
ax2.grid(which="major", alpha=0.6, axis='y')
ax2.grid(which="minor", alpha=0.3, axis='y')

ax2.set_xlim(4, 10)
ax2.set_ylim(0, 1)
ax2.set_ylabel("Membership degrees")
plt.tight_layout()

plt.title('QTO')

save_fig = f'img/{dataset}_QTO_symbolic_4.7_valid.svg'
plt.savefig(save_fig, format='svg', dpi=200)
print(f'Save visualization in {save_fig}')

print('R2 score:', r2_score(ys_all, ys_all_smooth))
# -*- coding: utf-8 -*-
"""统一协议对比图:NCF 三结构 / LightGCN / SASRec,同一 ML-1M leave-one-out + 100 候选口径。"""
import io, os, glob, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

# 中文字体
for cand in ['Microsoft YaHei', 'SimHei', 'DengXian']:
    if any(cand.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [cand]
        break
plt.rcParams['axes.unicode_minus'] = False


def read_tsv(path):
    ep, hr, nd = [], [], []
    with io.open(path, encoding='utf-8') as f:
        next(f)
        for ln in f:
            p = ln.strip().split('\t')
            if len(p) < 3:
                continue
            ep.append(int(p[0])); hr.append(float(p[1])); nd.append(float(p[2]))
    return ep, hr, nd


NCF_DIR = r"C:\research-training\repro-ncf\ncf_pytorch\logs"
series = {}
for name in ('GMF', 'MLP', 'NeuMF-end'):
    p = os.path.join(NCF_DIR, '%s_ml-1m_factor8.tsv' % name)
    if os.path.exists(p):
        series[name] = read_tsv(p)

lg = os.path.join(OUT, 'logs', 'lightgcn_ml-1m_run1.tsv')
if os.path.exists(lg):
    series['LightGCN'] = read_tsv(lg)

# SASRec:每 20 epoch 一行,来自 log.txt
sas_log = os.path.join(OUT, 'code', 'SASRec.pytorch', 'python', 'ml-1m-all_run1', 'log.txt')
if os.path.exists(sas_log):
    ep, hr, nd = [], [], []
    with io.open(sas_log, encoding='utf-8') as f:
        next(f)
        for ln in f:
            s = ln.replace('np.float64(', '').replace(')', '').replace('(', ' ').replace(',', ' ').replace(')', ' ')
            parts = [x for x in s.replace(',', ' ').split() if x.replace('.', '').replace('-', '').isdigit()]
            if len(parts) >= 5:
                ep.append(int(float(parts[0]))); nd.append(float(parts[3])); hr.append(float(parts[4]))
    if ep:
        series['SASRec'] = (ep, hr, nd)

# ---- 图1:逐 epoch 曲线(两联) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
for name, (ep, hr, nd) in series.items():
    axes[0].plot(ep, hr, label=name, linewidth=1.7, marker='o', markersize=3)
    axes[1].plot(ep, nd, label=name, linewidth=1.7, marker='o', markersize=3)
axes[0].set_title('HR@10 随训练进度变化'); axes[1].set_title('NDCG@10 随训练进度变化')
for ax in axes:
    ax.set_xlabel('epoch'); ax.grid(alpha=.3); ax.legend(fontsize=9)
axes[0].set_ylabel('HR@10'); axes[1].set_ylabel('NDCG@10')
fig.text(0.5, 0.005,
         '注:各模型训练预算不同 —— GMF/MLP/NeuMF 50 epoch(论文最优区间)、LightGCN 300、SASRec 200;'
         'SASRec 每 20 epoch 评估一次,故折点较稀',
         ha='center', fontsize=8.5, color='#555')
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(os.path.join(FIG, 'curves_all.png'), dpi=170); plt.close()

# ---- 图2:统一协议柱状图(取各模型 best-HR epoch;SASRec 用 3 seed 均值±std) ----
stats = {}
sp = os.path.join(OUT, 'stats_runs.json')
if os.path.exists(sp):
    stats = json.load(io.open(sp, encoding='utf-8'))

best = {k: (max(zip(v[1], v[2])), v[0][v[1].index(max(v[1]))]) for k, v in series.items()}
if 'SASRec' in best and stats.get('SASRec'):
    s = stats['SASRec']
    best['SASRec'] = ((s['HR@10_mean'], s['NDCG@10_mean']), 'mean of 3 seeds')

order = ['GMF', 'MLP', 'NeuMF-end', 'LightGCN', 'SASRec']
order = [o for o in order if o in best]
hrs = [best[o][0][0] for o in order]
nds = [best[o][0][1] for o in order]
yerr_hr = [stats['SASRec']['HR@10_std'] if o == 'SASRec' and stats.get('SASRec') else 0 for o in order]
yerr_nd = [stats['SASRec']['NDCG@10_std'] if o == 'SASRec' and stats.get('SASRec') else 0 for o in order]

fig, ax = plt.subplots(figsize=(8.8, 4.9))
x = range(len(order)); w = 0.38
b1 = ax.bar([i - w / 2 for i in x], hrs, w, label='HR@10', color='#2f6fb5',
            yerr=yerr_hr, capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': '#333'})
b2 = ax.bar([i + w / 2 for i in x], nds, w, label='NDCG@10', color='#e08a3c',
            yerr=yerr_nd, capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': '#333'})
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.016,
                '%.4f' % b.get_height(), ha='center', fontsize=8.5)
ax.set_xticks(list(x)); ax.set_xticklabels([o.replace('-end', '') for o in order])
ax.set_ylim(0, max(hrs) * 1.22); ax.set_ylabel('指标值')
ax.set_title('ML-1M · leave-one-out · 100 候选 — 同一协议下的模型对比')
ax.legend(); ax.grid(axis='y', alpha=.3)
fig.text(0.5, 0.005, 'SASRec 为 3 个随机种子的均值,误差棒为标准差(HR ±0.0014 / NDCG ±0.0007);其余为单次运行',
         ha='center', fontsize=8.5, color='#555')
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(os.path.join(FIG, 'bars_unified.png'), dpi=170); plt.close()

summary = {o: {'HR@10': round(best[o][0][0], 4), 'NDCG@10': round(best[o][0][1], 4),
               'best_epoch': best[o][1]} for o in order}
json.dump(summary, io.open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('figures ->', FIG)

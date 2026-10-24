# -*- coding: utf-8 -*-
"""一键汇总：扫描所有模型日志 → results/summary.json + 对比图。

约定日志格式（所有新模型统一走 common.eval.Logger 写出的 tsv）：
    epoch \t HR@10 \t NDCG@10
旧模型（NCF / LightGCN / SASRec）的日志也兼容读取。
"""
import io
import os
import glob
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, 'figures')
RES = os.path.join(HERE, 'results')
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

for cand in ['Microsoft YaHei', 'SimHei', 'DengXian']:
    if any(cand.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [cand]
        break
plt.rcParams['axes.unicode_minus'] = False

PAPER = {          # 论文报告值（None = 该协议下未报告）
    'BPR-MF':   ('—', '—'),
    'NGCF':     (None, None),
    'LightGCN': (None, None),
    'NeuMF':    ('0.688', '0.410'),
    'MLP':      (None, None),
    'GMF':      (None, None),
    'SASRec':   ('0.8235', '0.5958'),
    'BERT4Rec': (None, None),
}
ORDER = ['BPR-MF', 'GMF', 'MLP', 'NeuMF', 'NGCF', 'LightGCN', 'SASRec', 'BERT4Rec']


def read_tsv(path):
    ep, hr, nd = [], [], []
    with io.open(path, encoding='utf-8') as f:
        next(f, None)
        for ln in f:
            p = ln.strip().split('\t')
            if len(p) >= 3:
                try:
                    ep.append(int(p[0])); hr.append(float(p[1])); nd.append(float(p[2]))
                except ValueError:
                    pass
    return ep, hr, nd


def best_of(hr, nd, ep):
    i = hr.index(max(hr))
    return max(hr), nd[i], ep[i]


series = {}

# 1) 新框架：runs/<model>/<tag>/metrics.tsv
for p in sorted(glob.glob(os.path.join(HERE, 'runs', '*', '*', 'metrics.tsv'))):
    model = p.split(os.sep)[-3]
    ep, hr, nd = read_tsv(p)
    if not hr:
        continue
    name = {'bpr_mf': 'BPR-MF', 'ngcf': 'NGCF', 'lightgcn': 'LightGCN',
            'bert4rec': 'BERT4Rec'}.get(model, model)
    # 同一模型多个 tag 时保留最好的那个
    if name in series and max(series[name][1]) >= max(hr):
        continue
    series[name] = (ep, hr, nd, p)

# 2) 旧目录：lightgcn/logs、ncf logs
for p, name in [
    (os.path.join(HERE, 'lightgcn', 'logs', 'lightgcn_ml-1m_run1.tsv'), 'LightGCN'),
]:
    if os.path.exists(p) and name not in series:
        ep, hr, nd = read_tsv(p)
        if hr:
            series[name] = (ep, hr, nd, p)

NCF = r"C:\research-training\repro-ncf\ncf_pytorch\logs"
for fn, name in [('GMF_ml-1m_factor8.tsv', 'GMF'), ('MLP_ml-1m_factor8.tsv', 'MLP'),
                 ('NeuMF-end_ml-1m_factor8.tsv', 'NeuMF')]:
    p = os.path.join(NCF, fn)
    if os.path.exists(p):
        ep, hr, nd = read_tsv(p)
        if hr:
            series[name] = (ep, hr, nd, p)

# 3) SASRec：3 seed 均值
stats_path = os.path.join(RES, 'stats_runs.json')
sas = None
if os.path.exists(stats_path):
    st = json.load(io.open(stats_path, encoding='utf-8'))
    if st.get('SASRec'):
        sas = st['SASRec']

if sas:
    series['SASRec'] = ([sas['runs'][0]['best_epoch']], [sas['HR@10_mean']], [sas['NDCG@10_mean']], '3 seeds')

# ---------- 汇总表 ----------
rows = []
for name in ORDER:
    if name not in series:
        continue
    ep, hr, nd, src = series[name]
    b = best_of(hr, nd, ep)
    rows.append({'model': name, 'HR@10': round(b[0], 4), 'NDCG@10': round(b[1], 4),
                 'best_epoch': b[2], 'paper_HR': PAPER.get(name, (None, None))[0],
                 'paper_NDCG': PAPER.get(name, (None, None))[1], 'source': src})

print("%-10s %-9s %-9s %-8s %s" % ("模型", "HR@10", "NDCG@10", "@epoch", "论文"))
print("-" * 62)
for r in rows:
    print("%-10s %-9.4f %-9.4f %-8s %s / %s" % (r['model'], r['HR@10'], r['NDCG@10'],
                                                r['best_epoch'], r['paper_HR'], r['paper_NDCG']))

json.dump(rows, io.open(os.path.join(RES, 'summary_all.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

# ---------- 图 ----------
names = [r['model'] for r in rows]
hrs = [r['HR@10'] for r in rows]
nds = [r['NDCG@10'] for r in rows]

fig, ax = plt.subplots(figsize=(9.2, 4.9))
x = range(len(names)); w = 0.38
b1 = ax.bar([i - w / 2 for i in x], hrs, w, label='HR@10', color='#2f6fb5')
b2 = ax.bar([i + w / 2 for i in x], nds, w, label='NDCG@10', color='#e08a3c')
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                '%.4f' % b.get_height(), ha='center', fontsize=8.6)
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylim(0, max(hrs) * 1.2); ax.set_ylabel('指标值')
ax.set_title('ML-1M · leave-one-out · 100 候选 — 全部模型同一协议')
ax.legend(); ax.grid(axis='y', alpha=.3)
fig.text(0.5, 0.005, 'SASRec 为 3 个随机种子均值;其余为单次运行,取 best-HR epoch',
         ha='center', fontsize=8.5, color='#555')
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig(os.path.join(FIG, 'bars_all.png'), dpi=170)
plt.close()

# ---------- 训练曲线 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for name in names:
    ep, hr, nd, _ = series[name]
    axes[0].plot(ep, hr, label=name, linewidth=1.6)
    axes[1].plot(ep, nd, label=name, linewidth=1.6)
for ax, t in zip(axes, ['HR@10', 'NDCG@10']):
    ax.set_title(t + ' 随训练进度'); ax.set_xlabel('epoch'); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'curves_all.png'), dpi=170)
plt.close()

print("\nfigures ->", FIG)

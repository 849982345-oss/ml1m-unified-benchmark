# -*- coding: utf-8 -*-
"""一键汇总：扫描 papers/<论文>/results/ 下所有成绩 → 总表 + 对比图。

目录约定（一篇论文一个文件夹）：
    papers/01_bpr_mf_2009/results/metrics.tsv      ← run.py 自动写在这
    papers/02_ncf_2017/results/GMF_*.tsv / MLP_*.tsv / NeuMF-end_*.tsv
    papers/03_ngcf_2019/results/metrics.tsv
    papers/04_lightgcn_2020/results/metrics.tsv
    papers/05_sasrec_2018/results/seed*_log.txt   ← 序列模型格式不同
    papers/06_bert4rec_2019/results/metrics.tsv
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPERS = os.path.join(REPO, 'papers')
FIG = os.path.join(REPO, 'figures')
RES = os.path.join(REPO, 'results')
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

for cand in ['Microsoft YaHei', 'SimHei', 'DengXian']:
    if any(cand.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [cand]
        break
plt.rcParams['axes.unicode_minus'] = False

# 文件夹 -> 表格里显示的名字
FOLDER2MODEL = {
    '01_bpr_mf_2009': 'BPR-MF',
    '03_ngcf_2019': 'NGCF',
    '04_lightgcn_2020': 'LightGCN',
    '06_bert4rec_2019': 'BERT4Rec',
}
NCF_FILES = {'GMF_ml-1m_factor8.tsv': 'GMF', 'MLP_ml-1m_factor8.tsv': 'MLP',
             'NeuMF-end_ml-1m_factor8.tsv': 'NeuMF'}
PAPER = {'BPR-MF': ('—', '—'), 'GMF': ('≈0.645', '≈0.335'), 'MLP': ('≈0.640', '≈0.345'),
         'NeuMF': ('0.688', '0.410'), 'NGCF': (None, None), 'LightGCN': (None, None),
         'SASRec': ('0.8235', '0.5958'), 'BERT4Rec': (None, None)}
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
where = {}

# 1) 每篇论文自己的 results/
for folder, model in FOLDER2MODEL.items():
    p = os.path.join(PAPERS, folder, 'results', 'metrics.tsv')
    if os.path.exists(p):
        ep, hr, nd = read_tsv(p)
        if hr:
            series[model] = (ep, hr, nd)
            where[model] = 'papers/%s/results/metrics.tsv' % folder

# 2) NCF 的三个结构
for fn, model in NCF_FILES.items():
    p = os.path.join(PAPERS, '02_ncf_2017', 'results', fn)
    if os.path.exists(p):
        ep, hr, nd = read_tsv(p)
        if hr:
            series[model] = (ep, hr, nd)
            where[model] = 'papers/02_ncf_2017/results/%s' % fn

# 3) SASRec 三个种子（取均值，读 stats_runs.json）
st_path = os.path.join(RES, 'stats_runs.json')
if os.path.exists(st_path):
    st = json.load(io.open(st_path, encoding='utf-8'))
    if st.get('SASRec'):
        s = st['SASRec']
        ep = [r['best_epoch'] for r in s['runs']]
        series['SASRec'] = (ep, [r['HR@10'] for r in s['runs']], [r['NDCG@10'] for r in s['runs']])
        where['SASRec'] = 'papers/05_sasrec_2018/results/seed*_log.txt（3 个种子）'

# 汇总
rows = []
for name in ORDER:
    if name not in series:
        continue
    ep, hr, nd = series[name]
    if name == 'SASRec':
        b = (sum(hr) / len(hr), sum(nd) / len(nd), '均值')
    else:
        b = best_of(hr, nd, ep)
    rows.append({'论文文件夹': '', 'model': name, 'HR@10': round(b[0], 4),
                 'NDCG@10': round(b[1], 4), 'best_epoch': b[2],
                 '论文报告': '%s / %s' % PAPER.get(name, (None, None)),
                 '结果位置': where.get(name, '')})

print("%-10s %-9s %-9s %-10s %s" % ("模型", "HR@10", "NDCG@10", "@epoch", "结果在哪"))
print("-" * 78)
for r in rows:
    print("%-10s %-9.4f %-9.4f %-10s %s" % (r['model'], r['HR@10'], r['NDCG@10'],
                                            r['best_epoch'], r['结果位置']))

json.dump(rows, io.open(os.path.join(RES, 'summary_all.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

# 图
names = [r['model'] for r in rows]
hrs = [r['HR@10'] for r in rows]
nds = [r['NDCG@10'] for r in rows]

fig, ax = plt.subplots(figsize=(9.4, 5.0))
x = range(len(names)); w = 0.38
b1 = ax.bar([i - w / 2 for i in x], hrs, w, label='HR@10', color='#2f6fb5')
b2 = ax.bar([i + w / 2 for i in x], nds, w, label='NDCG@10', color='#e08a3c')
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                '%.4f' % b.get_height(), ha='center', fontsize=8.6)
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylim(0, max(hrs) * 1.2); ax.set_ylabel('指标值')
ax.set_title('ML-1M · leave-one-out · 100 候选 — 六篇论文 / 八个模型')
ax.legend(); ax.grid(axis='y', alpha=.3)
fig.text(0.5, 0.005, 'SASRec 为 3 个种子均值；其余取 best-HR epoch', ha='center', fontsize=8.5, color='#555')
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig(os.path.join(FIG, 'bars_all.png'), dpi=170)
plt.close()

# 曲线
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for name in names:
    ep, hr, nd = series[name]
    axes[0].plot(ep, hr, label=name, linewidth=1.6, marker='o', markersize=2.5)
    axes[1].plot(ep, nd, label=name, linewidth=1.6, marker='o', markersize=2.5)
for ax, t in zip(axes, ['HR@10', 'NDCG@10']):
    ax.set_title(t + ' 随训练进度'); ax.set_xlabel('epoch'); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'curves_all.png'), dpi=170)
plt.close()

print("\n图 ->", FIG)

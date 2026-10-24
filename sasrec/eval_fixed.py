# -*- coding: utf-8 -*-
"""用**固定的** 100 候选(NCF 官方 test.negative)评估已训练的 SASRec。

存在的理由：
    上游 SASRec.pytorch 的 evaluate() 每次从全库**随机采** 100 个负样本，
    与 NCF 官方固定的 test.negative 不是同一份候选集。
    严格说这两种「100 候选」仍然不能同表比较——候选不同，难度不同。

本脚本把 SASRec 拉到 NCF 的固定候选上，做到真正同协议。
两个口径都报：
    · 论文口径（随机 100 负）= 与论文 Table 可比
    · 统一口径（固定 test.negative）= 与 NCF/LightGCN/NGCF 可比

用法::
    python eval_fixed.py --state_dict_path ml-1m-all_seed42/xxx.pth --maxlen 200
"""
import argparse
import io
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SASRec
from utils import data_partition

NCF_DIR = r"C:\research-training\repro-ncf\ncf_pytorch\ml1m_data"


def build_raw2new():
    """重建 build_ml1m_seq.py 里的映射：NCF 原始 item id(0-based) -> 序列数据 item id(1-based)。"""
    items = set()
    for fn in ("ml-1m.train.rating", "ml-1m.test.rating"):
        for line in io.open(os.path.join(NCF_DIR, fn), encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            items.add(int(p[1]))
    order = sorted(items)
    return {raw: k + 1 for k, raw in enumerate(order)}


def load_fixed_candidates(raw2new):
    """{user_new(1-based): [1 正 + 99 负]}，全部转成序列数据 id。"""
    cand = {}
    with io.open(os.path.join(NCF_DIR, "ml-1m.test.negative"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            u, pos = parts[0].strip("()").split(",")
            u = int(u) + 1                       # NCF 0-based -> 序列 1-based
            ids = [int(pos)] + [int(x) for x in parts[1:] if x.strip()]
            cand[u] = [raw2new[i] for i in ids]
    return cand


@torch.no_grad()
def evaluate_fixed(model, dataset, cand, args, k=10):
    [train, valid, test, usernum, itemnum] = dataset
    model.eval()
    ndcg = hit = n = 0.0
    for u in sorted(cand):
        if len(train.get(u, [])) < 1:
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if len(valid.get(u, [])):
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        item_idx = cand[u]
        logits = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        logits = logits.squeeze(0).cpu().numpy()
        pos_score = logits[0]
        rank = int((logits > pos_score).sum())    # 0-based 名次
        n += 1
        if rank < k:
            hit += 1
            ndcg += 1.0 / np.log2(rank + 2)
    return hit / n, ndcg / n, int(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state_dict_path', required=True)
    p.add_argument('--dataset', default='ml-1m-all')
    p.add_argument('--maxlen', type=int, default=200)
    p.add_argument('--hidden_units', type=int, default=50)
    p.add_argument('--num_blocks', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=1)
    p.add_argument('--dropout_rate', type=float, default=0.2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--norm_first', action='store_true', default=False)
    p.add_argument('--l2_emb', type=float, default=0.0)
    p.add_argument('--num_epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    dataset = data_partition(args.dataset)
    [_, _, _, usernum, itemnum] = dataset
    print("usernum=%d itemnum=%d" % (usernum, itemnum), flush=True)

    model = SASRec(usernum, itemnum, args).to(args.device)
    sd = torch.load(args.state_dict_path, map_location=args.device)
    model.load_state_dict(sd)
    print("loaded", os.path.basename(args.state_dict_path), flush=True)

    raw2new = build_raw2new()
    cand = load_fixed_candidates(raw2new)
    print("固定候选用户数:", len(cand), "| 每用户候选宽度:", len(next(iter(cand.values()))), flush=True)

    hr, nd, n = evaluate_fixed(model, dataset, cand, args)
    print("\n[统一口径 · 固定 test.negative] 用户 %d | HR@10 %.4f | NDCG@10 %.4f" % (n, hr, nd), flush=True)


if __name__ == '__main__':
    main()

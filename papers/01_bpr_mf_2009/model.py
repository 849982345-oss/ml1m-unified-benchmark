# -*- coding: utf-8 -*-
"""BPR-MF（Rendle et al., UAI 2009）—— 矩阵分解 + 成对排序损失。

为什么必须有这个模型：
    "NCF vs MF Revisited"(2020) 的核心结论是——同等调参下，简单的 MF 点积
    在所有指标上都优于 MLP，NeuMF 相对 MLP 只有微小提升。也就是说：
    **不跑 BPR-MF，你无法回答「神经网络到底赢没赢」。**

超参对齐：与 LightGCN 相同（dim 64 / lr 1e-3 / reg 1e-4 / batch 2048），
唯一差别是**没有图传播**——这正是消融的意义。
"""
import os
import sys
import numpy as np
import torch

# 向上找到含 common/ 的仓库根（本文件在 papers/<论文>/ 下）
_p = os.path.abspath(__file__)
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, 'common')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
from common.data import ML1M
from common.eval import evaluate, Logger, set_seed, dump_config


class BPRMF(torch.nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.embedding_user = torch.nn.Embedding(n_users, dim)
        self.embedding_item = torch.nn.Embedding(n_items, dim)
        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std=0.1)

    def score_candidates(self, users, items):
        """users: [B]  items: [B, C] -> [B, C]  —— 纯点积，无传播"""
        return (self.embedding_user(users).unsqueeze(1) * self.embedding_item(items)).sum(-1)

    def bpr_loss(self, users, pos, neg):
        u = self.embedding_user(users)
        p = self.embedding_item(pos)
        n = self.embedding_item(neg)
        pos_s = (u * p).sum(1)
        neg_s = (u * n).sum(1)
        loss = torch.mean(torch.nn.functional.softplus(neg_s - pos_s))
        reg = (1 / 2) * (u.norm(2).pow(2) + p.norm(2).pow(2) + n.norm(2).pow(2)) / float(len(users))
        return loss, reg


def train(args):
    set_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ML1M()
    print("data:", ds.summary(), "| device:", dev, flush=True)

    model = BPRMF(ds.n_users, ds.n_items, args.dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    log = Logger(os.path.join(out, 'metrics.tsv'))
    dump_config(out, model='bpr_mf', dim=args.dim, lr=args.lr, reg=args.reg,
                bs=args.bs, epochs=args.epochs, seed=args.seed, dataset=ds.summary())

    tr = ds.train
    n_batch = len(tr) // args.bs + 1
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(n_batch):
            idx = np.random.randint(0, len(tr), size=args.bs)
            us = tr[idx, 0].astype(np.int64)
            pos = tr[idx, 1].astype(np.int64)
            neg = np.random.randint(0, ds.n_items, size=len(us))
            for k in range(len(us)):                       # 排除已交互
                while neg[k] in ds.pos_by_user[int(us[k])]:
                    neg[k] = np.random.randint(0, ds.n_items)
            u_t = torch.LongTensor(us).to(dev)
            p_t = torch.LongTensor(pos).to(dev)
            n_t = torch.LongTensor(neg).to(dev)
            loss, reg = model.bpr_loss(u_t, p_t, n_t)
            opt.zero_grad()
            (loss + args.reg * reg).backward()
            opt.step()
            total += float(loss.detach())

        if ep % args.eval_every == 0 or ep == args.epochs:
            hr, ndcg = evaluate(model, ds, dev)
            log.log(ep, hr, ndcg)

    print("BEST:", log.summary(), flush=True)
    return log.summary()

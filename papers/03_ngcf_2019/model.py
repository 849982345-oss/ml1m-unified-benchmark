# -*- coding: utf-8 -*-
"""NGCF（Wang et al., SIGIR 2019）—— 带特征变换与非线性的图卷积推荐。

与 LightGCN 的关系：**NGCF 是 LightGCN 要删的那一半**。
LightGCN 论文的消融证明：把这里的 W 和 σ 去掉，性能反而更好。
所以这两个模型必须成对出现，才能讲清「少即是多」这条结论。

传播规则（论文式 3）::
    e_u^(l+1) = LeakyReLU( W1·(e_u^l + Σ_{i∈N(u)} 1/√(|N(u)||N(i)|) · e_i^l)
                           + W2·( Σ_{i∈N(u)} 1/√(|N(u)||N(i)|) · e_i^l ⊙ e_u^l ) )
输出为各层表示的**拼接**（NGCF 的做法，LightGCN 改成了加权和）。
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import torch

# 向上找到含 common/ 的仓库根（本文件在 papers/<论文>/ 下）
_p = os.path.abspath(__file__)
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, 'common')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
from common.data import ML1M
from common.eval import evaluate, Logger, set_seed, dump_config


def build_norm_adj(train, n_users, n_items):
    """对称归一化二部图邻接 D^-1/2 A D^-1/2（与 LightGCN 用同一份，便于对照）"""
    users = train[:, 0]
    items = train[:, 1] + n_users
    rows = np.concatenate([users, items])
    cols = np.concatenate([items, users])
    A = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(n_users + n_items, n_users + n_items)).tocsr()
    deg = np.array(A.sum(axis=1)).flatten()
    dinv = np.power(deg, -0.5, where=deg > 0)
    dinv[deg == 0] = 0.0
    A_hat = sp.diags(dinv) @ A @ sp.diags(dinv)
    A_hat = A_hat.tocoo()
    idx = torch.LongTensor(np.vstack([A_hat.row, A_hat.col]))
    return torch.sparse_coo_tensor(idx, torch.FloatTensor(A_hat.data),
                                   torch.Size(A_hat.shape)).coalesce()


class NGCF(torch.nn.Module):
    def __init__(self, n_users, n_items, dim, n_layers, adj, dropout=0.0):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.dropout = dropout
        self.embedding_user = torch.nn.Embedding(n_users, dim)
        self.embedding_item = torch.nn.Embedding(n_items, dim)
        torch.nn.init.xavier_uniform_(self.embedding_user.weight)
        torch.nn.init.xavier_uniform_(self.embedding_item.weight)
        self.adj = adj
        # 每层两个变换矩阵：W1 作用在「聚合+自身」，W2 作用在「聚合⊙自身」的双线性项
        self.W1 = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.W2 = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.act = torch.nn.LeakyReLU(0.2)
        self.out_dim = dim * (n_layers + 1)

    def computer(self):
        ego = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        embs = [ego]
        for l in range(self.n_layers):
            side = torch.sparse.mm(self.adj, ego)               # 邻居聚合
            sum_part = self.W1[l](side + ego)                   # 自连接 + 聚合
            bi_part = self.W2[l](side * ego)                    # 亲和项
            ego = self.act(sum_part + bi_part)
            if self.training and self.dropout > 0:
                ego = torch.nn.functional.dropout(ego, self.dropout)
            embs.append(ego)
        out = torch.cat(embs, dim=1)                            # 拼接各层
        return torch.split(out, [self.n_users, self.n_items])

    def score_candidates(self, users, items):
        all_u, all_i = self.computer()
        return (all_u[users].unsqueeze(1) * all_i[items]).sum(-1)

    def bpr_loss(self, users, pos, neg):
        all_u, all_i = self.computer()
        u = all_u[users]
        pos_s = (u * all_i[pos]).sum(1)
        neg_s = (u * all_i[neg]).sum(1)
        loss = torch.mean(torch.nn.functional.softplus(neg_s - pos_s))
        # NGCF 用 L2 作用于第 0 层嵌入（与 LightGCN 一致）
        reg = (1 / 2) * (self.embedding_user(users).norm(2).pow(2)
                         + self.embedding_item(pos).norm(2).pow(2)
                         + self.embedding_item(neg).norm(2).pow(2)) / float(len(users))
        return loss, reg


def train(args):
    set_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ML1M()
    print("data:", ds.summary(), "| device:", dev, flush=True)

    adj = build_norm_adj(ds.train, ds.n_users, ds.n_items).to(dev)
    model = NGCF(ds.n_users, ds.n_items, args.dim, args.layers, adj, args.dropout).to(dev)
    print("model out_dim =", model.out_dim, "| adj nnz =", adj._nnz(), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    log = Logger(os.path.join(out, 'metrics.tsv'))
    dump_config(out, model='ngcf', dim=args.dim, layers=args.layers, lr=args.lr,
                reg=args.reg, bs=args.bs, epochs=args.epochs, seed=args.seed,
                dropout=args.dropout, dataset=ds.summary())

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
            for k in range(len(us)):
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

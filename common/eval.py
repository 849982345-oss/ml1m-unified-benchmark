# -*- coding: utf-8 -*-
"""统一评估：1 正 + 99 负 = 100 候选，HR@10 / NDCG@10，逐 epoch 落盘。

所有模型必须调用这一个 evaluate()，协议不允许各自实现。
"""
import io
import os
import time
import json
import numpy as np
import torch


def hr_ndcg_at_k(scores, k=10):
    """scores: [B, C] 张量，第 0 列是正样本。返回 (hr_sum, ndcg_sum)。"""
    pos = scores[:, 0:1]
    rank = (scores > pos).sum(dim=1) + 1          # 正样本在候选中的名次（1-based）
    rank = rank.cpu().numpy()
    hit = rank <= k
    ndcg = np.where(hit, 1.0 / np.log2(rank + 1), 0.0)
    return float(hit.sum()), float(ndcg.sum())


@torch.no_grad()
def evaluate(model, dataset, device, k=10, batch=1024, user_ids=None):
    """通用评估：模型需提供 score_candidates(users, items) -> [B, C]。

    对不提供该接口的模型（如序列模型），传 user_ids 并实现 score_one 也行；
    这里只约定候选批次评测这一种，保持协议单一。
    """
    model.eval()
    users = sorted(dataset.cand) if user_ids is None else list(user_ids)
    hr = ndcg = 0.0
    for s in range(0, len(users), batch):
        us = users[s:s + batch]
        items = torch.LongTensor([dataset.cand[u] for u in us]).to(device)
        u_t = torch.LongTensor(us).to(device)
        scores = model.score_candidates(u_t, items)
        h, n = hr_ndcg_at_k(scores, k)
        hr += h
        ndcg += n
    n_users = len(users)
    return hr / n_users, ndcg / n_users


class Logger:
    """逐 epoch 写入 tsv：epoch \t HR@10 \t NDCG@10（6 位小数），并记录 best。"""

    def __init__(self, path, header_extra=None):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.best = (0.0, 0.0, 0)
        self.t0 = time.time()
        with io.open(path, "w", encoding="utf-8") as f:
            f.write("epoch\tHR@10\tNDCG@10\n")

    def log(self, epoch, hr, ndcg, prefix=""):
        with io.open(self.path, "a", encoding="utf-8") as f:
            f.write("%d\t%.6f\t%.6f\n" % (epoch, hr, ndcg))
        if hr > self.best[0]:
            self.best = (hr, ndcg, epoch)
        print("%s%sepoch %3d | HR@10 %.4f | NDCG@10 %.4f | best HR %.4f @%d | %.0fs"
              % (prefix, "" if not prefix else " ", epoch, hr, ndcg, self.best[0], self.best[2],
                 time.time() - self.t0), flush=True)

    def summary(self):
        return {"best_HR@10": round(self.best[0], 6),
                "best_NDCG@10": round(self.best[1], 6),
                "best_epoch": self.best[2],
                "seconds": round(time.time() - self.t0, 1)}


def dump_config(out_dir, **kw):
    os.makedirs(out_dir, exist_ok=True)
    with io.open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                   for k, v in kw.items()}, f, ensure_ascii=False, indent=2)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

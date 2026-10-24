# -*- coding: utf-8 -*-
"""BERT4Rec（Sun et al., CIKM 2019）—— 双向 Transformer + 完形填空式训练。

与 SASRec 的关系：SASRec 是**单向（causal）自注意力**，只能看左边；
BERT4Rec 改成**双向**，用「随机 mask 掉序列中的物品再还原」来训练（完形填空）。

评估协议与 SASRec 严格一致（便于对比）：
  序列 = 训练段 + 验证段（倒数第二条），预测最后一条；
  候选 = 固定 100 个（NCF 官方 test.negative），1 正 + 99 负。

⚠️ 与论文的差异（诚实说明，汇报要提）：
  论文用全库 softmax + 20% 位置 mask；本实现训练时 mask 比例同为 0.2，
  但 loss 只对被 mask 位置计算（等价，且避免 [B,L,3706] 的大张量）。
"""
import io
import os
import sys
import json
import numpy as np
import torch

# 向上找到含 common/ 的仓库根（本文件在 papers/<论文>/ 下）
_p = os.path.abspath(__file__)
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, 'common')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
from common.eval import Logger, set_seed, dump_config, hr_ndcg_at_k

NCF_DIR = r"C:\research-training\repro-ncf\ncf_pytorch\ml1m_data"
SEQ_FILE = os.environ.get(
    "SEQ_FILE", r"C:\research-training\check-2026-10\code\SASRec.pytorch\python\data\ml-1m-all.txt")


# ---------------- 数据 ----------------
class SeqData:
    """从 ml-1m-all.txt(user item,按时间升序)构造序列任务数据。"""

    def __init__(self, path=SEQ_FILE, maxlen=200):
        self.maxlen = maxlen
        User = {}
        usernum = itemnum = 0
        for line in io.open(path, encoding="utf-8"):
            p = line.split()
            if len(p) != 2:
                continue
            u, i = int(p[0]), int(p[1])
            User.setdefault(u, []).append(i)
            usernum = max(usernum, u)
            itemnum = max(itemnum, i)
        self.train, self.valid, self.test = {}, {}, {}
        for u, seq in User.items():
            if len(seq) < 4:
                self.train[u] = seq
                self.valid[u] = []
                self.test[u] = []
            else:
                self.train[u] = seq[:-2]
                self.valid[u] = seq[-2:-1]
                self.test[u] = seq[-1:]
        self.usernum, self.itemnum = usernum, itemnum
        self.MASK = itemnum + 1          # 追加一个 [mask] 令牌

    def input_seq(self, u):
        """训练段 + 验证段，截断到 maxlen，左侧补 0。"""
        seq = self.train[u] + self.valid[u]
        seq = seq[-self.maxlen:]
        out = np.zeros(self.maxlen, dtype=np.int64)
        out[self.maxlen - len(seq):] = seq
        return out


def build_raw2new():
    items = set()
    for fn in ("ml-1m.train.rating", "ml-1m.test.rating"):
        for line in io.open(os.path.join(NCF_DIR, fn), encoding="utf-8"):
            items.add(int(line.rstrip("\n").split("\t")[1]))
    return {raw: k + 1 for k, raw in enumerate(sorted(items))}


def load_fixed_candidates(raw2new):
    cand = {}
    with io.open(os.path.join(NCF_DIR, "ml-1m.test.negative"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            u, pos = parts[0].strip("()").split(",")
            u = int(u) + 1
            ids = [int(pos)] + [int(x) for x in parts[1:] if x.strip()]
            cand[u] = [raw2new[i] for i in ids]
    return cand


# ---------------- 模型 ----------------
class BERT4Rec(torch.nn.Module):
    def __init__(self, itemnum, maxlen, hidden=50, blocks=2, heads=1, dropout=0.2):
        super().__init__()
        self.item_emb = torch.nn.Embedding(itemnum + 2, hidden, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen, hidden)
        self.emb_dropout = torch.nn.Dropout(dropout)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=blocks)
        self.out_proj = torch.nn.Linear(hidden, hidden)   # 输出投影(权重与嵌入绑定见 forward)
        self.maxlen = maxlen
        torch.nn.init.normal_(self.item_emb.weight, std=0.02)
        torch.nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, seq):
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.emb_dropout(x)
        pad_mask = (seq == 0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return h

    def logits_all(self, hidden, items):
        """hidden: [B, H]  items: [B, C] -> scores [B, C]（与嵌入共享权重）"""
        emb = self.item_emb.weight
        return (hidden.unsqueeze(1) * emb[items]).sum(-1)   # items 已是 [B,C]，按索引取即可


def train(args):
    set_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SeqData(maxlen=args.maxlen)
    print("seqdata: users=%d items=%d users_with_test=%d" %
          (ds.usernum, ds.itemnum, sum(1 for u in ds.test if ds.test[u])), flush=True)

    raw2new = build_raw2new()
    cand = load_fixed_candidates(raw2new)
    print("固定候选用户数:", len(cand), flush=True)

    model = BERT4Rec(ds.itemnum, args.maxlen, args.dim, args.layers, args.heads, args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    bce = torch.nn.BCEWithLogitsLoss()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    log = Logger(os.path.join(out, 'metrics.tsv'))
    dump_config(out, model='bert4rec', dim=args.dim, layers=args.layers, lr=args.lr,
                bs=args.bs, epochs=args.epochs, seed=args.seed, maxlen=args.maxlen,
                dropout=args.dropout, mask_prob=args.mask_prob)

    bs = min(args.bs, 128)          # 序列模型 batch 上限：2048 会把 [N,itemnum] 打爆显存
    print("batch_size =", bs, flush=True)
    users = [u for u in sorted(ds.train) if len(ds.train[u]) >= 2]
    seqs = {u: ds.input_seq(u) for u in users}

    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(users)
        total = 0.0
        nb = len(users) // bs + 1
        for bi in range(nb):
            batch = users[bi * bs:(bi + 1) * bs]
            if not batch:
                continue
            seq = torch.LongTensor([seqs[u] for u in batch]).to(dev)
            # 随机 mask（只 mask 非 padding 位置）
            prob = torch.full(seq.shape, args.mask_prob, device=dev)
            m = (torch.rand(seq.shape, device=dev) < prob) & (seq != 0)
            # 保证每条序列至少一个 mask
            empty = ~m.any(dim=1)
            if empty.any():
                idx = torch.randint(0, seq.shape[1], (int(empty.sum()),), device=dev)
                m[empty.nonzero().squeeze(1), idx] = True
            target = seq.clone()
            masked = seq.clone()
            masked[m] = ds.MASK

            h = model.encode(masked)                 # [B, L, H]
            sel = m.nonzero()                        # [N, 2]
            h_sel = h[sel[:, 0], sel[:, 1]]          # [N, H]
            tgt = target[sel[:, 0], sel[:, 1]]       # [N]
            logits = h_sel @ model.item_emb.weight.t()          # [N, itemnum+2]，用矩阵乘避免大张量
            loss = torch.nn.functional.cross_entropy(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())

        if ep % args.eval_every == 0 or ep == args.epochs:
            hr, ndcg = evaluate(model, ds, seqs, cand, dev)
            log.log(ep, hr, ndcg)

    print("BEST:", log.summary(), flush=True)
    return log.summary()


@torch.no_grad()
def evaluate(model, ds, seqs, cand, dev, k=10):
    """在序列末尾放 [mask]，预测最后一条，用固定 100 候选打分。"""
    model.eval()
    users = [u for u in sorted(cand) if u in seqs and len(ds.test[u])]
    hr = ndcg = 0.0
    for s in range(0, len(users), 512):
        batch = users[s:s + 512]
        seq = torch.LongTensor([seqs[u] for u in batch]).to(dev)
        seq = seq.clone()
        seq[:, -1] = ds.MASK                      # 末位换成 mask
        h = model.encode(seq)[:, -1, :]           # 取末位隐状态
        items = torch.LongTensor([cand[u] for u in batch]).to(dev)
        scores = model.logits_all(h, items)
        hh, nn = hr_ndcg_at_k(scores, k)
        hr += hh
        ndcg += nn
    n = len(users)
    return hr / n, ndcg / n

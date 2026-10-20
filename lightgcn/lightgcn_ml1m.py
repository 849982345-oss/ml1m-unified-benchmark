# -*- coding: utf-8 -*-
"""LightGCN (He et al., SIGIR 2020) 在 MovieLens-1M 上的复现 —— 与 NCF 同一评估协议。

协议说明(关键,汇报必讲):
  * 数据切分:与 NCF 完全一致 —— 官方 hexiangnan/NCF 的 leave-one-out 产物
    (train.rating 994,169 行 + test.rating 6,040 行(每用户最新一条)+ test.negative)
  * 评估:1 正 + 99 负 = 100 候选,HR@10 / NDCG@10 —— 与 NCF 逐字一致
  * 训练目标:LightGCN 原论文用 BPR + 1:1 负采样(不是 NCF 的 BCE + 1:4)。
    这是**方法本身的设定差异**,不是评估协议不一致;评估口径相同,结果才可同表比较。

模型与官方实现 (gusye1234/LightGCN-PyTorch) 逐项一致:
  归一化邻接 A_hat = D^-1/2 A D^-1/2;第 0 层只保留 ID 嵌入;K 层传播;
  层组合取各层均值(含第 0 层);BPR loss = softplus(neg - pos) + reg * ||e0||^2。
"""
import io, os, sys, json, time, argparse, collections
import numpy as np
import scipy.sparse as sp
import torch

DATA = r"C:\research-training\repro-ncf\ncf_pytorch\ml1m_data"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """返回 train 三元组、test 候选、n_users, n_items(沿用 NCF 的 0-based id)。"""
    tr = np.loadtxt(os.path.join(DATA, "ml-1m.train.rating"), usecols=(0, 1), dtype=np.int32)
    te = np.loadtxt(os.path.join(DATA, "ml-1m.test.rating"), usecols=(0, 1), dtype=np.int32)
    n_users = int(max(tr[:, 0].max(), te[:, 0].max())) + 1
    n_items = int(max(tr[:, 1].max(), te[:, 1].max())) + 1

    # test.negative: 第 1 列 = (user,item) 正样本, 之后 99 列 = 负采样候选
    cand = {}
    with io.open(os.path.join(DATA, "ml-1m.test.negative"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            u, pos = parts[0].strip("()").split(",")
            u, pos = int(u), int(pos)
            negs = [int(x) for x in parts[1:] if x.strip()]
            cand[u] = ([pos] + negs, 0)   # (候选列表, 正样本位置=0)
    assert len(cand) == n_users == 6040, (len(cand), n_users)
    widths = collections.Counter(len(v[0]) for v in cand.values())
    print("test.negative 候选宽度分布:", dict(widths), flush=True)
    return tr, te, cand, n_users, n_items


def build_norm_adj(tr, n_users, n_items):
    """对称归一化二部图邻接,返回 torch sparse tensor。"""
    users = tr[:, 0]
    items = tr[:, 1] + n_users
    rows = np.concatenate([users, items])
    cols = np.concatenate([items, users])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_users + n_items, n_users + n_items)).tocsr()
    deg = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[deg == 0] = 0.0
    D = sp.diags(d_inv_sqrt)
    A_hat = D @ A @ D
    A_hat = A_hat.tocoo()
    idx = torch.LongTensor(np.vstack([A_hat.row, A_hat.col]))
    val = torch.FloatTensor(A_hat.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(A_hat.shape)).coalesce()


class LightGCN(torch.nn.Module):
    def __init__(self, n_users, n_items, dim, n_layers, adj):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.n_layers = n_layers
        self.embedding_user = torch.nn.Embedding(n_users, dim)
        self.embedding_item = torch.nn.Embedding(n_items, dim)
        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.adj = adj

    def computer(self):
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            embs.append(all_emb)
        light_out = torch.mean(torch.stack(embs, dim=1), dim=1)   # 层组合:均值(含第0层)
        return torch.split(light_out, [self.n_users, self.n_items])

    def bpr_loss(self, users, pos, neg):
        all_u, all_i = self.computer()
        u_e, p_e, n_e = all_u[users], all_i[pos], all_i[neg]
        pos_s = (u_e * p_e).sum(1)
        neg_s = (u_e * n_e).sum(1)
        loss = torch.mean(torch.nn.functional.softplus(neg_s - pos_s))
        reg = (1 / 2) * (self.embedding_user(users).norm(2).pow(2)
                         + self.embedding_item(pos).norm(2).pow(2)
                         + self.embedding_item(neg).norm(2).pow(2)) / float(len(users))
        return loss, reg

    @torch.no_grad()
    def rank_candidates(self, users, cand_items):
        """users: [B], cand_items: [B, C] -> 每个用户在候选里的分数 [B, C]"""
        all_u, all_i = self.computer()
        return (all_u[users].unsqueeze(1) * all_i[cand_items]).sum(-1)


def ndcg_at_k(rank, k=10):
    return 1.0 / np.log2(rank + 1) if rank < k else 0.0


def evaluate(model, cand, batch=1024, k=10):
    users = sorted(cand)
    hr = ndcg = 0.0
    dev = next(model.parameters()).device
    with torch.no_grad():
        for s in range(0, len(users), batch):
            us = users[s:s + batch]
            items = torch.LongTensor([cand[u][0] for u in us]).to(dev)
            scores = model.rank_candidates(torch.LongTensor(us).to(dev), items)
            # 正样本在位置 0;排名 = 比它严格大的候选数 + 1
            pos_score = scores[:, 0:1]
            rank = (scores > pos_score).sum(1) + 1
            rank = rank.cpu().numpy()
            hr += float((rank <= k).sum())
            ndcg += sum(ndcg_at_k(int(r), k) for r in rank)
    n = len(users)
    return hr / n, ndcg / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--reg', type=float, default=1e-4)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', default='run1')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', dev, '| args:', vars(args), flush=True)

    tr, te, cand, n_users, n_items = load_data()
    print("users=%d items=%d train=%d test_users=%d" % (n_users, n_items, len(tr), len(cand)), flush=True)

    adj = build_norm_adj(tr, n_users, n_items).to(dev)
    print("adj nnz=%d" % adj._nnz(), flush=True)

    model = LightGCN(n_users, n_items, args.dim, args.layers, adj).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 用户 -> 已交互物品集合(负采样排除)
    pos_by_user = collections.defaultdict(set)
    for u, i in tr:
        pos_by_user[int(u)].add(int(i))
    train_users = np.array(sorted(pos_by_user), dtype=np.int64)
    pos_list = {u: np.array(sorted(v), dtype=np.int64) for u, v in pos_by_user.items()}

    log_path = os.path.join(OUT_DIR, 'logs', 'lightgcn_ml-1m_%s.tsv' % args.tag)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with io.open(log_path, 'w', encoding='utf-8') as flog:
        flog.write("epoch\tHR@10\tNDCG@10\n")

    n_batch = len(tr) // args.bs + 1
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(n_batch):
            # 按**交互**采样(与官方实现一致:每 epoch 遍历 trainDataSize 条交互)
            idx = np.random.randint(0, len(tr), size=args.bs)
            us = tr[idx, 0].astype(np.int64)
            pos = tr[idx, 1].astype(np.int64)
            neg = np.random.randint(0, n_items, size=len(us))
            for k in range(len(us)):          # 撞到已交互物品就重采
                while neg[k] in pos_by_user[int(us[k])]:
                    neg[k] = np.random.randint(0, n_items)
            u_t = torch.LongTensor(us).to(dev)
            p_t = torch.LongTensor(pos).to(dev)
            n_t = torch.LongTensor(neg).to(dev)
            loss, reg = model.bpr_loss(u_t, p_t, n_t)
            opt.zero_grad()
            (loss + args.reg * reg).backward()
            opt.step()
            total += float(loss)

        if ep % 10 == 0 or ep == args.epochs:
            model.eval()
            hr, ndcg = evaluate(model, cand)
            print("epoch %3d | loss %.4f | HR@10 %.4f | NDCG@10 %.4f | %.0fs"
                  % (ep, total / n_batch, hr, ndcg, time.time() - t0), flush=True)
            with io.open(log_path, 'a', encoding='utf-8') as flog:
                flog.write("%d\t%.6f\t%.6f\n" % (ep, hr, ndcg))
            torch.save(model.state_dict(), os.path.join(OUT_DIR, 'logs', 'lightgcn_%s_ep%d.pth' % (args.tag, ep)))
    print("done. log:", log_path, flush=True)


if __name__ == '__main__':
    main()

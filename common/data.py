# -*- coding: utf-8 -*-
"""统一数据加载：ML-1M 官方 leave-one-out 切分 + 100 候选评估集。

所有模型共用这一份 loader，保证「同一份数据、同一份候选集」。
"""
import io
import os
import collections
import numpy as np

DEFAULT_DATA = os.environ.get(
    "ML1M_DIR", r"C:\research-training\repro-ncf\ncf_pytorch\ml1m_data")


class ML1M:
    """字段：
        train        : [(user, item)]            994,169 条
        test         : [(user, item)]            6,040 条（每用户最新一条）
        cand         : {user: [1 正 + 99 负]}     6,040 个用户，每个 100 候选
        n_users, n_items
        pos_by_user  : {user: set(items)}        训练集已交互（负采样/评估时要用）
    """

    def __init__(self, data_dir=DEFAULT_DATA):
        self.dir = data_dir
        tr = np.loadtxt(os.path.join(data_dir, "ml-1m.train.rating"),
                        usecols=(0, 1), dtype=np.int32)
        te = np.loadtxt(os.path.join(data_dir, "ml-1m.test.rating"),
                        usecols=(0, 1), dtype=np.int32)
        self.train = tr
        self.test = te
        self.n_users = int(max(tr[:, 0].max(), te[:, 0].max())) + 1
        self.n_items = int(max(tr[:, 1].max(), te[:, 1].max())) + 1

        self.cand = {}
        with io.open(os.path.join(data_dir, "ml-1m.test.negative"), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                u, pos = parts[0].strip("()").split(",")
                u = int(u)
                negs = [int(x) for x in parts[1:] if x.strip()]
                self.cand[u] = [int(pos)] + negs      # 正样本固定在第 0 位

        self.pos_by_user = collections.defaultdict(set)
        for u, i in tr:
            self.pos_by_user[int(u)].add(int(i))

        self._check()

    def _check(self):
        assert len(self.cand) == self.n_users, (len(self.cand), self.n_users)
        widths = collections.Counter(len(v) for v in self.cand.values())
        assert widths == {100: self.n_users}, widths     # 每个用户严格 1+99

    def summary(self):
        return {
            "users": self.n_users, "items": self.n_items,
            "train": int(len(self.train)), "test": int(len(self.test)),
            "candidates": 100,
        }

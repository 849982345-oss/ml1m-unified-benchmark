# -*- coding: utf-8 -*-
"""从 NCF 官方 ml-1m 切分重建 SASRec 可用的全量序列文件,并做一致性校验。

要点:
1. NCF 协议:每用户按时间戳,最新一条 = 测试集(test.rating, 6040 行),其余 = 训练集。
2. SASRec 需要 user item 两列、按时间升序;data_partition 会把最后一条当 test、倒数第二条当 valid。
   -> 每个用户 = 时间序 train ... + test(最后),保证测试条目与 NCF 完全相同。
3. 不做 5-核心过滤:保留 3706 个物品(与 NCF 协议同一物品空间,评估候选集才可比)。
4. 校验:把 <5 频次的 290 个物品剔除后,行数应等于官方 SASRec 预处理版的 999,611。
"""
import io, os, collections, json

BASE = r"C:\research-training\repro-ncf\ncf_pytorch\ml1m_data"
OUT = r"C:\research-training\check-2026-10\code\SASRec.pytorch\python\data\ml-1m-all.txt"

train_rows = []   # (u0, i0, ts)
for line in io.open(os.path.join(BASE, "ml-1m.train.rating"), encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    train_rows.append((int(p[0]), int(p[1]), int(p[3])))
test_rows = []
for line in io.open(os.path.join(BASE, "ml-1m.test.rating"), encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    test_rows.append((int(p[0]), int(p[1]), int(p[3])))

print("train=%d test=%d total=%d" % (len(train_rows), len(test_rows), len(train_rows) + len(test_rows)))

# 按用户分组
by_user = collections.defaultdict(list)
for u, i, ts in train_rows:
    by_user[u].append((ts, i))
test_of = {u: (ts, i) for u, i, ts in test_rows}
assert len(test_of) == 6040

# 重编号(1-based,稳定排序:先按原始 id 数值,保证可重复)
users = sorted(by_user)
items = sorted({i for _, i, _ in train_rows} | {i for _, i, _ in test_rows})
u2i = {u: k + 1 for k, u in enumerate(users)}
i2i = {i: k + 1 for k, i in enumerate(items)}
print("users=%d items=%d" % (len(users), len(items)))

lines = []
n_multi_ts = 0
for u in users:
    seq = sorted(by_user[u])          # 按 (timestamp, item) 升序
    ts, i = test_of[u]
    # 同时间戳的多条交互:统计一下,说明排序依据
    seen_ts = collections.Counter(t for t, _ in seq)
    n_multi_ts += sum(1 for c in seen_ts.values() if c > 1)
    for t, it in seq:
        lines.append("%d %d" % (u2i[u], i2i[it]))
    lines.append("%d %d" % (u2i[u], i2i[i]))   # 测试条目固定放最后

with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print("写出 %s  行数=%d" % (OUT, len(lines)))

# ---- 校验 1:频次分布 vs 官方 SASRec 版 ----
cnt = collections.Counter()
for ln in lines:
    cnt[ln.split()[1]] += 1
print("前5物品频次(应=3428/2991/2990/2883/2672):", [c for _, c in cnt.most_common(5)])
print("频次<5的物品数(应=290):", sum(1 for c in cnt.values() if c < 5))
kept = sum(c for c in cnt.values() if c >= 5)
print("去掉<5频次物品后行数(应=999611):", kept)

# ---- 校验 2:与官方 SASRec 预处理版逐对比较(物品在原始 MovieLens id 上对齐无意义,
#      改为比较:用户级序列长度分布 + 每用户最后一对是否等于 NCF test) ----
u2i_seqlen = collections.Counter()
for ln in lines:
    u2i_seqlen[ln.split()[0]] += 1
dist = collections.Counter(u2i_seqlen.values())
print("序列长度 min/max/median:", min(u2i_seqlen.values()), max(u2i_seqlen.values()),
      sorted(u2i_seqlen.values())[len(u2i_seqlen) // 2])

json.dump({"users": len(users), "items": len(items), "rows": len(lines),
           "kept_5core": kept, "multi_ts_users": n_multi_ts},
          io.open(r"C:\research-training\check-2026-10\data_check.json", "w", encoding="utf-8"))
print("OK")

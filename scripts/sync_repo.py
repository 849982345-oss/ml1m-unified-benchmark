# -*- coding: utf-8 -*-
"""把本地仓库内容同步到 GitHub（走 api.github.com，绕开 github.com:443 被重置）。

比 push_via_api.py 多一件事：**支持删除**——远端有、本地没有的文件会被从树上摘掉，
这样冒烟测试产物、临时 log 不会残留在仓库里。

用法：
    python sync_repo.py             # 同步
    python sync_repo.py --dry-run   # 只看会改什么
"""
import base64
import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

OWNER, REPO = "849982345-oss", "ml1m-unified-benchmark"
ROOT = r"C:\research-training\ml1m-unified-benchmark"
BRANCH = "main"
DRY = "--dry-run" in sys.argv

TOKEN = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
if not TOKEN:
    sys.exit("no token -- run: gh auth login")


def api(method, path, payload=None):
    url = "https://api.github.com" + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "hermes-agent")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


EXCLUDE_DIRS = {'.git', '__pycache__', 'runs_smoke', 'ml1m_data'}
EXCLUDE_EXT = {'.pth', '.log', '.pyc'}


def local_files():
    out = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in EXCLUDE_EXT:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT).replace("\\", "/")
            out[rel] = full
    return out


def remote_files(tree_sha):
    t = api("GET", f"/repos/{OWNER}/{REPO}/git/trees/{tree_sha}?recursive=1")
    return {e["path"]: e["sha"] for e in t["tree"] if e["type"] == "blob"}


def main():
    try:
        ref = api("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
        head = ref["object"]["sha"]
    except urllib.error.HTTPError as e:
        if e.code not in (404, 409):
            raise
        print("空仓库，先把已有文件上传（用 push_via_api.py 建种子提交）")
        return

    commit = api("GET", f"/repos/{OWNER}/{REPO}/git/commits/{head}")
    base_tree = commit["tree"]["sha"]
    remote = remote_files(base_tree)
    local = local_files()

    to_add = [p for p in local if p not in remote]
    to_upd = [p for p in local if p in remote]
    to_del = [p for p in remote if p not in local]

    print("远端 %d 个文件 | 本地 %d 个" % (len(remote), len(local)))
    print("新增 %d · 更新 %d · 删除 %d" % (len(to_add), len(to_upd), len(to_del)))
    if to_del:
        print("将删除：" + ", ".join(to_del[:12]) + (" ..." if len(to_del) > 12 else ""))
    if DRY:
        return

    if not (to_add or to_upd or to_del):
        print("已是最新，无需提交")
        return

    entries = []
    for rel in to_add + to_upd:
        raw = open(local[rel], "rb").read()
        blob = api("POST", f"/repos/{OWNER}/{REPO}/git/blobs",
                   {"content": base64.b64encode(raw).decode("ascii"), "encoding": "base64"})
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        print("  +", rel)
    for rel in to_del:
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": None})
        print("  -", rel)

    tree = api("POST", f"/repos/{OWNER}/{REPO}/git/trees", {"base_tree": base_tree, "tree": entries})
    msg = "sync: 新增 %d / 更新 %d / 删除 %d\n\n模型：BPR-MF · NGCF · LightGCN · SASRec · BERT4Rec\n" % (
        len(to_add), len(to_upd), len(to_del))
    c = api("POST", f"/repos/{OWNER}/{REPO}/git/commits",
            {"message": msg, "tree": tree["sha"], "parents": [head]})
    api("PATCH", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
        {"sha": c["sha"], "force": True})
    print("commit:", c["sha"])
    info = api("GET", f"/repos/{OWNER}/{REPO}")
    print("repo:", info["html_url"])


if __name__ == '__main__':
    main()

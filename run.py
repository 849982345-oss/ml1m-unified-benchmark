# -*- coding: utf-8 -*-
"""统一入口：一篇论文一个文件夹，每个文件夹里跑自己的模型。

用法::
    python run.py --model bpr_mf --epochs 300
        → 结果自动写进 papers/01_bpr_mf_2009/results/metrics.tsv

    python run.py --model ngcf   --epochs 300
    python run.py --model bert4rec --epochs 200

LightGCN 有自己的入口（它不是 BPR 式训练）::
    python papers/04_lightgcn_2020/model.py --epochs 300

新增一篇论文：在 papers/ 下建 0X_名字_年份/，放 model.py（实现 train(args)），
然后在下面 PAPER_MODELS 里加一行。
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PAPER_MODELS = {
    'bpr_mf':   'papers/01_bpr_mf_2009/model.py',
    'ngcf':     'papers/03_ngcf_2019/model.py',
    'bert4rec': 'papers/06_bert4rec_2019/model.py',
}


def load(path):
    spec = importlib.util.spec_from_file_location('paper_model', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=sorted(PAPER_MODELS))
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--heads', type=int, default=1)
    p.add_argument('--maxlen', type=int, default=200)
    p.add_argument('--mask_prob', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--reg', type=float, default=1e-4)
    p.add_argument('--bs', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tag', default='run1')
    p.add_argument('--out', default=None, help='默认写进该论文的 results/')
    args = p.parse_args()

    path = os.path.join(HERE, PAPER_MODELS[args.model])
    print("=" * 66)
    print(" 论文目录 : papers/%s/" % os.path.basename(os.path.dirname(path)))
    print(" 模型文件 : %s" % PAPER_MODELS[args.model])
    print(" 超参     : %s" % {k: v for k, v in vars(args).items() if k != 'model'})
    print("=" * 66, flush=True)

    mod = load(path)
    mod.train(args)


if __name__ == '__main__':
    main()

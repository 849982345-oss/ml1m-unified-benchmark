# -*- coding: utf-8 -*-
"""统一入口：所有模型共用同一套数据、评估、日志、随机种子。

用法::
    python run.py --model bpr_mf  --epochs 300 --dim 64 --lr 1e-3 --reg 1e-4 --bs 2048
    python run.py --model ngcf    --epochs 300 --dim 64 --layers 3 --lr 1e-3 --reg 1e-4 --bs 2048
    python run.py --model lightgcn --epochs 300 ...

新增模型只需：在 models/ 下实现 `train(args)` 与 `score_candidates(users, items)`，
然后在 MODELS 里注册一行。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODELS = {
    'bpr_mf': 'models.bpr_mf',
    'ngcf': 'models.ngcf',
    'lightgcn': 'models.lightgcn',
    'bert4rec': 'models.bert4rec',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=sorted(MODELS))
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
    p.add_argument('--out', default=os.path.join(HERE, 'runs'))
    args = p.parse_args()

    mod = __import__(MODELS[args.model], fromlist=['train'])
    print("=" * 60)
    print(" model = %s | %s" % (args.model, vars(args)))
    print("=" * 60, flush=True)
    mod.train(args)


if __name__ == '__main__':
    main()

"""
评估脚本
========
用法（在 mit_states_mtl/ 目录下运行）:
    python evaluate.py --checkpoint checkpoints/best_model.pt \
                       --data-root datasets/release_dataset \
                       [--split test] [--save-results results.json]
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import MITStatesDataset
from models import ResNetMTL
from utils import MTLMetrics
from utils.metrics import ConfusionAccumulator


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    data_root: str,
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 4,
    save_results: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] 设备: {device} | split: {split}")

    # ── 加载模型 ──────────────────────────────────────────────
    model, payload = ResNetMTL.load_checkpoint(checkpoint_path, device=str(device))
    model.eval()

    best_epoch = payload.get("epoch", "?")
    val_joint  = payload.get("val_metrics", {}).get("joint_acc")
    val_str    = f"{val_joint:.4f}" if isinstance(val_joint, float) else "?"
    print(f"[evaluate] epoch: {best_epoch} | 训练时 val joint_acc: {val_str}")

    num_objects = model.num_objects
    num_states  = model.num_states

    # ── 数据 ─────────────────────────────────────────────────
    dataset = MITStatesDataset(root=data_root, split=split)
    loader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    print(f"[evaluate] {split} 样本数: {len(dataset)}")

    # ── 推理 ─────────────────────────────────────────────────
    metrics    = MTLMetrics()
    obj_conf   = ConfusionAccumulator(num_objects)
    state_conf = ConfusionAccumulator(num_states)

    for images, obj_labels, state_labels in loader:
        images       = images.to(device, non_blocking=True)
        obj_dev      = obj_labels.to(device, non_blocking=True)
        state_dev    = state_labels.to(device, non_blocking=True)

        out = model(images)
        metrics.update(out["obj_logits"], out["state_logits"], obj_dev, state_dev)
        obj_conf.update(out["obj_logits"],   obj_labels)
        state_conf.update(out["state_logits"], state_labels)

    # ── 结果输出 ─────────────────────────────────────────────
    results  = metrics.compute()
    id2obj   = {v: k for k, v in (payload.get("obj2id")   or dataset.obj2id).items()}
    id2state = {v: k for k, v in (payload.get("state2id") or dataset.state2id).items()}

    obj_per_class   = obj_conf.per_class_accuracy()
    state_per_class = state_conf.per_class_accuracy()

    worst_obj_ids   = obj_per_class.argsort()[:10].tolist()
    worst_state_ids = state_per_class.argsort()[:10].tolist()

    print("\n" + "="*60)
    print(f"  Object Acc  (Top-1): {results['obj_acc']:.4f}")
    print(f"  State  Acc  (Top-1): {results['state_acc']:.4f}")
    print(f"  Joint  Acc         : {results['joint_acc']:.4f}  ← 核心指标")
    print(f"  Object Acc  (Top-5): {results['obj_top5']:.4f}")
    print(f"  State  Acc  (Top-5): {results['state_top5']:.4f}")
    print()
    print("最差 10 个 Object 类别:")
    for i in worst_obj_ids:
        print(f"    {id2obj.get(i, str(i)):30s} {obj_per_class[i]:.4f}")
    print()
    print("最差 10 个 State 类别:")
    for i in worst_state_ids:
        print(f"    {id2state.get(i, str(i)):30s} {state_per_class[i]:.4f}")
    print("="*60)

    if save_results:
        out_data = {
            "split":      split,
            "checkpoint": checkpoint_path,
            "metrics":    results,
            "worst_objects": [
                {"name": id2obj.get(i, str(i)), "acc": float(obj_per_class[i])}
                for i in worst_obj_ids
            ],
            "worst_states": [
                {"name": id2state.get(i, str(i)), "acc": float(state_per_class[i])}
                for i in worst_state_ids
            ],
        }
        with open(save_results, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"[evaluate] 结果已保存 → {save_results}")

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--data-root",    required=True)
    p.add_argument("--split",        default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--save-results", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path = args.checkpoint,
        data_root       = args.data_root,
        split           = args.split,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        save_results    = args.save_results,
    )

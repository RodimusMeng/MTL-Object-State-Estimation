"""
多任务评估指标
==============
核心指标:
  obj_acc    — 物体分类 Top-1 准确率
  state_acc  — 状态分类 Top-1 准确率
  joint_acc  — 联合准确率（object AND state 同时正确）← 核心指标
  obj_top5   — 物体 Top-5 准确率
  state_top5 — 状态 Top-5 准确率
"""

from typing import Dict

import torch


class MTLMetrics:
    """
    流式指标累加器，支持 mini-batch 累加后统一计算。

    用法:
        metrics = MTLMetrics()
        for batch in loader:
            metrics.update(obj_logits, state_logits, obj_labels, state_labels)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._obj_correct         = 0
        self._state_correct       = 0
        self._joint_correct       = 0
        self._obj_top5_correct    = 0
        self._state_top5_correct  = 0
        self._total               = 0

    @torch.no_grad()
    def update(
        self,
        obj_logits: torch.Tensor,
        state_logits: torch.Tensor,
        obj_labels: torch.Tensor,
        state_labels: torch.Tensor,
    ):
        B = obj_labels.size(0)
        self._total += B

        obj_pred   = obj_logits.argmax(dim=1)
        state_pred = state_logits.argmax(dim=1)
        obj_ok     = (obj_pred   == obj_labels)
        state_ok   = (state_pred == state_labels)

        self._obj_correct   += obj_ok.sum().item()
        self._state_correct += state_ok.sum().item()
        self._joint_correct += (obj_ok & state_ok).sum().item()

        k_obj   = min(5, obj_logits.size(1))
        k_state = min(5, state_logits.size(1))

        if k_obj > 1:
            top5_obj = obj_logits.topk(k_obj, dim=1).indices
            self._obj_top5_correct += (
                top5_obj == obj_labels.unsqueeze(1)).any(dim=1).sum().item()
        else:
            self._obj_top5_correct += obj_ok.sum().item()

        if k_state > 1:
            top5_state = state_logits.topk(k_state, dim=1).indices
            self._state_top5_correct += (
                top5_state == state_labels.unsqueeze(1)).any(dim=1).sum().item()
        else:
            self._state_top5_correct += state_ok.sum().item()

    def compute(self) -> Dict[str, float]:
        if self._total == 0:
            return {k: 0.0 for k in
                    ["obj_acc", "state_acc", "joint_acc", "obj_top5", "state_top5"]}
        n = self._total
        return {
            "obj_acc":    self._obj_correct         / n,
            "state_acc":  self._state_correct       / n,
            "joint_acc":  self._joint_correct       / n,
            "obj_top5":   self._obj_top5_correct    / n,
            "state_top5": self._state_top5_correct  / n,
        }

    def __str__(self) -> str:
        m = self.compute()
        return (
            f"obj={m['obj_acc']:.4f} | "
            f"state={m['state_acc']:.4f} | "
            f"joint={m['joint_acc']:.4f} | "
            f"obj_top5={m['obj_top5']:.4f} | "
            f"state_top5={m['state_top5']:.4f}"
        )


class ConfusionAccumulator:
    """
    单分支混淆矩阵累积器（用于错误分析）。

    用法:
        ca = ConfusionAccumulator(num_classes=245)
        ca.update(obj_logits, obj_labels)
        matrix = ca.get_matrix()  # [num_classes, num_classes]
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        preds  = logits.argmax(dim=1).cpu()
        labels = labels.cpu()
        for p, t in zip(preds, labels):
            self.matrix[t, p] += 1

    def get_matrix(self) -> torch.Tensor:
        return self.matrix

    def per_class_accuracy(self) -> torch.Tensor:
        diag  = self.matrix.diagonal().float()
        total = self.matrix.sum(dim=1).float()
        return diag / (total + 1e-8)

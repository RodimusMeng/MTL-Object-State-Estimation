"""
训练脚本
========
用法（在 mit_states_mtl/ 目录下运行）:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml model.backbone=resnet18
    python train.py --config configs/default.yaml loss.mode=dynamic

命令行 key=value 参数会覆盖配置文件对应项（支持点分路径）。
"""

import argparse
import os
import random
import sys
import time
from typing import Dict

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import build_dataloaders
from models import build_model
from utils import JointMTLoss, MTLMetrics

# ─────────────────────────────────────────────────────────────
#  配置工具
# ─────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_nested(d: dict, key_path: str, value):
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    try:
        value = yaml.safe_load(value)
    except Exception:
        pass
    d[keys[-1]] = value


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
#  优化器 & 调度器
# ─────────────────────────────────────────────────────────────


def build_optimizer(model: nn.Module, criterion: nn.Module, cfg: dict):
    """差分学习率：backbone 比分类头低 10 倍。"""
    opt_cfg = cfg["optimizer"]
    lr = float(opt_cfg["lr"])
    backbone_lr = float(opt_cfg.get("backbone_lr", lr * 0.1))
    wd = float(opt_cfg.get("weight_decay", 1e-4))

    param_groups = [
        {"params": model.obj_head.parameters(), "lr": lr},
        {"params": model.state_head.parameters(), "lr": lr},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": backbone_lr},
    ]
    # dynamic loss 的可学习方差参数也要优化
    if hasattr(criterion.loss_fn, "log_var_obj"):
        param_groups.append(
            {
                "params": [criterion.loss_fn.log_var_obj, criterion.loss_fn.log_var_state],
                "lr": lr,
            }
        )

    name = opt_cfg.get("name", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=wd)
    elif name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(
            param_groups, weight_decay=wd, momentum=float(opt_cfg.get("momentum", 0.9)), nesterov=True
        )
    else:
        raise ValueError(f"不支持的优化器: {name}")


def build_scheduler(optimizer, cfg: dict, num_batches: int):
    sch_cfg = cfg["scheduler"]
    train_cfg = cfg["training"]
    name = sch_cfg.get("name", "cosine").lower()
    epochs = int(train_cfg["epochs"])
    warmup = int(sch_cfg.get("warmup_epochs", 0))
    min_lr = float(sch_cfg.get("min_lr", 1e-6))

    def warmup_lambda(step):
        total = warmup * num_batches
        if step < total:
            return step / max(1, total)
        return 1.0

    warmup_sch = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    if name == "cosine":
        main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(epochs - warmup) * num_batches,
            eta_min=min_lr,
        )
    elif name == "step":
        main_sch = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_cfg.get("step_size", 10)) * num_batches,
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )
    elif name == "plateau":
        main_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=int(sch_cfg.get("patience", 5)),
            factor=float(sch_cfg.get("factor", 0.5)),
            min_lr=min_lr,
        )
    else:
        raise ValueError(f"不支持的调度器: {name}")

    return warmup_sch, main_sch, name, warmup


# ─────────────────────────────────────────────────────────────
#  单 Epoch 训练
# ─────────────────────────────────────────────────────────────


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    warmup_sch,
    main_sch,
    device,
    epoch,
    cfg,
    global_step,
    warmup_steps,
    scheduler_name,
) -> Dict[str, float]:
    model.train()
    metrics = MTLMetrics()
    total_loss = 0.0
    log_interval = int(cfg["training"].get("log_interval", 50))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    for batch_idx, (images, obj_labels, state_labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        obj_labels = obj_labels.to(device, non_blocking=True)
        state_labels = state_labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(images)
        loss_dict = criterion(out["obj_logits"], out["state_logits"], obj_labels, state_labels)
        loss_dict["loss"].backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            nn.utils.clip_grad_norm_(criterion.parameters(), grad_clip)

        optimizer.step()

        step = global_step[0]
        if step < warmup_steps:
            warmup_sch.step()
        elif scheduler_name != "plateau":
            main_sch.step()
        global_step[0] += 1

        total_loss += loss_dict["loss"].item()
        metrics.update(out["obj_logits"].detach(), out["state_logits"].detach(), obj_labels, state_labels)

        if (batch_idx + 1) % log_interval == 0:
            m = metrics.compute()
            print(
                f"  [E{epoch:03d} B{batch_idx+1:04d}] "
                f"loss={loss_dict['loss'].item():.4f} | "
                f"w_obj={loss_dict['weight_obj'].item():.3f} "
                f"w_st={loss_dict['weight_state'].item():.3f} | "
                f"obj={m['obj_acc']:.3f} st={m['state_acc']:.3f} "
                f"joint={m['joint_acc']:.3f}"
            )

    result = metrics.compute()
    result["loss"] = total_loss / len(loader)
    return result


# ─────────────────────────────────────────────────────────────
#  验证
# ─────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    metrics = MTLMetrics()
    total_loss = 0.0

    for images, obj_labels, state_labels in loader:
        images = images.to(device, non_blocking=True)
        obj_labels = obj_labels.to(device, non_blocking=True)
        state_labels = state_labels.to(device, non_blocking=True)

        out = model(images)
        loss_dict = criterion(out["obj_logits"], out["state_logits"], obj_labels, state_labels)
        total_loss += loss_dict["loss"].item()
        metrics.update(out["obj_logits"], out["state_logits"], obj_labels, state_labels)

    result = metrics.compute()
    result["loss"] = total_loss / len(loader)
    return result


# ─────────────────────────────────────────────────────────────
#  主训练循环
# ─────────────────────────────────────────────────────────────


def train(cfg: dict):
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] 设备: {device}")

    # ── 数据 ─────────────────────────────────────────────────
    data_cfg = cfg["data"]
    train_loader, val_loader, _, train_ds = build_dataloaders(
        root=data_cfg["root"],
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 4)),
        split_ratio=tuple(data_cfg.get("split_ratio", [0.7, 0.15, 0.15])),
        seed=int(data_cfg.get("seed", 42)),
        min_samples=int(data_cfg.get("min_samples", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    num_objects = train_ds.num_objects
    num_states = train_ds.num_states
    print(f"[train] #objects={num_objects} | #states={num_states}")

    # ── 模型 ─────────────────────────────────────────────────
    model = build_model(num_objects, num_states, cfg["model"]).to(device)

    # ── 损失 ─────────────────────────────────────────────────
    loss_cfg = cfg["loss"]
    obj_w, state_w = None, None
    if loss_cfg.get("use_class_weights", True):
        obj_w, state_w = train_ds.get_class_weights()
        obj_w, state_w = obj_w.to(device), state_w.to(device)

    criterion = JointMTLoss(
        mode=loss_cfg.get("mode", "static"),
        alpha=float(loss_cfg.get("alpha", 0.5)),
        obj_weight=obj_w,
        state_weight=state_w,
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
    ).to(device)

    # ── 优化器 & 调度器 ───────────────────────────────────────
    optimizer = build_optimizer(model, criterion, cfg)
    warmup_sch, main_sch, scheduler_name, warmup_epochs = build_scheduler(optimizer, cfg, len(train_loader))
    warmup_steps = warmup_epochs * len(train_loader)

    # ── 训练状态 ──────────────────────────────────────────────
    train_cfg = cfg["training"]
    epochs = int(train_cfg["epochs"])
    unfreeze_epoch = int(train_cfg.get("unfreeze_epoch", 5))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 10))
    ckpt_dir = train_cfg.get("checkpoint_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_joint_acc = 0.0
    patience_counter = 0
    global_step = [0]

    print(f"\n{'='*60}")
    print(f"开始训练 | epochs={epochs} | batch={data_cfg['batch_size']}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # backbone 解冻
        if cfg["model"].get("freeze_backbone") and epoch == unfreeze_epoch:
            n = train_cfg.get("unfreeze_layers")
            model.unfreeze_backbone(unfreeze_layers=n if n else None)

        train_m = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            warmup_sch,
            main_sch,
            device,
            epoch,
            cfg,
            global_step,
            warmup_steps,
            scheduler_name,
        )
        val_m = validate(model, val_loader, criterion, device)

        if scheduler_name == "plateau":
            main_sch.step(val_m["joint_acc"])

        elapsed = time.time() - t0
        print(
            f"[E{epoch:03d}/{epochs}] {elapsed:.1f}s | "
            f"train loss={train_m['loss']:.4f} joint={train_m['joint_acc']:.4f} | "
            f"val   loss={val_m['loss']:.4f}  joint={val_m['joint_acc']:.4f} "
            f"obj={val_m['obj_acc']:.4f} state={val_m['state_acc']:.4f}"
        )

        is_best = val_m["joint_acc"] > best_joint_acc
        if is_best:
            best_joint_acc = val_m["joint_acc"]
            patience_counter = 0
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            model.save_checkpoint(
                ckpt_path,
                extra={
                    "epoch": epoch,
                    "val_metrics": val_m,
                    "obj2id": train_ds.obj2id,
                    "state2id": train_ds.state2id,
                },
            )
            print(f"  ✓ best checkpoint (joint_acc={best_joint_acc:.4f})")
        else:
            patience_counter += 1
            if not train_cfg.get("save_best_only", True):
                model.save_checkpoint(
                    os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
                    extra={"epoch": epoch},
                )

        if patience_counter >= early_stop_patience:
            print(f"\n早停: {early_stop_patience} epoch 内 val joint_acc 未提升。")
            break

    print(f"\n训练完成 | 最优 val joint_acc = {best_joint_acc:.4f}")
    return best_joint_acc


# ─────────────────────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="MIT States MTL Training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value 覆盖，如 model.backbone=resnet18")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    for ov in args.overrides:
        if "=" in ov:
            k, v = ov.split("=", 1)
            set_nested(cfg, k, v)
    train(cfg)

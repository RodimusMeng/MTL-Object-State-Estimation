"""
VAW 属性预测训练脚本（对齐论文版）
==================================
关键修正：
  1. 物体名是 INPUT（embedding 后做 gating），不再预测物体
  2. 使用 ALL 显式正/负标注；缺失标签作软负样本（小权重）
  3. RW-BCE：每类正/负独立加权（论文 Eq.13）
  4. Federated mAP：每类只在有标注的样本上算 AP

用法:
    python train_vaw_mtl.py --ann-dir datasets/vaw_repo/data
    python train_vaw_mtl.py --workers 0          # 显存/虚拟内存紧张
    python train_vaw_mtl.py --resume runs/vaw_mtl/best.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# fmt: off
if sys.platform == "win32":
    import ctypes as _ct, importlib.util as _ilu
    _spec = _ilu.find_spec("torch")
    if _spec and _spec.origin:
        _tlib = os.path.join(os.path.dirname(_spec.origin), "lib")
        if os.path.isdir(_tlib):
            os.add_dll_directory(_tlib)
            _iomp = os.path.join(_tlib, "libiomp5md.dll")
            if os.path.isfile(_iomp): _ct.CDLL(_iomp)
    _cbin = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "Library", "bin"))
    if os.path.isdir(_cbin): os.add_dll_directory(_cbin)
# fmt: on

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

VAW_ANN_DIR = "datasets/vaw_repo/data"
VG_IMG_DIRS = [
    "datasets/VG_images/VG_100K",
    "datasets/VG_images/VG_100K_2",
]
OUT_DIR = Path("runs/vaw_mtl")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann-dir",      default=VAW_ANN_DIR)
    p.add_argument("--img-dirs",     nargs="+", default=VG_IMG_DIRS)
    p.add_argument("--base",         default="weights/yolov8m-cls.pt")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch",        type=int,   default=64)
    p.add_argument("--lr",           type=float, default=7e-4,   help="论文 ResNet 用 7e-4")
    p.add_argument("--warmup",       type=int,   default=2,
                   help="前 N epoch 冻结骨干")
    p.add_argument("--min-obj-cnt",  type=int,   default=30)
    p.add_argument("--min-attr-cnt", type=int,   default=50)
    p.add_argument("--img-size",     type=int,   default=224)
    p.add_argument("--workers",      type=int,   default=0,
                   help="Windows 上虚拟内存不足时设 0")
    p.add_argument("--alpha",        type=float, default=0.1,
                   help="RW-BCE 平滑因子（论文用 0.1）")
    p.add_argument("--missing-w",    type=float, default=0.05,
                   help="缺失标签的软负样本权重")
    p.add_argument("--resume",       default="")
    p.add_argument("--device",       default="")
    return p.parse_args()


def get_device(s):
    if s:
        return torch.device(s)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────
def build_loaders(args):
    sys.path.insert(0, ".")
    from datasets.vaw_dataset import VAWDataset, build_vocab, compute_attr_stats

    train_json = os.path.join(args.ann_dir, "train.json")
    val_json   = os.path.join(args.ann_dir, "val.json")
    test_json  = os.path.join(args.ann_dir, "test.json")

    print("扫描训练集构建词典...")
    obj2id, attr2id = build_vocab(train_json, args.min_obj_cnt, args.min_attr_cnt)
    print(f"  物体类别: {len(obj2id)}  属性类别: {len(attr2id)}")

    print("统计每属性正负样本数（用于 RW-BCE）...")
    n_pos, n_neg = compute_attr_stats(train_json, attr2id)
    print(f"  正样本总数: {n_pos.sum():,}  负样本总数: {n_neg.sum():,}")

    kw = dict(img_dirs=args.img_dirs, obj2id=obj2id, attr2id=attr2id,
              img_size=args.img_size)
    train_ds = VAWDataset(train_json, augment=True,  **kw)
    val_ds   = VAWDataset(val_json,   augment=False, **kw)
    test_ds  = VAWDataset(test_json,  augment=False, **kw)

    loader_kw = dict(batch_size=args.batch, num_workers=args.workers,
                     pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    print(f"  train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, obj2id, attr2id, n_pos, n_neg


# ─────────────────────────────────────────────────────────────────────
# RW-BCE 损失（论文 Eq.13）
# ─────────────────────────────────────────────────────────────────────
def compute_rw_weights(n_pos, n_neg, alpha=0.1):
    """
    返回 (w_c, p_c, n_c) 三个 [C] 张量。
      - w_c ∝ 1/n_pos^α，归一化使 sum(w_c) = C
      - p_c ∝ 1/n_pos^α，n_c ∝ 1/n_neg^α，归一化使 p_c + n_c = 2
    所有值都至少为 1（避免分母为 0）。
    """
    n_pos = np.maximum(n_pos, 1).astype(np.float64)
    n_neg = np.maximum(n_neg, 1).astype(np.float64)
    C = len(n_pos)

    wc = (1.0 / n_pos) ** alpha
    wc = wc / wc.sum() * C

    pc_raw = (1.0 / n_pos) ** alpha
    nc_raw = (1.0 / n_neg) ** alpha
    s = pc_raw + nc_raw
    pc = 2.0 * pc_raw / s
    nc = 2.0 * nc_raw / s

    return (torch.tensor(wc, dtype=torch.float32),
            torch.tensor(pc, dtype=torch.float32),
            torch.tensor(nc, dtype=torch.float32))


def rw_bce_loss(logits, y, wc, pc, nc, missing_w=0.05):
    """
    logits : [B, C]
    y      : [B, C] ∈ {1, 0, -1}（-1 = 缺失）
    返回标量损失。
    """
    log_sig  = F.logsigmoid(logits)        # log p
    log_1msig = F.logsigmoid(-logits)      # log(1-p)

    is_pos = (y == 1).float()
    is_neg = (y == 0).float()
    is_mis = (y == -1).float()

    # 显式正例：w_c * p_c * log(p)
    # 显式负例：w_c * n_c * log(1-p)
    # 缺失：软负样本，权重 missing_w * w_c * n_c
    loss_pos = -wc * pc * log_sig    * is_pos
    loss_neg = -wc * nc * log_1msig * is_neg
    loss_mis = -wc * nc * log_1msig * is_mis * missing_w

    loss = loss_pos + loss_neg + loss_mis
    return loss.sum() / (is_pos.sum() + is_neg.sum() + is_mis.sum() * missing_w + 1e-6)


# ─────────────────────────────────────────────────────────────────────
# Federated mAP（每类只在标注过的样本上算 AP）
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, mean, std):
    model.eval()
    all_y = []
    all_s = []

    for imgs, obj_ids, attr_y in loader:
        imgs    = imgs.to(device, non_blocking=True).float().div_(255)
        imgs    = (imgs - mean) / std
        obj_ids = obj_ids.to(device)
        logits  = model(imgs, obj_ids)
        scores  = torch.sigmoid(logits).cpu().numpy()
        all_y.append(attr_y.numpy())
        all_s.append(scores)

    y = np.concatenate(all_y, axis=0)      # [N, C], ∈ {1,0,-1}
    s = np.concatenate(all_s, axis=0)      # [N, C]

    n_attr = y.shape[1]
    aps = []
    valid_classes = 0
    for c in range(n_attr):
        labeled = (y[:, c] != -1)
        if labeled.sum() < 2:
            continue
        y_c = y[labeled, c]
        if y_c.sum() < 1 or y_c.sum() == len(y_c):
            continue   # 全正或全负，AP 没意义
        s_c = s[labeled, c]
        aps.append(average_precision_score(y_c, s_c))
        valid_classes += 1

    return {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "n_classes_evaluated": valid_classes,
    }


# ─────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # cuDNN autotuner：输入尺寸固定，自动选最快卷积算法（纯赚）
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ImageNet 归一化常数（在 GPU 上对 uint8 batch 做 float 化 + 归一化）
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print(f"设备: {device}")
    train_loader, val_loader, test_loader, obj2id, attr2id, n_pos, n_neg = \
        build_loaders(args)
    n_obj  = len(obj2id)
    n_attr = len(attr2id)

    # ── RW-BCE 权重 ───────────────────────────────────────────────
    wc, pc, nc = compute_rw_weights(n_pos, n_neg, alpha=args.alpha)
    wc, pc, nc = wc.to(device), pc.to(device), nc.to(device)
    print(f"权重范围: w_c[{wc.min():.3f}, {wc.max():.3f}]  "
          f"p_c[{pc.min():.3f}, {pc.max():.3f}]  "
          f"n_c[{nc.min():.3f}, {nc.max():.3f}]")

    # ── 模型 ──────────────────────────────────────────────────────
    from models.yolo_mtl import VAWModel

    if args.resume:
        print(f"续训: {args.resume}")
        model = torch.load(args.resume, map_location=device, weights_only=False)
    else:
        print(f"初始化 VAWModel（骨干: {args.base}，"
              f"物体: {n_obj}，属性: {n_attr}）")
        model = VAWModel(n_obj, n_attr, base_weights=args.base)

    model = model.to(device)

    # 保存词典
    vocab = {"obj2id": obj2id, "attr2id": attr2id}
    with open(OUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # ── 优化器（论文用 Adam）──────────────────────────────────────
    backbone_params = list(model.backbone.parameters()) + \
                      list(model.neck_conv.parameters())
    head_params = list(model.obj_emb.parameters()) + \
                  list(model.gate.parameters()) + \
                  list(model.attr_head.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": 1e-5},     # 论文骨干 lr
        {"params": head_params,     "lr": args.lr},  # 论文头 lr=7e-4
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=2
    )

    # ── 混合精度（AMP）──────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── 训练循环 ──────────────────────────────────────────────────
    log = []
    best_map = 0.0

    # 续训时：先评估已有 best.pt，把它的真实分数作为起点，
    # 避免本次第一个 epoch 把更好的旧模型覆盖掉。
    best_ckpt = OUT_DIR / "best.pt"
    if args.resume and best_ckpt.exists():
        print("评估已有 best.pt 以保护历史最优...", flush=True)
        prev = torch.load(best_ckpt, map_location=device, weights_only=False)
        best_map = evaluate(prev, val_loader, device, mean, std)["mAP"]
        print(f"  历史最优 val mAP = {best_map:.4f}（低于此值不会覆盖 best.pt）",
              flush=True)
        del prev

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if epoch <= args.warmup:
            model.freeze_backbone(True)
        else:
            model.freeze_backbone(False)

        model.train()
        loss_sum = 0.0
        n_batches = 0
        n_total = len(train_loader)
        print(f"  [epoch {epoch}] 启动 DataLoader（Windows 上 spawn workers "
              f"+ 首个 batch 约需 1-2 分钟，请勿 Ctrl+C）...", flush=True)

        for imgs, obj_ids, attr_y in train_loader:
            imgs    = imgs.to(device, non_blocking=True).float().div_(255)
            imgs    = (imgs - mean) / std
            obj_ids = obj_ids.to(device)
            attr_y  = attr_y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs, obj_ids)
                loss = rw_bce_loss(logits, attr_y, wc, pc, nc,
                                   missing_w=args.missing_w)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum  += loss.item()
            n_batches += 1

            if n_batches == 1 or n_batches % 200 == 0:
                print(f"    batch {n_batches:5d}/{n_total} | "
                      f"loss {loss_sum/n_batches:.4f} | "
                      f"{time.time()-t0:.0f}s", flush=True)

        elapsed = time.time() - t0
        metrics = evaluate(model, val_loader, device, mean, std)
        scheduler.step(metrics["mAP"])

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss {loss_sum/n_batches:.4f} | "
            f"val mAP {metrics['mAP']:.4f} "
            f"(on {metrics['n_classes_evaluated']} classes) | "
            f"{elapsed:.0f}s"
        )

        log.append({"epoch": epoch, "loss": loss_sum/n_batches, **metrics})

        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            torch.save(model, OUT_DIR / "best.pt")

        torch.save(model, OUT_DIR / "last.pt")

    # ── 测试集 ────────────────────────────────────────────────────
    print("\n─── 测试集评估（best.pt）───")
    best_model = torch.load(OUT_DIR / "best.pt", map_location=device,
                            weights_only=False)
    test_m = evaluate(best_model, test_loader, device, mean, std)
    print(f"  test mAP: {test_m['mAP']:.4f}  "
          f"({test_m['n_classes_evaluated']} classes)")

    log_path = OUT_DIR / "results.json"
    with open(log_path, "w") as f:
        json.dump({"args": vars(args), "train_log": log, "test": test_m},
                  f, indent=2)
    print(f"\n日志: {log_path}")
    print(f"最优 val mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()

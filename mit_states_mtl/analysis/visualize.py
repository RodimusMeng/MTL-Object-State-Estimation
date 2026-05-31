"""
可视化脚本 — 满足任务书"至少3种不同类型图表"要求
=====================================================
生成以下图表（保存到 analysis/figures/）:
  1. class_dist_mit.png      — MIT States 类别长尾分布（柱状图）
  2. state_obj_heatmap.png   — MIT States (state, object) 频率热力图
  3. training_curves.png     — 各实验训练曲线对比（折线图）
  4. model_comparison.png    — 各实验最终指标横向对比（条形图）
  5. domain_bias.png         — 域偏差分析（杯子实验，条形图）

用法:
    python analysis/visualize.py
    python analysis/visualize.py --skip-mit   # 跳过 MIT States 分析（较慢）
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

# Windows 中文字体支持
_CN_FONTS = ["Microsoft YaHei", "SimHei", "SimSun", "STSong", "WenQuanYi Micro Hei"]
_found = next((f for f in _CN_FONTS
               if any(f.lower() in fm.name.lower()
                      for fm in font_manager.fontManager.ttflist)), None)
if _found:
    matplotlib.rcParams["font.family"] = _found
else:
    # fallback：把中文标签换成英文（在 plot 函数里处理）
    pass
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

OUT_DIR  = Path("analysis/figures")
MIT_ROOT = Path("datasets/release_dataset/images")
RUNS_DIR = Path("runs")


def ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 图1：MIT States 类别长尾分布 ───────────────────────────────────────────

def plot_class_dist_mit(min_samples: int = 2):
    if not MIT_ROOT.is_dir():
        print("[跳过] MIT States 目录不存在")
        return

    print("分析 MIT States 类别分布...")
    counts = []
    for folder in sorted(MIT_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        parts = folder.name.split(" ", 1)
        if len(parts) != 2:
            continue
        n = sum(1 for f in folder.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if n >= min_samples:
            counts.append(n)

    counts.sort(reverse=True)
    x = np.arange(len(counts))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MIT States 数据集类别分布", fontsize=14)

    # 左：全量分布
    ax1.bar(x, counts, width=1.0, color="#4C72B0", alpha=0.8)
    ax1.set_xlabel("类别 (按样本数排序)")
    ax1.set_ylabel("样本数")
    ax1.set_title(f"长尾分布（共 {len(counts)} 个有效类别）")
    ax1.axhline(np.median(counts), color="red", linestyle="--",
                label=f"中位数={int(np.median(counts))}")
    ax1.axhline(np.mean(counts), color="orange", linestyle="--",
                label=f"均值={np.mean(counts):.1f}")
    ax1.legend()

    # 右：分布直方图
    ax2.hist(counts, bins=40, color="#DD8452", alpha=0.8, edgecolor="white")
    ax2.set_xlabel("每类样本数")
    ax2.set_ylabel("类别数量")
    ax2.set_title("样本数分布直方图")
    ax2.axvline(10, color="red", linestyle="--", label="min=10")
    ax2.legend()

    plt.tight_layout()
    out = OUT_DIR / "class_dist_mit.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  保存: {out}")


# ─── 图2：MIT States state×object 热力图 ────────────────────────────────────

def plot_state_obj_heatmap(top_n: int = 20):
    if not MIT_ROOT.is_dir():
        print("[跳过] MIT States 目录不存在")
        return

    print("生成 state-object 热力图（Top states & objects）...")
    state_counts  = defaultdict(int)
    obj_counts    = defaultdict(int)
    combo_counts  = defaultdict(int)

    for folder in MIT_ROOT.iterdir():
        if not folder.is_dir():
            continue
        parts = folder.name.split(" ", 1)
        if len(parts) != 2:
            continue
        state, obj = parts
        n = sum(1 for f in folder.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        state_counts[state] += n
        obj_counts[obj]     += n
        combo_counts[(state, obj)] = n

    top_states = [s for s, _ in sorted(state_counts.items(),
                  key=lambda x: -x[1])[:top_n] if s != "adj"]
    top_objs   = [o for o, _ in sorted(obj_counts.items(),
                  key=lambda x: -x[1])[:top_n]]

    mat = np.zeros((len(top_states), len(top_objs)))
    for i, s in enumerate(top_states):
        for j, o in enumerate(top_objs):
            mat[i, j] = combo_counts.get((s, o), 0)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(top_objs)))
    ax.set_xticklabels(top_objs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_states)))
    ax.set_yticklabels(top_states, fontsize=8)
    ax.set_title(f"MIT States：Top-{top_n} 状态 × Top-{top_n} 物体 样本分布热力图")
    ax.set_xlabel("物体类别")
    ax.set_ylabel("状态类别")
    plt.colorbar(im, ax=ax, label="样本数")
    plt.tight_layout()
    out = OUT_DIR / "state_obj_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  保存: {out}")


# ─── 图3：训练曲线对比（折线图）───────────────────────────────────────────

def _load_results_csv(run_dir: Path) -> pd.DataFrame | None:
    csv = run_dir / "results.csv"
    if not csv.is_file():
        return None
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    return df


def plot_training_curves():
    print("生成训练曲线对比图...")

    # 收集所有已完成的实验
    experiments = {
        "Exp B: YOLOv8s-cls (MIT)":   RUNS_DIR / "classify" / "mit_classify",
        "Exp D: YOLOv8s (OSDD)":       RUNS_DIR / "osdd_yolov8s",
        "Exp E1: YOLOv8n (OSDD)":      RUNS_DIR / "osdd_yolov8n",
        "Exp E2: YOLOv8m (OSDD)":      RUNS_DIR / "osdd_yolov8m",
        "Exp F: YOLOv8s no-pretrain":  RUNS_DIR / "osdd_no_pretrain",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("各实验训练曲线对比", fontsize=14)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    any_loaded = False
    for (label, run_dir), color in zip(experiments.items(), colors):
        df = _load_results_csv(run_dir)
        if df is None:
            continue
        any_loaded = True
        epoch = df.get("epoch", pd.Series(range(len(df))))

        # 检测实验：画 box_loss 和 mAP50
        if "train/box_loss" in df.columns:
            axes[0].plot(epoch, df["train/box_loss"], color=color,
                         label=label, linewidth=1.5)
            if "metrics/mAP50(B)" in df.columns:
                axes[1].plot(epoch, df["metrics/mAP50(B)"], color=color,
                             label=label, linewidth=1.5)

        # 分类实验：画 train/loss 和 metrics/accuracy_top1
        elif "train/loss" in df.columns:
            axes[0].plot(epoch, df["train/loss"], color=color,
                         label=label, linewidth=1.5)
            if "metrics/accuracy_top1" in df.columns:
                axes[1].plot(epoch, df["metrics/accuracy_top1"], color=color,
                             label=label, linewidth=1.5)

    if not any_loaded:
        print("  [警告] 暂无训练结果，等训练完成后重新运行")
        plt.close()
        return

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("训练 Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP50 / Top-1 Acc")
    axes[1].set_title("验证集指标")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  保存: {out}")


# ─── 图4：模型对比条形图 ─────────────────────────────────────────────────────

def plot_model_comparison():
    print("生成模型对比条形图...")

    # 已知结果（Exp A 来自中期报告；其他训练完成后会自动读取）
    results = {
        "Exp A\nResNet+MTL\n(MIT States)": {
            "joint_acc": 0.35, "dataset": "MIT States", "type": "baseline"
        },
    }

    # 自动读取 OSDD 实验结果
    osdd_exps = {
        "Exp D\nYOLOv8s\n(OSDD)":      RUNS_DIR / "osdd_yolov8s",
        "Exp E1\nYOLOv8n\n(OSDD)":     RUNS_DIR / "osdd_yolov8n",
        "Exp E2\nYOLOv8m\n(OSDD)":     RUNS_DIR / "osdd_yolov8m",
        "Exp F\nNo pretrain\n(OSDD)":   RUNS_DIR / "osdd_no_pretrain",
    }
    for label, run_dir in osdd_exps.items():
        df = _load_results_csv(run_dir)
        if df is not None and "metrics/mAP50(B)" in df.columns:
            best_map = df["metrics/mAP50(B)"].max()
            results[label] = {"joint_acc": best_map, "dataset": "OSDD", "type": "yolo"}

    # 读取 Exp B 结果
    df_b = _load_results_csv(RUNS_DIR / "classify" / "mit_classify")
    if df_b is not None and "metrics/accuracy_top1" in df_b.columns:
        best_acc = df_b["metrics/accuracy_top1"].max()
        results["Exp B\nYOLOv8s-cls\n(MIT States)"] = {
            "joint_acc": best_acc, "dataset": "MIT States", "type": "yolo"
        }

    if len(results) < 2:
        print("  [警告] 训练结果不足，等实验完成后重新运行")
        return

    labels = list(results.keys())
    values = [results[l]["joint_acc"] for l in labels]
    colors = ["#C44E52" if results[l]["type"] == "baseline" else "#4C72B0"
              for l in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("mAP50 / Joint Accuracy")
    ax.set_title("各实验最终性能对比")
    ax.set_ylim(0, min(1.05, max(values) * 1.15))
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.35, color="red", linestyle="--", alpha=0.5, label="ResNet MTL 基线 (35%)")
    ax.legend(fontsize=9)

    legend_patches = [
        mpatches.Patch(color="#C44E52", label="Baseline (ResNet MTL)"),
        mpatches.Patch(color="#4C72B0", label="YOLOv8 (本项目)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  保存: {out}")


# ─── 图5：OSDD 类别分布 ──────────────────────────────────────────────────────

def plot_osdd_class_dist():
    osdd_lbl = Path("datasets/OSDD/labels/train")
    if not osdd_lbl.is_dir():
        print("[跳过] OSDD labels 目录不存在")
        return

    print("分析 OSDD 类别分布...")
    class_names = [
        "close", "containing_liquid", "containing_solid", "empty",
        "folded", "open", "plugged", "unplugged", "unfolded"
    ]
    counts = np.zeros(len(class_names), dtype=int)
    for f in osdd_lbl.glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, counts, color="#55A868", alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("状态类别")
    ax.set_ylabel("标注框数量")
    ax.set_title("OSDD 训练集类别分布")
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "osdd_class_dist.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  保存: {out}")


# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-mit", action="store_true", help="跳过 MIT States 分析（较慢）")
    args = p.parse_args()

    ensure_out()

    if not args.skip_mit:
        plot_class_dist_mit()
        plot_state_obj_heatmap()

    plot_osdd_class_dist()
    plot_training_curves()
    plot_model_comparison()

    print(f"\n所有图表保存到: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

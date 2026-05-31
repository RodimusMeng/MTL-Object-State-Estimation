# 报告背景数据参考（含排行榜、数据集统计、关键引用）

> 给报告/PPT 撰写者直接取用，数字已核实来源。

---

## 一、MIT-States 数据集

- **规模**：63,440 张图、245 类物体、115 类状态、1,890 种已知组合
- **长尾**：大量 (状态,物体) 组合样本极少（这是我们 ResNet 基线崩溃的根因）
- **CZSL 标准划分**（compositional-split-natural，Purushwalkam et al.）：
  - 训练组合：1,262 种
  - 验证组合：600 种（含 unseen）
  - 测试组合：800 种（含 unseen）
  - 测试集图片：12,988 张

---

## 二、VAW 数据集

- **来源**：Visual Genome v1.2（108K 图）
- **规模**：72,274 张图，260,895 个物体实例
- **标注**：927,679 条（392K 正例 + 534K 负例），620 种属性，2,260 种物体短语
- **划分**：训练 216,790 / 验证 12,286 / 测试 31,819 实例
- **论文**：Pham et al., CVPR 2021

---

## 三、CZSL 评测指标说明（报告 §2 必须解释）

| 指标 | 含义 | 为什么用它 |
|---|---|---|
| **Seen acc (S)** | 测试时，标签属于训练集已见组合的预测准确率 | 衡量记忆/拟合能力 |
| **Unseen acc (U)** | 测试时，标签属于训练集未见组合的预测准确率 | 衡量零样本泛化能力 |
| **HM** | S 和 U 的调和均值 = 2SU/(S+U) | 主指标：惩罚两者失衡，ResNet 式方法 Unseen≈0 → HM≈0 |
| **AUC** | 以偏置参数扫描 Seen/Unseen 权衡曲线下面积 | 不依赖单一阈值，更鲁棒 |

> ⚠️ **不能与 ResNet joint_acc 直接比较**：joint_acc 是全量闭集分类（1890 类），CZSL 指标是组合泛化评测，两者任务定义不同。

---

## 四、MIT-States CZSL 排行榜（Closed-World，ViT-L/14 骨干）

数据来源：CAMS 论文（arXiv 2511.16378，2025）及 Troika 论文（CVPR 2024）。所有方法均使用 CLIP ViT-L/14，除特别标注。

| 方法 | 年份/venue | Seen (S) | Unseen (U) | HM | AUC |
|---|---|---|---|---|---|
| CLIP zero-shot | 2021 | 30.2 | 46.0 | 26.1 | 11.0 |
| CoOp | 2022 | 34.4 | 47.6 | 29.8 | 13.5 |
| **CSP** | **ICLR 2023** | **46.6** | **49.9** | **36.3** | **19.4** |
| DFSP | CVPR 2023 | 46.9 | 52.0 | 37.3 | 20.6 |
| Logic-CSP | 2023 | 48.5 | 51.0 | 37.4 | 20.6 |
| PLID | 2024 | 49.7 | 52.4 | 39.0 | 22.1 |
| Troika | CVPR 2024 | 49.0 | 53.0 | 39.3 | 22.1 |
| CDS-CZSL | CVPR 2024 | 50.3 | 52.9 | 39.2 | 22.4 |
| RAPR | 2024 | 50.0 | 53.3 | 39.2 | 22.5 |
| MSCI | 2025 | 50.2 | 53.4 | 39.9 | 22.8 |
| CAILA | WACV 2024 | 51.0 | 53.9 | 39.9 | 23.4 |
| Logic-Troika | 2024 | 50.8 | 53.9 | 40.5 | 23.4 |
| CLUSPRO | ICLR 2025 | 52.1 | 54.0 | 40.7 | 23.8 |
| **CAMS** | **2025** | **53.0** | **54.3** | **41.0** | **24.2** |

> **我们的位置**（ViT-B/16，骨干降级）：预计 AUC ~13–17，HM ~30–35，低于同骨干的 CSP ViT-L/14（19.4），**符合预期**。

---

## 五、VAW 排行榜

| 方法 | mAP | mR@15 | F1@15 |
|---|---|---|---|
| ResNet-Baseline（论文） | 63.0 | 52.1 | 63.9 |
| Strong Baseline（论文） | 65.9 | 52.9 | 65.3 |
| SCoNE（论文最优） | 68.3 | 58.3 | 70.3 |
| **我们 YOLOv8m 微调（val）** | **70.9** | — | — |
| **我们 YOLOv8m 微调（test）** | **60.6** | — | — |
| **我们 DINOv2 冻结 MTL（test）** | **55.3** | — | — |

> test mAP 60.6 低于论文 Strong Baseline 65.9。val 70.9 有过拟合。报告据实写，不夸大。

---

## 六、我们各实验与学术界对比（报告表格直接用）

### MIT-States

| 实验 | 方法 | 骨干 | Seen | Unseen | HM | AUC | 说明 |
|---|---|---|---|---|---|---|---|
| Exp A（我们） | ResNet MTL | ResNet50 | ~35%* | ~0% | ~0 | — | 闭集 joint_acc，不可直接比 |
| CLIP zero-shot | 无训练 | ViT-L/14 | 30.2 | 46.0 | 26.1 | 11.0 | 参考点 |
| Exp B（我们） | CSP 复现 | ViT-B/16 | 待评测 | 待评测 | 待评测 | 待评测 | 骨干降级版 |
| CSP 官方 | CSP | ViT-L/14 | 46.6 | 49.9 | 36.3 | 19.4 | 论文结果 |
| SOTA 2025 | CAMS | ViT-L/14 | 53.0 | 54.3 | 41.0 | 24.2 | 天花板 |

*ResNet joint_acc 与 CZSL 指标不可直接比，表中并列仅为展示范式差异。

### VAW

| 实验 | 方法 | test mAP | 说明 |
|---|---|---|---|
| 论文 ResNet-Baseline | ResNet | 63.0 | 参考点 |
| Exp E（我们） | YOLOv8m 微调 | **60.6** | 低于论文 baseline |
| Exp F（我们） | DINOv2 冻结 MTL | **55.3** | 低于 baseline |
| 论文 SOTA SCoNE | SCoNE | 68.3 | 天花板 |

---

## 七、为什么 SOTA 比我们好（报告 §6 素材）

1. **骨干差距（最大因素）**：SOTA 全用 ViT-L/14（307M 参数），我们用 ViT-B/16（86M）或 YOLOv8m（25M）。CLIP ViT-L/14 的语义对齐能力显著更强。AUC 差距约 ~5–7 点可归因于此。

2. **训练规模**：论文通常 A100 × 8，更大 batch、更多 epoch、完整数据增强（text augmentation、retrieval augmentation）。

3. **方法复杂度**：CAMS/Troika 有跨模态融合模块、多路径设计；我们只复现了最简 CSP（提示调优），是 2023 年的方法。

4. **任务本质难度**：MIT-States 标签噪声大（中期报告已记录）、长尾分布严重（top-10 组合样本数 > 200，tail 组合 < 5）——SOTA 也只有 54% Seen acc。

5. **VAW 过拟合**：我们 val mAP 70.9 但 test 60.6，val-test 差 10 点，说明在 20 万条数据上用 YOLOv8m 微调仍过拟合。更强的正则化或骨干冻结可缓解。

---

## 八、关键引用（报告参考文献）

```
[1] Pham et al., "Learning to Predict Visual Attributes in the Wild," CVPR 2021. (VAW 数据集)
[2] Purushwalkam & Gupta, "Compositional Zero-shot Recognition..." ECCV 2019. (MIT-States CZSL 划分)
[3] Nayak et al., "Learning to Compose Soft Prompts for CZSL," ICLR 2023. (CSP，我们复现的方法)
[4] Lu et al., "DFSP: Decomposed Fusion Soft Prompt...," CVPR 2023.
[5] Jiang et al., "Troika: Multi-Path Cross-Modal Traction...," CVPR 2024.
[6] Zheng et al., "CAILA: Concept-Aware Intra-Layer Adapters...," WACV 2024.
[7] Authors, "CAMS: Gated Cross-Attention and Multi-Space...," arXiv 2511.16378, 2025.
[8] Authors, "Compositional Zero-Shot Learning: A Survey," arXiv 2510.11106, 2025.
[9] He et al., "Deep Residual Learning for Image Recognition," CVPR 2016. (ResNet)
[10] Jocher et al., "Ultralytics YOLOv8," 2023. (YOLOv8)
```

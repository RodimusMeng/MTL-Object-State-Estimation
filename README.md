# MTL-Object-State-Estimation 🤖

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> 本项目为《Python人工智能程序设计实践》课程实践项目。基于多任务学习（MTL）框架，实现具身智能场景下的桌面物品“类别”与“物理状态”联合推理估计。

---

## 📌 项目背景 (Background)

在具身智能（Embodied AI）的交互任务中，机器人不仅需要进行基础的语义识别（“这是什么杯子”），更需要感知物体的物理交互属性（“它是空杯还是满载”、“是直立还是倒伏”）。传统的单标签图像分类范式无法捕捉这种复合维度的信息。本项目旨在打破单一感知瓶颈，构建一个支持**端到端联合估计**的视觉感知基线系统。

## 🚀 技术架构 (Architecture)

本项目摒弃了串行识别的冗余设计，采用 **多任务学习 (Multi-Task Learning)** 机制：

1. **共享特征提取骨干 (Shared Feature Backbone)：**
   采用轻量级/经典的卷积神经网络作为主干，对输入图像的浅层纹理与深层语义特征进行统一降维与编码。
2. **任务解耦双分支 (Task-Specific Dual Heads)：**
   在网络高层将特征流进行解耦，切分为两个并行的全连接推理分支：
   - `Branch A`: Object Category Estimation（物品类别估计）
   - `Branch B`: Physical State Estimation（物理状态估计）
3. **梯度平衡与联合优化 (Joint Optimization)：**
   针对双标签数据极易出现的收敛速度不一致问题，项目设计了加权联合损失函数，通过动态/静态调优权重配比，解决长尾状态分布下的梯度主导问题。

## 📊 数据集 (Dataset)

依托经典的 **MIT States Dataset** 核心子集：
- 包含自然场景下的长尾分布物体与状态组合。
- 项目内建了从极度不平衡数据清洗、多标签分层抽样 (Stratified Splitting) 到数据增强的完整自动化流水线 (Data Pipeline)。
- **核心评估指标：** 突破常规单一 Accuracy，引入严格的 **联合准确率 (Joint Accuracy)** 体系。

## 👥 团队架构 (Team Pipeline)

本项目采用高度解耦的流水线作业模式：
- **Data Engineering:** 长尾数据清洗、共现矩阵分析与增强引擎构建。
- **Model Architecture:** MTL 核心框架设计、特征提取器搭建与联合 Loss 优化。
- **Evaluation System:** 多维混淆矩阵构建与双标签联合指标评估。

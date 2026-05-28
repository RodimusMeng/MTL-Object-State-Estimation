# 杯子数据集说明

## 目录结构

```
datasets/
├── cup/                    ← 所有杯子图片，按状态分目录
│   ├── empty/              ← state 0，450 张
│   ├── half/               ← state 1，1315 张
│   ├── full/               ← state 2，1234 张
│   ├── sideways/           ← state 3，200 张（20 原图 + 180 增强）
│   ├── inverted/           ← state 4，200 张（20 原图 + 180 增强）
│   └── yolo/               ← 生成的 YOLOv8 格式（gitignored，可重建）
│       ├── images/train/ val/ test/
│       ├── labels/train/ val/ test/
│       └── cup.yaml
├── cup_other/              ← 新图片来源目录（gitignored）
│   ├── sideways/           ← iPhone 原始照片
│   └── inverted/           ← iPhone 原始照片
├── cup_dataset.csv         ← 主标签文件（image_path 相对于 datasets/）
├── classes_state.json      ← state_id 映射
└── release_dataset/        ← MIT States 原始数据集（gitignored）
```

## 类别定义

| state_id | 目录名 | 含义 |
|---|---|---|
| 0 | empty | 正立，空杯 |
| 1 | half | 正立，半满 |
| 2 | full | 正立，装满 |
| 3 | sideways | 侧躺 |
| 4 | inverted | 倒立 |

## 数据来源

- **state 0/1/2**：网络爬取的裁剪图（原 cup_selected/），约 3000 张
- **state 3/4**：iPhone 16 实拍特写，存于 `cup_other/sideways/` 和 `cup_other/inverted/`，原始各 20 张，增强至 200 张

## 数据增强策略（仅 state 3/4）

| 变换 | 参数 | 原因 |
|---|---|---|
| 水平翻转 | 50% | 侧躺/倒立翻转后语义不变 |
| 亮度 | ×0.6~1.4 | 模拟不同光照 |
| 对比度 | ×0.7~1.3 | 模拟不同拍摄设置 |
| 饱和度 | ×0.7~1.3 | 模拟不同颜色的杯子 |
| 旋转 | ±15° | 模拟手持角度偏差 |
| 随机裁剪 | 保留 80%~100% | 模拟拍摄距离 |
| 高斯模糊 | 30% 概率 | 模拟对焦不实 |

**禁用**：垂直翻转（倒立↔正立语义互换）、旋转 ±90° 以上（姿态歧义）

## 操作脚本

| 脚本 | 作用 |
|---|---|
| `convert_to_yolo.py` | CSV → `cup/yolo/`（YOLOv8 格式） |
| `merge_cup_other.py` | 将 `cup_other/` 新图整合进 `cup/<state>/` |
| `augment_new_states.py` | 对少数类做增强扩充 |
| `migrate_to_cup_dir.py` | 一次性迁移脚本（已执行，勿重复运行） |

## 添加新状态的流程

1. 在 `cup_other/` 下新建英文目录，放入原始图片
2. 在 `merge_cup_other.py` 的 `FOLDER_TO_STATE` 添加映射
3. 在 `classes_state.json` 添加新 state_id
4. 运行 `python datasets/merge_cup_other.py`
5. 若新类少于 200 张，运行 `python datasets/augment_new_states.py`
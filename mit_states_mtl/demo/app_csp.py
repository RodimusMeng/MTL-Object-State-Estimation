"""
物体–状态联合估计 — 综合演示 (Streamlit)
==========================================
三个标签页：
  ① 项目总览：数据集、五个实验的架构与结果、指标预期
  ② MIT-States 实时对比：上传图 → CSP vs CLIP零样本 同图对比 + 零样本泛化演示
  ③ VAW 属性识别：数据集与可识别清单、Exp E/F 架构与结果（静态）

运行:
    streamlit run demo/app_csp.py
依赖: streamlit, torch, clip, ftfy, regex（ai_dev 环境）

技术说明：czsl_csp 与根目录存在同名包(models/datasets)，单进程只能实时加载
MIT-States(CSP)模型；VAW 模型以静态结果呈现。
"""

import os
import sys
from types import SimpleNamespace

# --- Windows: 修复 torch DLL 加载 ---
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
# ---
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSP_DIR = os.path.join(ROOT, "czsl_csp")
sys.path.insert(0, CSP_DIR)

import glob
import json
import re
import time

import numpy as np
import streamlit as st
import torch
from PIL import Image

# ─────────────────────────── 静态数据（来自 SPEC.md，实验最终结果）───────────────────────────

MITSTATES_SPLIT = os.path.join(CSP_DIR, "data", "mit-states", "compositional-split-natural")
CKPT_GLOB = os.path.join(CSP_DIR, "data", "model", "mit-states", "csp_b16", "soft_embeddings*.pt")
VAW_VOCAB = os.path.join(ROOT, "runs", "vaw_mtl", "vocab.json")

RESULTS_MITSTATES = [
    ("Exp A · ResNet MTL 基线",  "ResNet50",   "全微调",            "~35%*", "≈0%", "≈0",  "—",    True),
    ("Exp C · CLIP 零样本",       "ViT-B/16",   "无训练",            "28.2%", "41.1%", "23.6", "8.97", True),
    ("Exp B · CSP（我们复现）",   "ViT-B/16",   "提示微调(骨干冻结)", "43.0%", "46.3%", "33.0", "16.25", True),
    ("参考 · CSP 官方",           "ViT-L/14",   "提示微调",          "46.6%", "49.9%", "36.3", "19.4", False),
    ("参考 · Troika (CVPR'24)",   "ViT-L/14",   "提示+三路融合",      "49.0%", "53.0%", "39.3", "22.1", False),
    ("参考 · CAMS (SOTA'25)",     "ViT-L/14",   "门控注意力+多空间",   "53.0%", "54.3%", "41.0", "24.2", False),
]

RESULTS_VAW = [
    ("Exp E · YOLOv8m + 物体门控", "YOLOv8m-cls", "全微调（物体作输入）",   "60.6%", "—"),
    ("Exp F · DINOv2 + 双头 MTL",  "DINOv2 ViT-S", "冻结特征+头微调（物体作输出）", "55.3%", "obj 35.4%"),
    ("参考 · ResNet Baseline",     "ResNet",      "微调",                  "63.0%", "—"),
    ("参考 · SCoNE (SOTA)",        "ResNet",      "微调+对比学习",          "68.3%", "—"),
]


# ─────────────────────────── 词表/划分加载（纯文本，无需 torch）───────────────────────────

@st.cache_data
def load_mitstates_vocab():
    def parse(fn):
        with open(os.path.join(MITSTATES_SPLIT, fn), encoding="utf-8") as f:
            return [tuple(l.split()) for l in f.read().strip().split("\n")]
    train = parse("train_pairs.txt")
    val = parse("val_pairs.txt")
    test = parse("test_pairs.txt")
    all_pairs = sorted(set(train + val + test))
    states = sorted(set(a for a, o in all_pairs))
    objects = sorted(set(o for a, o in all_pairs))
    seen = set(train)
    unseen = sorted(set(test) - set(train))
    return {"states": states, "objects": objects, "pairs": all_pairs,
            "seen": seen, "unseen": unseen,
            "n_train": len(train), "n_test": len(test)}


@st.cache_data
def load_vaw_vocab():
    with open(VAW_VOCAB, encoding="utf-8") as f:
        v = json.load(f)
    return {"objects": sorted(v["obj2id"].keys()),
            "attrs": sorted(v["attr2id"].keys())}


# ─────────────────────────── CSP + CLIP 模型加载（实时推理）───────────────────────────

def _latest_ckpt():
    cands = glob.glob(CKPT_GLOB)
    if not cands:
        return None
    def epoch_of(p):
        m = re.search(r"epoch_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 1_000_000
    return sorted(cands, key=epoch_of)[-1]


@st.cache_resource(show_spinner="首次加载 CLIP ViT-B/16 + CSP 模型（约 15 秒）…")
def load_models(ckpt_path):
    from datasets.composition_dataset import CompositionDataset, transform_image
    from datasets.read_datasets import DATASET_PATHS
    from models.compositional_modules import get_model
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = SimpleNamespace(experiment_name="csp", dataset="mit-states",
                             clip_model="ViT-B/16", context_length=16,
                             attr_dropout=0.0, lr=5e-5, weight_decay=1e-5)
    dataset = CompositionDataset(DATASET_PATHS["mit-states"], phase="val",
                                 split="compositional-split-natural")
    model, _ = get_model(dataset, config, device)
    soft = torch.load(ckpt_path, weights_only=False)["soft_embeddings"]
    model.set_soft_embeddings(soft.to(device))
    model.eval()

    attr2idx, obj2idx = dataset.attr2idx, dataset.obj2idx
    pairs = dataset.pairs
    pair_idx = torch.tensor([(attr2idx[a], obj2idx[o]) for a, o in pairs]).to(device)

    prompts = [f"a photo of {a.replace('.',' ')} {o.replace('.',' ')}" for a, o in pairs]
    zs_rep = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        toks = clip.tokenize(prompts, context_length=config.context_length).to(device)
        for i in range(0, len(toks), 64):
            feat = model.text_encoder(toks[i:i+64], enable_pos_emb=True)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            zs_rep = torch.cat([zs_rep, feat], dim=0)

    tfm = transform_image("test")
    return model, pairs, pair_idx, zs_rep, tfm, device


@torch.no_grad()
def predict_both(model, pairs, pair_idx, zs_rep, tfm, device, pil_img, topk=5):
    img = tfm(pil_img).unsqueeze(0).to(device)
    feat = model.encode_image(img)
    csp_logits = model(feat, pair_idx)
    csp_p = torch.softmax(csp_logits, dim=-1)[0].float().cpu().numpy()
    feat_n = feat / feat.norm(dim=-1, keepdim=True)
    zs_logits = model.clip_model.logit_scale.exp() * feat_n @ zs_rep.t()
    zs_p = torch.softmax(zs_logits, dim=-1)[0].float().cpu().numpy()

    def top(probs):
        order = probs.argsort()[::-1][:topk]
        return [(pairs[i][0], pairs[i][1], float(probs[i])) for i in order]
    return top(csp_p), top(zs_p)


# ═══════════════════════════════════ 设计系统 ═══════════════════════════════════
st.set_page_config(page_title="物体–状态联合估计", page_icon="🧩", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

/* 全局背景 */
.stApp {
    background: radial-gradient(1200px 600px at 80% -10%, #1e1b4b 0%, transparent 60%),
                radial-gradient(900px 500px at 0% 0%, #312e81 0%, transparent 55%),
                #0b1020;
}
section.main > div { padding-top: 0.5rem; }

/* Hero 横幅 */
.hero {
    background: linear-gradient(110deg, #6366f1 0%, #8b5cf6 45%, #ec4899 100%);
    border-radius: 22px; padding: 34px 40px; margin-bottom: 8px;
    box-shadow: 0 18px 50px -12px rgba(139,92,246,0.55);
    position: relative; overflow: hidden;
}
.hero::after {
    content:""; position:absolute; right:-60px; top:-60px;
    width:260px; height:260px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 70%);
}
.hero h1 {
    font-family:'Inter',sans-serif; font-weight:800; font-size:2.5rem;
    color:#fff; margin:0; letter-spacing:-0.5px; line-height:1.1;
}
.hero .sub { color:rgba(255,255,255,0.92); font-size:1.05rem; margin-top:10px; max-width:760px; }
.hero .tag {
    display:inline-block; background:rgba(255,255,255,0.18); color:#fff;
    padding:4px 13px; border-radius:999px; font-size:0.8rem; margin:10px 8px 0 0;
    backdrop-filter:blur(4px); border:1px solid rgba(255,255,255,0.25);
}

/* 指标卡 */
.metric-row { display:flex; gap:14px; margin:16px 0 4px 0; flex-wrap:wrap; }
.metric-card {
    flex:1; min-width:150px;
    background:linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border:1px solid rgba(255,255,255,0.10); border-radius:16px; padding:18px 20px;
    backdrop-filter:blur(6px);
}
.metric-card .v { font-family:'Inter'; font-weight:800; font-size:1.9rem;
    background:linear-gradient(90deg,#a5b4fc,#f0abfc); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; line-height:1; }
.metric-card .l { color:#94a3b8; font-size:0.82rem; margin-top:8px; }

/* 预测条 */
.pred {
    display:flex; align-items:center; gap:10px; margin:8px 0;
    background:rgba(255,255,255,0.04); border-radius:12px; padding:9px 12px;
    border:1px solid rgba(255,255,255,0.07);
}
.badge { font-size:0.72rem; font-weight:700; padding:3px 9px; border-radius:999px; white-space:nowrap; }
.badge-new  { background:linear-gradient(90deg,#f43f5e,#fb923c); color:#fff; }
.badge-seen { background:rgba(148,163,184,0.25); color:#cbd5e1; }
.pred .name { font-weight:600; color:#e2e8f0; min-width:140px; }
.bar-wrap { flex:1; height:10px; background:rgba(255,255,255,0.08); border-radius:999px; overflow:hidden; }
.bar { height:100%; border-radius:999px; background:linear-gradient(90deg,#6366f1,#ec4899); }
.pct { color:#94a3b8; font-size:0.82rem; min-width:48px; text-align:right; }

/* 标题分隔 */
h2, h3 { color:#e2e8f0 !important; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
.stTabs [aria-selected="true"] { color:#c4b5fd !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def pred_html(top, seen_set):
    rows = ""
    for a, o, p in top:
        new = (a, o) not in seen_set
        badge = ('<span class="badge badge-new">🆕 未见</span>' if new
                 else '<span class="badge badge-seen">✓ 已见</span>')
        rows += (f'<div class="pred">{badge}'
                 f'<span class="name">{a} {o}</span>'
                 f'<div class="bar-wrap"><div class="bar" style="width:{max(p*100,2):.0f}%"></div></div>'
                 f'<span class="pct">{p*100:.1f}%</span></div>')
    return rows


# ─────────────── Hero ───────────────
st.markdown("""
<div class="hero">
  <h1>🧩 物体–状态联合估计</h1>
  <div class="sub">具身智能场景下，让机器同时识别<b>"是什么物体"</b>与<b>"处于什么状态"</b>。
  本项目系统对比三种建模范式 × 四种视觉骨干，并验证组合提示带来的<b>零样本泛化</b>能力。</div>
  <div>
    <span class="tag">MIT-States · VAW</span>
    <span class="tag">ResNet · YOLOv8 · DINOv2 · CLIP</span>
    <span class="tag">CSP (ICLR'23) 复现</span>
    <span class="tag">5 组对比实验</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="metric-row">
  <div class="metric-card"><div class="v">16.25</div><div class="l">CSP AUC（我们 · ViT-B/16）</div></div>
  <div class="metric-card"><div class="v">46.3%</div><div class="l">未见组合准确率（零样本）</div></div>
  <div class="metric-card"><div class="v">≈0%</div><div class="l">ResNet 未见组合（对照）</div></div>
  <div class="metric-card"><div class="v">1962</div><div class="l">已知"状态–物体"组合</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────── Sidebar ───────────────
st.sidebar.markdown("### 🧩 物体–状态联合估计")
st.sidebar.caption("多任务学习 × 组合零样本学习 · 对比研究")
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 一图看懂**")
st.sidebar.markdown(
    "- 任务很难：SOTA 也仅 AUC≈24\n"
    "- 我们 CSP：**AUC 16.25**\n"
    "- 超过更大骨干的 CoOp(13.5)\n"
    "- 核心：**零样本泛化** 46.3%\n  vs 朴素 MTL ≈0%")
st.sidebar.markdown("---")
st.sidebar.success(f"运行设备：{'🟢 CUDA' if torch.cuda.is_available() else '🔵 CPU'}")

tab1, tab2, tab3 = st.tabs(["① 项目总览", "② MIT-States 实时对比", "③ VAW 属性识别"])

# ─────────────── Tab 1：项目总览 ───────────────
with tab1:
    st.subheader("🎯 我们在做什么")
    st.markdown("""
在"物体–状态联合估计"任务上，系统对比**三种建模范式**（朴素联合预测 / 物体条件化 /
组合零样本学习）与**四种视觉骨干**（ResNet / YOLOv8 / DINOv2 / CLIP），揭示：
组合提示范式可使模型对**从未见过的"状态–物体"组合**具备预测能力，而朴素联合预测做不到。
""")

    st.subheader("🗂️ 两个数据集")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**MIT-States（主）**
- 测：**组合零样本泛化**
- 63K 图 · **115 状态 × 245 物体** · 1962 组合
- 用途：Exp A / B / C
- 指标：AUC / HM / Seen / Unseen
""")
    with c2:
        st.markdown("""
**VAW（次 · Visual Genome）**
- 测：**大规模多标签属性预测**
- 72K 图 · 260K 实例 · **610 属性 × 762 物体**
- 用途：Exp E / F
- 指标：Federated mAP
""")

    st.subheader("🏗️ 五个实验的架构")
    st.markdown("""
| 实验 | 范式 | 骨干 | 物体如何处理 | 数据集 |
|---|---|---|---|---|
| **A** | 联合预测 MTL | ResNet50 | 作为**输出**（分类头） | MIT-States |
| **C** | 零样本 | CLIP | 文本提示，无训练 | MIT-States |
| **B** | 组合软提示 | CLIP | 可学习软向量，与状态拼接 | MIT-States |
| **E** | 物体条件化 | YOLOv8m | 作为**输入**（门控调制特征） | VAW |
| **F** | 联合预测 MTL | DINOv2 | 作为**输出**（双头之一） | VAW |
""")

    st.subheader("📈 最终结果")
    st.markdown("**MIT-States（闭世界）** — 加粗为本项目；其余为学术界参考")
    st.markdown(
        "| 方法 | 骨干 | 训练 | Seen | Unseen | HM | AUC |\n|---|---|---|---|---|---|---|\n" +
        "\n".join(
            f"| {'**'+n+'**' if mine else n} | {bb} | {tr} | {s} | {u} | {hm} | {auc} |"
            for (n, bb, tr, s, u, hm, auc, mine) in RESULTS_MITSTATES))
    st.caption("*Exp A 的 35% 为 joint_acc（闭集多分类），与 CZSL 指标不可直接相减。")
    st.markdown("**VAW** — 加粗为本项目")
    st.markdown(
        "| 方法 | 骨干 | 训练 | test mAP | 其他 |\n|---|---|---|---|---|\n" +
        "\n".join(
            f"| {'**'+n+'**' if n.startswith('Exp') else n} | {bb} | {tr} | {m} | {o} |"
            for (n, bb, tr, m, o) in RESULTS_VAW))
    st.info("**四大核心发现**：① CSP 零样本泛化 46.3% vs ResNet ≈0%；"
            "② 微调>冻结（60.6%>55.3%）；③ 物体条件化>联合预测；"
            "④ 骨干降级代价可量化（B/16 vs L/14 差 ~3 AUC）。")

# ─────────────── Tab 2：MIT-States 实时对比 ───────────────
with tab2:
    mv = load_mitstates_vocab()
    st.subheader("🆚 同图对比：CLIP 零样本 vs CSP")
    st.markdown(f"给定一张图，两种方法都在 **{len(mv['pairs'])}** 个已知组合上打分排序。")

    cc1, cc2 = st.columns(2)
    with cc1:
        with st.expander(f"📋 {len(mv['states'])} 种状态"):
            st.write("、".join(mv["states"]))
    with cc2:
        with st.expander(f"📋 {len(mv['objects'])} 种物体"):
            st.write("、".join(mv["objects"]))

    st.markdown("#### 🎯 现场演示「零样本泛化」")
    st.markdown(f"""
测试集有 **{len(mv['unseen'])}** 个组合**训练时从未出现**。CSP 仍能预测（状态与物体分开学习，
组合免费）；朴素分类（ResNet）对这些组合得分恒为 0。**挑一个未见组合 → 找张图 → 上传 →
看 CSP 是否命中 + 旁边的 🆕未见 标记。**
""")
    demo_unseen = [f"{a} {o}" for a, o in mv["unseen"]]
    sample = demo_unseen[::max(1, len(demo_unseen)//30)][:30]
    with st.expander(f"💡 推荐演示用「未见组合」（共 {len(mv['unseen'])} 个）"):
        st.write("、".join(f"`{p}`" for p in sample))

    st.markdown("---")
    ckpt = _latest_ckpt()
    if ckpt is None:
        st.error("未找到 CSP 权重，请先训练。")
    else:
        topk = st.slider("显示 Top-K", 3, 10, 5, key="mit_topk")
        up = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "webp"], key="mit_up")
        if up is not None:
            pil = Image.open(up).convert("RGB")
            model, pairs, pair_idx, zs_rep, tfm, device = load_models(ckpt)
            t0 = time.time()
            csp_top, zs_top = predict_both(model, pairs, pair_idx, zs_rep, tfm, device, pil, topk)
            dt = (time.time() - t0) * 1000

            ci, cc = st.columns([1, 2])
            with ci:
                st.image(pil, caption=f"输入 · 推理 {dt:.0f}ms", use_container_width=True)
            with cc:
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**① CLIP 零样本** · 无训练")
                    st.markdown(pred_html(zs_top, mv["seen"]), unsafe_allow_html=True)
                with cb:
                    st.markdown("**② CSP（我们）** · 组合软提示")
                    st.markdown(pred_html(csp_top, mv["seen"]), unsafe_allow_html=True)
            if csp_top and csp_top[0][:2] not in mv["seen"]:
                st.success("🎉 CSP Top-1 命中了一个「未见组合」——零样本泛化的现场证据！")
        else:
            st.info("👆 上传图片开始对比。建议先试普通物体（苹果/瓶子），再试一张「未见组合」。")

# ─────────────── Tab 3：VAW 属性识别 ───────────────
with tab3:
    vv = load_vaw_vocab()
    st.subheader("🌐 VAW · 大规模属性识别")
    st.markdown("""
VAW 更大更真实（72K 图、260K 实例）。我们对比**物体条件化**（Exp E，物体作输入）
与**联合预测**（Exp F，物体作输出）两种范式。

> 技术说明：VAW 模型与 CSP 代码存在同名包冲突，单进程无法同时实时加载，故本页以**架构与结果静态呈现**。
""")
    cc1, cc2 = st.columns(2)
    with cc1:
        with st.expander(f"📋 {len(vv['attrs'])} 种属性/状态"):
            st.write("、".join(vv["attrs"]))
    with cc2:
        with st.expander(f"📋 {len(vv['objects'])} 种物体"):
            st.write("、".join(vv["objects"]))

    st.markdown("#### 🏗️ 两个 VAW 实验")
    st.markdown("""
**Exp E · YOLOv8m + 物体门控**（物体作输入）
```
图像 → YOLOv8m 骨干 → 1280维特征 ┐
物体ID → Embedding → MLP门控(sigmoid) ┘→ 特征⊙门控 → 属性头 → 610维属性
```
全微调骨干 · test mAP **60.6%**

**Exp F · DINOv2 + 双头 MTL**（物体作输出，真·多任务）
```
图像 → DINOv2(冻结) → 384维特征 → 共享Trunk ┬→ 物体头(762类)
                                            └→ 属性头(610类)
        Kendall 不确定性加权联合损失
```
骨干冻结，仅训练头部 · test attr mAP **55.3%** / obj acc **35.4%**
""")
    st.markdown(
        "| 方法 | 骨干 | 训练 | test mAP | 其他 |\n|---|---|---|---|---|\n" +
        "\n".join(
            f"| {'**'+n+'**' if n.startswith('Exp') else n} | {bb} | {tr} | {m} | {o} |"
            for (n, bb, tr, m, o) in RESULTS_VAW))
    st.info("**VAW 结论**：微调(60.6%) > 冻结(55.3%)；物体条件化 > 物体联合预测。")

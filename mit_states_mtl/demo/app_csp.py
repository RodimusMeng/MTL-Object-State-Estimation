"""
物体–状态联合估计 — 综合演示 (Streamlit)
==========================================
明亮编辑风单页：整屏起始页（大标题 + 真实样例瀑布拼贴）→ 下滑渐次涌现。
折叠侧栏含极简「显示」开关：繁/简 + 字体（MetroSung 港铁宋 / 思源 / 无衬线）。
运行: streamlit run demo/app_csp.py
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

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DEMO_DIR)
CSP_DIR = os.path.join(ROOT, "czsl_csp")
# 两套包名互不冲突：CSP 用 csp_data/csp_models（已重命名），VAW 用根目录 models/datasets
sys.path.insert(0, CSP_DIR)
sys.path.insert(0, ROOT)

import base64
import glob
import io
import json
import re
import time

import numpy as np
import streamlit as st
import torch
from PIL import Image

MITSTATES_SPLIT = os.path.join(CSP_DIR, "data", "mit-states", "compositional-split-natural")
MITSTATES_IMG = os.path.join(CSP_DIR, "data", "mit-states", "images")
CKPT_GLOB = os.path.join(CSP_DIR, "data", "model", "mit-states", "csp_b16", "soft_embeddings*.pt")
VAW_VOCAB = os.path.join(ROOT, "runs", "vaw_mtl", "vocab.json")
VAW_DINO_CKPT = os.path.join(ROOT, "runs", "vaw_mtl_dino", "best.pt")
VAW_FEAT_META = os.path.join(ROOT, "datasets", "vaw_features", "meta.json")
FONT_SUBSET = os.path.join(DEMO_DIR, "fonts", "MetroSung.subset.woff2")
FONT_FULL = os.path.join(DEMO_DIR, "fonts", "MetroSung_v1.0_FinalVersion.otf")

# 实验重新编号：MIT-States(1基线→2零样本→3核心) · VAW(4条件化 · 5联合预测)
RESULTS_MITSTATES = [
    ("实验1 · ResNet MTL 基线",  "ResNet50",   "全微调",            "~35%*", "≈0%", "≈0",  "—",    True),
    ("实验2 · CLIP 零样本",       "ViT-B/16",   "无训练",            "28.2%", "41.1%", "23.6", "8.97", True),
    ("实验3 · CSP（我们复现）",   "ViT-B/16",   "提示微调",          "43.0%", "46.3%", "33.0", "16.25", True),
    ("CSP 官方（参考）",          "ViT-L/14",   "提示微调",          "46.6%", "49.9%", "36.3", "19.4", False),
    ("Troika（CVPR·24 参考）",    "ViT-L/14",   "提示+三路融合",      "49.0%", "53.0%", "39.3", "22.1", False),
    ("CAMS（SOTA·25 参考）",      "ViT-L/14",   "门控+多空间",        "53.0%", "54.3%", "41.0", "24.2", False),
]
RESULTS_VAW = [
    ("实验4 · YOLOv8m + 物体门控", "YOLOv8m-cls", "全微调（物体作输入）",   "60.6%", "—", True),
    ("实验5 · DINOv2 + 双头 MTL",  "DINOv2 ViT-S", "冻结+头微调（物体作输出）", "55.3%", "obj 35.4%", True),
    ("ResNet Baseline（参考）",     "ResNet",      "微调",                  "63.0%", "—", False),
    ("SCoNE（SOTA 参考）",          "ResNet",      "微调+对比",              "68.3%", "—", False),
]
HERO_CANDIDATES = [
    "sliced_orange", "ripe_tomato", "rusty_chain", "wilted_flower", "melted_chocolate",
    "cracked_egg", "mossy_branch", "peeled_banana", "burnt_candle", "frozen_lake",
    "cooked_pasta", "broken_glass", "muddy_dog", "splintered_wood", "dewy_grass",
    "sliced_apple", "rusty_car", "wrinkled_paper", "spilled_paint", "engraved_ring",
]


# ─────────────────────────── 数据/模型加载 ───────────────────────────
@st.cache_data
def load_mitstates_vocab():
    def parse(fn):
        with open(os.path.join(MITSTATES_SPLIT, fn), encoding="utf-8") as f:
            return [tuple(l.split()) for l in f.read().strip().split("\n")]
    train, val, test = parse("train_pairs.txt"), parse("val_pairs.txt"), parse("test_pairs.txt")
    all_pairs = sorted(set(train + val + test))
    return {"states": sorted(set(a for a, o in all_pairs)),
            "objects": sorted(set(o for a, o in all_pairs)),
            "pairs": all_pairs, "seen": set(train),
            "unseen": sorted(set(test) - set(train))}


@st.cache_data
def load_vaw_vocab():
    with open(VAW_VOCAB, encoding="utf-8") as f:
        v = json.load(f)
    return {"objects": sorted(v["obj2id"].keys()), "attrs": sorted(v["attr2id"].keys())}


@st.cache_data(show_spinner=False)
def hero_samples(n=6):
    out = []
    for name in HERO_CANDIDATES:
        d = os.path.join(MITSTATES_IMG, name)
        if not os.path.isdir(d):
            continue
        for f in [x for x in os.listdir(d) if x.lower().endswith((".jpg", ".jpeg", ".png"))][:6]:
            try:
                im = Image.open(os.path.join(d, f)).convert("RGB")
                if min(im.size) < 150:
                    continue
                s = min(im.size)
                im = im.crop(((im.width - s)//2, (im.height - s)//2,
                              (im.width - s)//2 + s, (im.height - s)//2 + s)).resize((300, 300))
                buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
                out.append((base64.b64encode(buf.getvalue()).decode(), name.replace("_", " ")))
                break
            except Exception:
                continue
        if len(out) >= n:
            break
    return out


@st.cache_data(show_spinner=False)
def font_face_css():
    path, fmt = (FONT_SUBSET, "woff2") if os.path.exists(FONT_SUBSET) else \
                ((FONT_FULL, "opentype") if os.path.exists(FONT_FULL) else (None, None))
    if not path:
        return ""
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return (f"<style>@font-face{{font-family:'MetroSung';"
            f"src:url(data:font/{'woff2' if fmt=='woff2' else 'otf'};base64,{b64}) format('{fmt}');"
            f"font-display:swap;}}</style>")


def _latest_ckpt():
    cands = glob.glob(CKPT_GLOB)
    if not cands:
        return None
    def ep(p):
        m = re.search(r"epoch_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 1_000_000
    return sorted(cands, key=ep)[-1]


@st.cache_resource(show_spinner="首次加载 CLIP + CSP（约 15 秒）…")
def load_models(ckpt_path):
    from csp_data.composition_dataset import CompositionDataset, transform_image
    from csp_data.read_datasets import DATASET_PATHS
    from csp_models.compositional_modules import get_model
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = SimpleNamespace(experiment_name="csp", dataset="mit-states",
                             clip_model="ViT-B/16", context_length=16,
                             attr_dropout=0.0, lr=5e-5, weight_decay=1e-5)
    dataset = CompositionDataset(DATASET_PATHS["mit-states"], phase="val",
                                 split="compositional-split-natural")
    model, _ = get_model(dataset, config, device)
    model.set_soft_embeddings(torch.load(ckpt_path, weights_only=False)["soft_embeddings"].to(device))
    model.eval()
    attr2idx, obj2idx = dataset.attr2idx, dataset.obj2idx
    pairs = dataset.pairs
    pair_idx = torch.tensor([(attr2idx[a], obj2idx[o]) for a, o in pairs]).to(device)
    prompts = [f"a photo of {a.replace('.',' ')} {o.replace('.',' ')}" for a, o in pairs]
    zs = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        toks = clip.tokenize(prompts, context_length=config.context_length).to(device)
        for i in range(0, len(toks), 64):
            f = model.text_encoder(toks[i:i+64], enable_pos_emb=True)
            zs = torch.cat([zs, f / f.norm(dim=-1, keepdim=True)], dim=0)
    return model, pairs, pair_idx, zs, transform_image("test"), device


@torch.no_grad()
def predict_both(model, pairs, pair_idx, zs_rep, tfm, device, pil, topk=5):
    img = tfm(pil).unsqueeze(0).to(device)
    feat = model.encode_image(img)
    csp = torch.softmax(model(feat, pair_idx), dim=-1)[0].float().cpu().numpy()
    fn = feat / feat.norm(dim=-1, keepdim=True)
    zs = torch.softmax(model.clip_model.logit_scale.exp() * fn @ zs_rep.t(), dim=-1)[0].float().cpu().numpy()
    def top(p):
        return [(pairs[i][0], pairs[i][1], float(p[i])) for i in p.argsort()[::-1][:topk]]
    return top(csp), top(zs)


# ── VAW 实时推理（实验5：DINOv2 冻结 + 双头 MTL，图→物体+属性）──
@st.cache_resource(show_spinner="首次加载 DINOv2 + 实验5 模型（约 15 秒）…")
def load_vaw_model():
    import timm
    from models.feat_mtl import FeatMTL  # noqa: 供 torch.load 反序列化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                             num_classes=0, img_size=224).eval().to(device)
    for p in dino.parameters():
        p.requires_grad = False
    model = torch.load(VAW_DINO_CKPT, map_location=device, weights_only=False).eval()
    with open(VAW_FEAT_META, encoding="utf-8") as f:
        meta = json.load(f)
    id2obj = {v: k for k, v in meta["obj2id"].items()}
    id2attr = {v: k for k, v in meta["attr2id"].items()}
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return dino, model, id2obj, id2attr, mean, std, device


@torch.no_grad()
def predict_vaw(pil, topk_obj=3, topk_attr=8):
    dino, model, id2obj, id2attr, mean, std, device = load_vaw_model()
    im = pil.resize((224, 224))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    x = (x - mean) / std
    feat = dino(x)
    obj_logits, attr_logits = model(feat)
    obj_p = torch.softmax(obj_logits, dim=-1)[0].float().cpu().numpy()
    attr_p = torch.sigmoid(attr_logits)[0].float().cpu().numpy()
    objs = [(id2obj[i], float(obj_p[i])) for i in obj_p.argsort()[::-1][:topk_obj]]
    attrs = [(id2attr[i], float(attr_p[i])) for i in attr_p.argsort()[::-1][:topk_attr]]
    return objs, attrs


# ═══════════════════════════════════ 页面 ═══════════════════════════════════
st.set_page_config(page_title="物体–状态联合估计", page_icon="◐",
                   layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700;900&family=Noto+Sans+SC:wght@400;500&display=swap');
:root{ --ink:#1b1b1d; --muted:#73726c; --paper:#f4f1ea; --line:#e2ddd0; --accent:#2f5d50;
       --serif:'MetroSung','Source Han Serif SC','Noto Serif SC','Songti SC',Georgia,serif;
       /* 港铁规范：英文 Helvetica/Myriad，中文地铁宋 */
       --sans:'Helvetica Neue',Helvetica,Arial,'Noto Sans SC','PingFang SC',system-ui,sans-serif;
       --body:'Helvetica Neue',Helvetica,Arial,var(--serif); }
.stApp{ background:var(--paper); -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; }
header[data-testid="stHeader"]{ background:transparent; }  /* 透明而非隐藏：保留 Deploy 与侧边栏开关 */
footer{ visibility:hidden; }
.block-container{ padding-top:0 !important; padding-bottom:5rem; max-width:1120px; }
/* font-synthesis:none → 禁止浏览器伪造粗体，保住地铁宋的锐利收笔 */
html, body, [class*="css"]{ color:var(--ink); font-family:var(--body); font-synthesis:none; }
.serif{ font-family:var(--serif); }

.hero{ min-height:94vh; display:flex; flex-direction:column; justify-content:center; position:relative; padding-right:38vw; }
.hero .kicker{ font-size:0.82rem; letter-spacing:0.4em; text-transform:uppercase; color:var(--accent);
       font-weight:700; margin-bottom:30px; opacity:0; animation:rise .9s .05s cubic-bezier(.2,.7,.2,1) forwards; }
.hero h1{ font-family:var(--serif); font-weight:500; font-size:clamp(3.4rem,9vw,7rem); line-height:1.24;
       letter-spacing:2px; margin:0; color:var(--ink); }
.hero h1 .l1{ display:block; opacity:0; animation:rise 1.1s .16s cubic-bezier(.16,1,.3,1) forwards; }
.hero h1 .l2{ display:block; color:var(--accent); margin-top:0.12em; opacity:0; animation:rise 1.1s .34s cubic-bezier(.16,1,.3,1) forwards; }
.hero .rule{ width:0; height:3px; background:var(--accent); margin:34px 0 30px; animation:grow 1s .7s cubic-bezier(.2,.7,.2,1) forwards; }
.hero .lede{ font-size:clamp(1.05rem,1.5vw,1.28rem); color:#46453f; max-width:560px; line-height:1.85;
       opacity:0; animation:rise 1s .6s cubic-bezier(.2,.7,.2,1) forwards; }
.hero .meta{ margin-top:40px; display:flex; gap:36px; flex-wrap:wrap; opacity:0; animation:rise 1s .82s cubic-bezier(.2,.7,.2,1) forwards; }
.hero .meta .n{ font-family:var(--serif); font-weight:700; font-size:1.7rem; }
.hero .meta .c{ font-size:0.78rem; color:var(--muted); letter-spacing:0.05em; margin-top:2px; }
.scroll-cue{ margin-top:54px; color:var(--muted); font-size:0.78rem; letter-spacing:0.22em; opacity:0; animation:fade 1.2s 1.3s forwards; }
.scroll-cue span{ display:inline-block; margin-left:8px; animation:bob 1.8s ease-in-out infinite; }

.gallery{ position:absolute; right:-2vw; top:50%; transform:translateY(-50%); width:36vw; display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.gallery .col{ display:flex; flex-direction:column; gap:16px; }
.gallery .col:nth-child(2){ margin-top:46px; }
.tile{ position:relative; border-radius:16px; overflow:hidden; box-shadow:0 14px 34px -16px rgba(40,35,25,0.5); opacity:0; transform:translateY(34px) rotate(var(--r,0deg)); }
.tile img{ width:100%; display:block; aspect-ratio:1; object-fit:cover; }
.tile .cap{ position:absolute; left:0; right:0; bottom:0; padding:14px 12px 9px; background:linear-gradient(transparent, rgba(20,16,10,0.72)); color:#fff; font-size:0.82rem; font-weight:600; }
.tile:nth-child(1){ --r:-1.5deg; animation:tilein .9s .5s cubic-bezier(.16,1,.3,1) forwards; }
.tile:nth-child(2){ --r:1.2deg;  animation:tilein .9s .68s cubic-bezier(.16,1,.3,1) forwards; }
.tile:nth-child(3){ --r:-1deg;   animation:tilein .9s .86s cubic-bezier(.16,1,.3,1) forwards; }
@media (max-width:1000px){ .hero{ padding-right:0; } .gallery{ display:none; } }

.reveal{ opacity:1; }
@supports (animation-timeline: view()){
  .reveal{ opacity:0; animation:rise 1s linear forwards; animation-timeline:view(); animation-range:entry 3% cover 24%; } }
.section{ padding:78px 0 8px; border-top:1px solid var(--line); margin-top:64px; }
.eyebrow{ font-size:0.78rem; letter-spacing:0.3em; text-transform:uppercase; color:var(--accent); font-weight:700; margin-bottom:16px; }
.section h2{ font-family:var(--serif); font-weight:700; font-size:clamp(2rem,3.6vw,2.9rem); letter-spacing:0.5px; margin:0 0 22px; }
.section p.body{ font-size:1.08rem; line-height:1.9; color:#3a3a3c; max-width:740px; }
.cards{ display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:8px; }
.card{ background:#fbfaf5; border:1px solid var(--line); border-radius:18px; padding:26px 28px; }
.card h3{ font-family:var(--serif); font-weight:700; font-size:1.32rem; margin:0 0 6px; }
.card .tag{ font-size:0.72rem; letter-spacing:0.16em; color:var(--accent); text-transform:uppercase; font-weight:700; }
.card ul{ margin:14px 0 0; padding-left:18px; color:#4a4a4c; line-height:1.8; font-size:0.95rem; }
.explain{ display:grid; grid-template-columns:140px 1fr; gap:4px 24px; margin-top:6px; }
.explain .k{ font-family:var(--serif); font-weight:700; color:var(--accent); padding:16px 0; border-bottom:1px solid var(--line); font-size:1.05rem; }
.explain .v{ padding:16px 0; border-bottom:1px solid var(--line); color:#3a3a3c; line-height:1.75; }
.explain .v b{ color:var(--ink); }
table.t{ width:100%; border-collapse:collapse; margin-top:10px; font-size:0.92rem; }
table.t th{ text-align:left; font-weight:700; color:var(--muted); font-size:0.76rem; letter-spacing:0.06em; text-transform:uppercase; padding:10px 12px; border-bottom:2px solid var(--line); }
table.t td{ padding:11px 12px; border-bottom:1px solid var(--line); color:#3a3a3c; }
table.t tr.mine td{ background:#eaf1ed; color:var(--ink); font-weight:700; }
table.t tr.mine td:first-child{ box-shadow:inset 3px 0 0 var(--accent); }
.note{ font-size:0.85rem; color:var(--muted); margin-top:10px; line-height:1.7; }
.pred-card{ background:#fbfaf5; border:1px solid var(--line); border-radius:16px; padding:18px 20px; }
.pred-card h4{ margin:0 0 4px; font-size:1.02rem; }
.pred-card .desc{ color:var(--muted); font-size:0.82rem; margin-bottom:12px; }
.prow{ display:flex; align-items:center; gap:10px; margin:9px 0; }
.bdg{ font-size:0.68rem; font-weight:700; padding:3px 0; border-radius:6px; white-space:nowrap;
       display:inline-block; min-width:42px; text-align:center; box-sizing:border-box; }
.bdg-new{ background:#e7f0eb; color:var(--accent); border:1px solid #cfe0d8; }
.bdg-seen{ background:#efeee9; color:#8a8a8f; }
.pname{ font-weight:600; color:var(--ink); min-width:130px; font-size:0.92rem; }
.pbar{ flex:1; height:7px; background:#e6e3da; border-radius:999px; overflow:hidden; }
.pfill{ height:100%; background:var(--accent); border-radius:999px; }
.ppct{ color:var(--muted); font-size:0.82rem; min-width:46px; text-align:right; }
@keyframes rise{ from{opacity:0; transform:translateY(28px);} to{opacity:1; transform:translateY(0);} }
@keyframes tilein{ to{opacity:1; transform:translateY(0) rotate(var(--r,0deg));} }
@keyframes fade{ to{opacity:1;} }
@keyframes grow{ to{width:84px;} }
@keyframes bob{ 0%,100%{transform:translateY(0);} 50%{transform:translateY(6px);} }
</style>
"""
st.markdown(font_face_css(), unsafe_allow_html=True)
st.markdown(CSS, unsafe_allow_html=True)

# ── 极简「显示」开关（折叠侧栏 = 隐蔽角落）──
_SERIF = {
    "MetroSung 港铁宋": "'MetroSung','Noto Serif SC','Songti SC',serif",
    "思源宋体": "'Source Han Serif SC','Noto Serif SC','Songti SC',serif",
    "系统无衬线": "'Noto Sans SC','PingFang SC','Microsoft YaHei',sans-serif",
}
with st.sidebar:
    st.markdown("#### ◐ 显示")
    script = st.radio("字形", ["简", "繁"], index=0, horizontal=True)
    fontname = st.radio("标题字体", list(_SERIF.keys()), index=0)


@st.cache_resource(show_spinner=False)
def _converter():
    import opencc
    return opencc.OpenCC("s2hk")  # 简→繁（香港变体，贴合港铁）


def L(s: str) -> str:
    """繁体模式下转换中文（仅转汉字，HTML/英文/数字不动）。"""
    if script == "繁":
        try:
            return _converter().convert(s)
        except Exception:
            return s
    return s


# 覆盖 serif 栈（字体选择）
st.markdown(f"<style>:root{{--serif:{_SERIF[fontname]};}}</style>", unsafe_allow_html=True)

# ── 起始页 ──
samples = hero_samples(6)
if samples:
    half = (len(samples) + 1) // 2
    g1 = "".join(f'<div class="tile"><img src="data:image/jpeg;base64,{b}"><div class="cap">{c}</div></div>' for b, c in samples[:half])
    g2 = "".join(f'<div class="tile"><img src="data:image/jpeg;base64,{b}"><div class="cap">{c}</div></div>' for b, c in samples[half:])
    gallery = f'<div class="gallery"><div class="col">{g1}</div><div class="col">{g2}</div></div>'
else:
    gallery = ""

st.markdown(gallery + L(f"""
<section class="hero">
  <div class="kicker">具身智能 · 视觉感知</div>
  <h1 class="serif"><span class="l1">物体与状态</span><span class="l2">联合估计</span></h1>
  <div class="rule"></div>
  <div class="lede">让机器在看见物体的同时，理解它"正处于什么状态"——切开的橙子、生锈的链条、枯萎的花。
  我们系统对比三种范式、四种骨干，并验证组合提示带来的<b>零样本泛化</b>。</div>
  <div class="meta">
    <div><div class="n serif">3 × 4</div><div class="c">范式 × 骨干</div></div>
    <div><div class="n serif">2</div><div class="c">公开数据集</div></div>
    <div><div class="n serif">5</div><div class="c">对比实验</div></div>
  </div>
  <div class="scroll-cue">向下滚动<span>↓</span></div>
</section>
"""), unsafe_allow_html=True)

# ── 研究问题 ──
st.markdown(L("""
<div class="section reveal">
  <div class="eyebrow">研究问题</div>
  <h2 class="serif">从"是什么"到"怎么样"</h2>
  <p class="body">传统图像分类只回答"这是什么物体"。但在具身智能场景中，机器人更需要感知物体的
  <b>物理状态</b>——杯子是空是满、门是开是关、面包是新鲜还是发霉。本项目把这一任务拆成三种
  建模范式来对比：<b>朴素联合预测</b>、<b>物体条件化</b>、<b>组合零样本学习</b>。核心发现是：
  组合范式让模型能识别<b>训练时从未见过的"状态–物体"组合</b>，而朴素方法做不到。</p>
</div>"""), unsafe_allow_html=True)

# ── 两个数据集 ──
st.markdown(L("""
<div class="section reveal">
  <div class="eyebrow">数据基础</div>
  <h2 class="serif">两个数据集，两个维度</h2>
  <div class="cards">
    <div class="card"><div class="tag">主 · 泛化边界</div><h3 class="serif">MIT-States</h3>
      <ul><li>测组合零样本泛化能力</li><li>63K 图 · 115 状态 × 245 物体</li>
      <li>1962 个已知组合，含未见组合</li><li>指标：AUC / HM / Seen / Unseen</li></ul></div>
    <div class="card"><div class="tag">次 · 规模上限</div><h3 class="serif">VAW</h3>
      <ul><li>测大规模多标签属性预测</li><li>72K 图 · 260K 实例</li>
      <li>610 属性 × 762 物体</li><li>指标：Federated mAP</li></ul></div>
  </div>
</div>"""), unsafe_allow_html=True)

# ── 五个实验 + 每个在讲什么 ──
exp_rows = "".join(
    f"<tr><td>{e}</td><td>{para}</td><td>{bb}</td><td>{ds}</td></tr>" for e, para, bb, ds in [
        ("实验1", "联合预测 MTL", "ResNet50", "MIT-States"),
        ("实验2", "零样本", "CLIP ViT-B/16", "MIT-States"),
        ("实验3", "组合软提示 (CSP)", "CLIP ViT-B/16", "MIT-States"),
        ("实验4", "物体条件化", "YOLOv8m", "VAW"),
        ("实验5", "联合预测 MTL", "DINOv2", "VAW"),
    ])
st.markdown(L(f"""
<div class="section reveal">
  <div class="eyebrow">方法 · 五个实验在讲什么</div>
  <h2 class="serif">同一任务，五种思路</h2>
  <table class="t"><tr><th>实验</th><th>范式</th><th>骨干</th><th>数据集</th></tr>{exp_rows}</table>
  <div class="explain" style="margin-top:30px;">
    <div class="k">实验1</div><div class="v"><b>朴素联合预测</b>：ResNet 上接"物体头"和"状态头"一起训。
      最直觉，但它把每个"状态+物体"当独立类别——没见过的组合直接抓瞎（Unseen≈0）。</div>
    <div class="k">实验2</div><div class="v"><b>CLIP 零样本</b>：完全不训练，直接问预训练 CLIP
      "这张图是不是某个状态的某个物体"。用来看预训练知识的天花板在哪。</div>
    <div class="k">实验3</div><div class="v"><b>组合软提示 (CSP)</b>：给每个状态词、物体词各学一个"软向量"，
      用时拼起来。因为分开学，<b>没见过的组合也能拼出来</b> → 这就是零样本泛化的来源。</div>
    <div class="k">实验4</div><div class="v"><b>物体条件化</b>：把物体身份当作<b>已知输入</b>，用它生成门控去调制
      图像特征，再预测属性。物体不是要预测的目标，而是辅助线索。</div>
    <div class="k">实验5</div><div class="v"><b>真·多任务</b>：共享一份冻结特征，一个头预测物体、一个头预测属性，
      用 Kendall 不确定性加权自动平衡两个任务。</div>
  </div>
</div>"""), unsafe_allow_html=True)

# ── 指标怎么读 ──
st.markdown(L("""
<div class="section reveal">
  <div class="eyebrow">如何读懂数字</div>
  <h2 class="serif">先看懂指标，再看结果</h2>
  <p class="body">组合零样本学习用一套专门的指标。看懂它们，下一页的数字才有意义——
  尤其要理解：为什么"准确率高"不等于"好模型"。</p>
  <div class="explain">
    <div class="k">Seen</div><div class="v">测试图属于"<b>训练见过</b>的组合"时答对的比例。衡量拟合/记忆能力。</div>
    <div class="k">Unseen</div><div class="v">测试图属于"<b>训练没见过</b>的组合"时答对的比例。<b>衡量零样本泛化——本项目最关心的能力</b>。</div>
    <div class="k">HM</div><div class="v">Seen 与 Unseen 的调和平均 2·S·U/(S+U)。<b>主指标</b>：只会背训练集的模型（Unseen≈0）会被惩罚到接近 0。</div>
    <div class="k">AUC</div><div class="v">扫描"偏向已见 / 偏向未见"的阈值，画出 Seen–Unseen 权衡曲线，取曲线下面积。<b>不依赖单一阈值，最综合</b>。</div>
    <div class="k">mAP</div><div class="v">VAW 用。每个属性单独算平均精度(AP)再求平均；Federated 版只在"有明确标注"的样本上算。</div>
  </div>
  <p class="note">关键直觉：MIT-States 上 SOTA 的 AUC 也仅约 24——这是个公认很难的任务，别用日常"准确率 90%+"的直觉去看这些数字。</p>
</div>"""), unsafe_allow_html=True)

# ── 实时演示 ──
st.markdown(L("""
<div class="section reveal">
  <div class="eyebrow">实时演示</div>
  <h2 class="serif">同图对比：零样本 vs 组合提示</h2>
  <p class="body">上传一张图，两种方法都在全部已知"状态–物体"组合上排序。若你上传一张
  <b>训练时未见过的组合</b>，观察 CSP 能否命中（旁标「未见」）——这就是零样本泛化的现场证据。</p>
</div>"""), unsafe_allow_html=True)

mv = load_mitstates_vocab()
with st.expander(L(f"可识别清单：{len(mv['states'])} 状态 · {len(mv['objects'])} 物体 · {len(mv['unseen'])} 个未见组合可试")):
    cc1, cc2 = st.columns(2)
    cc1.markdown(L("**状态**") + "\n\n" + "、".join(mv["states"]))
    cc2.markdown(L("**物体**") + "\n\n" + "、".join(mv["objects"]))
    st.markdown(L("**推荐演示用「未见组合」**") + "\n\n" +
                "、".join(f"`{a} {o}`" for a, o in mv["unseen"][::max(1, len(mv['unseen'])//24)][:24]))

ckpt = _latest_ckpt()
if ckpt is None:
    st.warning(L("未找到 CSP 权重，请先训练。"))
else:
    topk = st.slider(L("显示 Top-K"), 3, 8, 5)
    up = st.file_uploader(L("上传图片"), type=["jpg", "jpeg", "png", "webp"])
    if up is not None:
        pil = Image.open(up).convert("RGB")
        model, pairs, pair_idx, zs_rep, tfm, device = load_models(ckpt)
        t0 = time.time()
        csp_top, zs_top = predict_both(model, pairs, pair_idx, zs_rep, tfm, device, pil, topk)
        dt = (time.time() - t0) * 1000

        def rows(top):
            h = ""
            for a, o, p in top:
                new = (a, o) not in mv["seen"]
                bd = (f'<span class="bdg bdg-new">{L("未见")}</span>' if new
                      else f'<span class="bdg bdg-seen">{L("已见")}</span>')
                h += (f'<div class="prow">{bd}<span class="pname">{a} {o}</span>'
                      f'<div class="pbar"><div class="pfill" style="width:{max(p*100,2):.0f}%"></div></div>'
                      f'<span class="ppct">{p*100:.1f}%</span></div>')
            return h

        ci, cc = st.columns([1, 1.6])
        with ci:
            st.image(pil, caption=L(f"输入 · 推理 {dt:.0f}ms · CLIP ViT-B/16"), use_container_width=True)
        with cc:
            a, b = st.columns(2)
            a.markdown(L(f'<div class="pred-card"><h4>CLIP 零样本</h4><div class="desc">无训练 · 纯预训练知识</div>{rows(zs_top)}</div>'), unsafe_allow_html=True)
            b.markdown(L(f'<div class="pred-card"><h4>CSP（我们）</h4><div class="desc">训练了组合软提示</div>{rows(csp_top)}</div>'), unsafe_allow_html=True)
        if csp_top and csp_top[0][:2] not in mv["seen"]:
            st.success(L("CSP Top-1 命中了一个「未见组合」——零样本泛化的现场证据。"))
    else:
        st.info(L("上传一张图片开始。建议先试普通物体（苹果 / 瓶子），再试一个「未见组合」。"))

# ── 结果 ──
def res_table(rows, headers, mine_idx):
    body = ""
    for r in rows:
        cells = "".join(f"<td>{c}</td>" for c in r[:mine_idx])
        body += f'<tr class="{"mine" if r[mine_idx] else ""}">{cells}</tr>'
    head = "".join(f"<th>{h}</th>" for h in headers)
    return f'<table class="t"><tr>{head}</tr>{body}</table>'

# ── VAW 实时识别（实验5：DINOv2 双头）──
st.markdown(L("""
<div class="section reveal">
  <div class="eyebrow">实时演示 · VAW</div>
  <h2 class="serif">大规模属性识别（实验5 · DINOv2 双头）</h2>
  <p class="body">第二个数据集 VAW 更大更真实。这里用实验5（DINOv2 冻结 + 双头 MTL）实时识别：
  上传一张图 → <b>同时</b>预测物体类别与多个属性（762 物体 / 610 属性）。</p>
</div>"""), unsafe_allow_html=True)

vv = load_vaw_vocab()
with st.expander(L(f"VAW 可识别清单：{len(vv['attrs'])} 属性 · {len(vv['objects'])} 物体")):
    c1, c2 = st.columns(2)
    c1.markdown(L("**属性**") + "\n\n" + "、".join(vv["attrs"]))
    c2.markdown(L("**物体**") + "\n\n" + "、".join(vv["objects"]))

if not os.path.exists(VAW_DINO_CKPT):
    st.warning(L("未找到 VAW 实验5 权重（runs/vaw_mtl_dino/best.pt）。"))
else:
    upv = st.file_uploader(L("上传图片（VAW 实时识别）"), type=["jpg", "jpeg", "png", "webp"], key="vaw_up")
    if upv is not None:
        pilv = Image.open(upv).convert("RGB")
        t0 = time.time(); objs, attrs = predict_vaw(pilv); dt = (time.time() - t0) * 1000

        def vrows(items):
            h = ""
            for name, p in items:
                h += (f'<div class="prow"><span class="pname">{name}</span>'
                      f'<div class="pbar"><div class="pfill" style="width:{max(p*100,2):.0f}%"></div></div>'
                      f'<span class="ppct">{p*100:.0f}%</span></div>')
            return h

        civ, ccv = st.columns([1, 1.6])
        with civ:
            st.image(pilv, caption=L(f"输入 · 推理 {dt:.0f}ms · DINOv2 ViT-S"), use_container_width=True)
        with ccv:
            av, bv = st.columns(2)
            av.markdown(L(f'<div class="pred-card"><h4>物体 Top-3</h4><div class="desc">单标签 softmax</div>{vrows(objs)}</div>'), unsafe_allow_html=True)
            bv.markdown(L(f'<div class="pred-card"><h4>属性 Top-8</h4><div class="desc">多标签 sigmoid</div>{vrows(attrs)}</div>'), unsafe_allow_html=True)
    else:
        st.info(L("上传图片 → 实验5 同时预测物体与属性。"))

mit_tbl = res_table(RESULTS_MITSTATES, ["方法", "骨干", "训练", "Seen", "Unseen", "HM", "AUC"], 7)
vaw_tbl = res_table(RESULTS_VAW, ["方法", "骨干", "训练", "test mAP", "其他"], 5)
st.markdown(L(f"""
<div class="section reveal">
  <div class="eyebrow">结果</div>
  <h2 class="serif">数字会说话</h2>
  <p class="body">绿色行为本项目结果，其余为学术界参考。我们用较小的 ViT-B/16 复现 CSP 得到
  <b>AUC 16.25</b>，并超过用更大骨干的 CoOp。最关键的一行：CSP 未见组合 <b>46.3%</b> vs ResNet <b>≈0%</b>。</p>
  <h3 class="serif" style="font-size:1.15rem;margin:26px 0 0;color:#555;">MIT-States · 闭世界</h3>
  {mit_tbl}
  <p class="note">* 实验1 的 35% 为 joint_acc（闭集多分类），与 CZSL 指标不可直接相减。</p>
  <h3 class="serif" style="font-size:1.15rem;margin:34px 0 0;color:#555;">VAW · 属性预测</h3>
  {vaw_tbl}
  <p class="note">结论：微调(60.6%) &gt; 冻结(55.3%)；物体条件化 &gt; 物体联合预测。</p>
</div>
<div class="section reveal" style="text-align:center;color:var(--muted);font-size:0.85rem;">
  物体–状态联合估计 · 多任务学习 × 组合零样本学习 · 基于 CSP (ICLR 2023) 复现与对比研究
</div>"""), unsafe_allow_html=True)

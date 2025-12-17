# =============================================================================
# CSIRO Image2Biomass - Two-Stream DINO (v3→v2 fallback) + Plain/Tiled/Tiled-FiLM
# with Training-only Multimodal Regularization (A: CLIP-style, B: Distill, C: FiLM-consistency)
# =============================================================================
# 关键修复：
# - 使 DataParallel 也能走“带文本的训练前向”。做法：把 forward 统一成一个接口：
#     forward(x_left, x_right, input_ids=None, attention_mask=None, use_text=False)
#   当 use_text=True 时，内部走多模态路径（原 forward_with_text 逻辑）；
#   当 use_text=False 时，走纯图像路径（验证/推理）。
# - 训练环节调用：model(xl, xr, input_ids, attn_mask, use_text=True)
#   验证/OOF环节调用：model(xl, xr)
# 这样既能让 DataParallel 正常 scatter/gather，又不会再触发
# “DataParallel 没有 forward_with_text 属性”的错误。
# =============================================================================

import os
import gc
import random
import logging
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# === NEW: text encoder (HuggingFace)
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Config
# -----------------------------
class CFG:
    # 基础路径（按你的本地路径）
    seed = 42
    data_path = r"/home/aaa/Kaggle-Series-Competition/CSIRO-Image2Biomass-Prediction/csiro-biomass"
    train_csv_path = os.path.join(data_path, "train.csv")
    image_dir = os.path.join(data_path, "train")
    model_dir = r"/home/aaa/Kaggle-Series-Competition/CSIRO-Image2Biomass-Prediction/model"

    # KFold
    n_splits = 5
    stratify_col = "Dry_Total_g"
    stratify_bins = 10

    # 模型/输入
    final_res = 1000  # 仅日志用途；真正输入分辨率取 backbone 的 input_res
    dropout = 0.30
    hidden_ratio = 0.25

    # 变体：plain / tiled / tiled_film
    model_variant = "tiled_film"
    tiled_grid = (2, 2)     # (rows, cols)
    tiled_overlap = 0       # 可选：重叠像素，0 表示不重叠

    # DINO 主干候选（优先 v3，其次 v2）
    dino_candidates = [
        "vit_base_patch14_dinov3",
        "vit_base_patch14_reg4_dinov3",
        "vit_small_patch14_dinov3",
        "vit_base_patch14_reg4_dinov2",
        "vit_base_patch14_dinov2",
        "vit_small_patch14_dinov2",
    ]

    # 训练策略
    epochs = 110
    freeze_epochs = 10
    head_lr = 5e-4
    finetune_lr = 3e-5
    grad_acc = 4

    # 运行参数
    batch_size = 4
    num_workers = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = True

    # 目标列
    TRAIN_TARGET_COLS = ["Dry_Total_g", "GDM_g", "Dry_Green_g"]
    ALL_TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

    # 指标权重
    METRIC_WEIGHTS = {
        "Dry_Green_g": 0.1,
        "Dry_Dead_g": 0.1,
        "Dry_Clover_g": 0.1,
        "GDM_g": 0.2,
        "Dry_Total_g": 0.5,
    }

    # 可视化开关
    plot_history = True
    plot_scatter = True
    save_history_csv = True

    # === NEW: 文本分支/多模态正则配置（仅训练期启用）
    text_model_name = "bert-base-uncased"   # 若中文可改为 "bert-base-chinese"
    text_max_len    = 48
    text_proj_dim   = 256                   # 图文对齐投影维度 d
    use_text_losses = True                  # 训练期打开 A/B/C-stable
    lambda_nce      = 0.2                   # A: InfoNCE 权重
    lambda_dist     = 0.5                   # B: 蒸馏权重
    lambda_film     = 0.1                   # C-stable: FiLM一致性权重
    nce_temperature = 0.07                  # A: 对比温度


def set_seed(seed: int = CFG.seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Logging
# -----------------------------
def setup_logger():
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


LOGGER = setup_logger()


# -----------------------------
# Data
# -----------------------------
_TOKENIZER = None
def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(CFG.text_model_name, use_fast=True)
    return _TOKENIZER


def _build_prompt_row(sr: pd.Series) -> str:
    # 训练集元字段
    date = str(sr.get("Sampling_Date", "unknown"))
    state = str(sr.get("State", "unknown"))
    species = str(sr.get("Species", "unknown")).replace("_", " ")
    ndvi = sr.get("Pre_GSHH_NDVI", "nan")
    hcm  = sr.get("Height_Ave_cm", "nan")
    return f"{date} {state} {species}; NDVI={ndvi}; Height={hcm}cm."


def load_train_df() -> pd.DataFrame:
    df_long = pd.read_csv(CFG.train_csv_path)
    # 生成每张图一条的 meta（去重）
    meta_cols = ["image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"]
    df_meta = df_long[meta_cols].drop_duplicates("image_path").reset_index(drop=True)
    df_meta["prompt"] = df_meta.apply(_build_prompt_row, axis=1)

    # 宽表：五目标
    df_wide = df_long.pivot_table(
        index="image_path",
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None

    # 合并 prompt
    df_wide = df_wide.merge(df_meta[["image_path", "prompt"]], on="image_path", how="left")
    df_wide["prompt"] = df_wide["prompt"].fillna("unknown date state species; NDVI=nan; Height=nan cm.")
    return df_wide


def add_folds(df_wide: pd.DataFrame) -> pd.DataFrame:
    df_wide["stratify_bin"] = pd.qcut(
        df_wide[CFG.stratify_col],
        q=CFG.stratify_bins,
        labels=False,
        duplicates="drop",
    )
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    df_wide["fold"] = -1
    for fold_id, (_, val_idx) in enumerate(skf.split(df_wide, df_wide["stratify_bin"])):
        df_wide.loc[val_idx, "fold"] = fold_id
    return df_wide


def _get_tf(res: int, is_train: bool):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05),
                     rotate=(-15, 15), shear=0, p=0.30),
            A.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.10, p=0.50),
            A.RandomBrightnessContrast(p=0.30),
            A.Resize(res, res, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(res, res, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class DualStreamDataset(Dataset):
    """
    左/右两路输入；返回 (left_tensor, right_tensor, targets_5, input_ids, attention_mask)
    targets_5 顺序与 CFG.ALL_TARGET_COLS 一致：
    [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """
    def __init__(self, df: pd.DataFrame, image_dir: str, transforms, is_train: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_train = is_train

        self.img_paths = self.df["image_path"].values
        self.prompts   = self.df["prompt"].values if "prompt" in self.df.columns else np.array([""]*len(self.df))
        if self.is_train:
            self.targets_5 = self.df[CFG.ALL_TARGET_COLS].values.astype(np.float32)

        self.tokenizer = _get_tokenizer()

    def __len__(self):
        return len(self.df)

    def _apply_tf(self, img):
        out = self.transforms(image=img)
        if out is None or "image" not in out:
            raise RuntimeError("Albumentations returns None/invalid dict.")
        return out["image"]

    def __getitem__(self, idx):
        filename = os.path.basename(self.img_paths[idx])
        path = os.path.join(self.image_dir, filename)

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]

        left_t = self._apply_tf(left)
        right_t = self._apply_tf(right)

        prompt = str(self.prompts[idx])
        toks = self.tokenizer(
            prompt, max_length=CFG.text_max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        input_ids = toks["input_ids"].squeeze(0)
        attn_mask = toks["attention_mask"].squeeze(0)

        if self.is_train:
            tgt = torch.tensor(self.targets_5[idx], dtype=torch.float32)
        else:
            tgt = torch.tensor(self.targets_5[idx], dtype=torch.float32)

        return left_t, right_t, tgt, input_ids, attn_mask


# -----------------------------
# Backbone helpers
# -----------------------------
def _infer_input_res(m) -> int:
    if hasattr(m, 'patch_embed') and hasattr(m.patch_embed, 'img_size'):
        isz = m.patch_embed.img_size
        return int(isz if isinstance(isz, (int, float)) else isz[0])
    if hasattr(m, 'img_size'):
        isz = m.img_size
        return int(isz if isinstance(isz, (int, float)) else isz[0])
    dc = getattr(m, 'default_cfg', {}) or {}
    ins = dc.get('input_size', None)
    if ins:
        if isinstance(ins, (tuple, list)) and len(ins) >= 2:
            return int(ins[1])
        return int(ins if isinstance(ins, (int, float)) else 224)
    name = getattr(m, 'default_cfg', {}).get('architecture', '') or str(type(m))
    return 518 if ('dinov2' in name.lower()) else 224


def build_dino_backbone():
    last_err = None
    for name in CFG.dino_candidates:
        for gp in ["__default__", "token", "avg"]:
            try:
                if gp == "__default__":
                    m = timm.create_model(name, pretrained=True, num_classes=0)
                    if hasattr(m, "set_grad_checkpointing"):
                        m.set_grad_checkpointing(True)
                    gp_str = "default"
                else:
                    m = timm.create_model(name, pretrained=True, num_classes=0, global_pool=gp)
                    gp_str = gp
                feat = m.num_features
                input_res = _infer_input_res(m)
                LOGGER.info(f"✅ 使用 DINO 主干: {name} | global_pool={gp_str} | feat_dim={feat} | input_res={input_res}")
                return m, feat, name, input_res
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(
        f"无法创建任何 DINO 主干，请检查 timm 版本/模型名。最后错误: {last_err}"
    )


# -----------------------------
# Model Variants
# -----------------------------
class TwoStreamDINOBase(nn.Module):
    """
    公共基类：构建 backbone、三头（green/clover/dead）与正值约束（softplus）
    前向接口：返回 (total, gdm, green) —— 外部损失/指标复用
    """
    def __init__(self, dropout: float = 0.3, hidden_ratio: float = 0.25):
        super().__init__()
        self.backbone, feat, used_name, input_res = build_dino_backbone()
        self.used_backbone_name = used_name
        self.input_res = int(input_res)
        self.feat_dim = feat
        self.combined = feat * 2

        hidden = max(8, int(self.combined * hidden_ratio))

        def head():
            return nn.Sequential(
                nn.Linear(self.combined, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1)
            )

        self.head_green = head()
        self.head_clover = head()
        self.head_dead = head()
        self.softplus = nn.Softplus(beta=1.0)

    def _merge_heads(self, f_l: torch.Tensor, f_r: torch.Tensor):
        f = torch.cat([f_l, f_r], dim=1)      # (B, 2F)
        green_pos = self.softplus(self.head_green(f))
        clover_pos = self.softplus(self.head_clover(f))
        dead_pos = self.softplus(self.head_dead(f))
        gdm = green_pos + clover_pos
        total = gdm + dead_pos
        return total, gdm, green_pos


class TwoStreamDINOPlain(TwoStreamDINOBase):
    """整图编码"""
    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        f_l = self.backbone(x_left)
        f_r = self.backbone(x_right)
        return self._merge_heads(f_l, f_r)


# --- Tiled helpers ---
def _make_edges(L: int, parts: int):
    """把 [0, L) 均分为 parts 份，返回边界 [(s, e), ...]"""
    step = L // parts
    edges = []
    start = 0
    for _ in range(parts - 1):
        edges.append((start, start + step))
        start += step
    edges.append((start, L))
    return edges


class TwoStreamDINOTiled(TwoStreamDINOBase):
    """切块编码后取 mean 池化"""
    def __init__(self, grid=(2, 2), overlap=0, **kwargs):
        super().__init__(**kwargs)
        self.grid = tuple(grid)
        self.overlap = int(overlap)

    def _encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)

        feats = []
        for (rs, re) in rows:
            for (cs, ce) in cols:
                xt = x[:, :, rs:re, cs:ce]
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, size=(self.input_res, self.input_res),
                                       mode="bilinear", align_corners=False)
                ft = self.backbone(xt)       # (B, F)
                feats.append(ft)
        feats = torch.stack(feats, dim=0).permute(1, 0, 2)  # (B, T, F)
        feat_stream = feats.mean(dim=1)
        return feat_stream

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        f_l = self._encode_tiles(x_left)
        f_r = self._encode_tiles(x_right)
        return self._merge_heads(f_l, f_r)


class FiLM(nn.Module):
    """轻量 FiLM：全局上下文 → 每 tile 特征的 γ/β"""
    def __init__(self, in_dim: int):
        super().__init__()
        hid = max(32, in_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, in_dim * 2)  # 输出 concat [gamma, beta]
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gb = self.mlp(context)               # (B, 2F)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return gamma, beta


class TwoStreamDINOTiledFiLM(TwoStreamDINOBase):
    """切块 + FiLM 调制，每路用全局上下文调制各 tile 特征，再聚合"""
    def __init__(self, grid=(2, 2), overlap=0, **kwargs):
        super().__init__(**kwargs)
        self.grid = tuple(grid)
        self.overlap = int(overlap)
        self.film_left = FiLM(self.feat_dim)
        self.film_right = FiLM(self.feat_dim)

    def _tiles_backbone(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)

        feats = []
        for (rs, re) in rows:
            for (cs, ce) in cols:
                xt = x[:, :, rs:re, cs:ce]
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, size=(self.input_res, self.input_res),
                                       mode="bilinear", align_corners=False)
                ft = self.backbone(xt)       # (B, F)
                feats.append(ft)
        feats = torch.stack(feats, dim=0).permute(1, 0, 2)  # (B, T, F)
        return feats

    def _encode_stream(self, x: torch.Tensor, film: FiLM) -> torch.Tensor:
        tiles = self._tiles_backbone(x)      # (B, T, F)
        context = tiles.mean(dim=1)          # (B, F)
        gamma, beta = film(context)          # (B, F), (B, F)
        tiles = tiles * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        feat_stream = tiles.mean(dim=1)      # (B, F)
        return feat_stream

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        f_l = self._encode_stream(x_left, self.film_left)
        f_r = self._encode_stream(x_right, self.film_right)
        return self._merge_heads(f_l, f_r)


# -----------------------------
# Multimodal (训练期) 包装 & 损失
# -----------------------------
class TextEncoder(nn.Module):
    """
    冻结 + 常驻CPU的 BERT 编码器：
    - 关键：用 object.__setattr__ 把 HF BERT 放到 self._bert_cpu，这样它不是 nn.Module 的子模块，
      不会被 net.to(device)/DataParallel 搬到 GPU。
    - 只把投影层 proj 放在和图像分支相同的设备上。
    """
    def __init__(self, name: str, out_dim: int):
        super().__init__()
        bert = AutoModel.from_pretrained(name)
        for p in bert.parameters():
            p.requires_grad = False
        bert.eval()
        bert.to("cpu")
        # 不注册为子模块，避免被 .to()/DataParallel 迁移
        object.__setattr__(self, "_bert_cpu", bert)

        # 只有这一层是可训练/会随 net.to(device) 迁移
        self.proj = nn.Linear(bert.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        # 确保 tokens 在 CPU（和 _bert_cpu 同设备）
        input_ids = input_ids.to("cpu", non_blocking=True)
        attention_mask = attention_mask.to("cpu", non_blocking=True)
        with torch.no_grad():
            out = self._bert_cpu(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]

        # 投影到图像分支所在的设备
        return self.proj(cls.to(self.proj.weight.device, non_blocking=True))
    

class TextFiLM(nn.Module):
    def __init__(self, d_in: int, feat_dim: int):
        super().__init__()
        hid = max(64, feat_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(d_in, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, 2 * feat_dim)
        )
    def forward(self, gtext):                   # (B, d) -> (B, F),(B, F)
        gb = self.net(gtext)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return gamma, beta


def _run_heads_from_features(model: nn.Module, f_l: torch.Tensor, f_r: torch.Tensor):
    f = torch.cat([f_l, f_r], dim=1)
    green = model.softplus(model.head_green(f))
    clover = model.softplus(model.head_clover(f))
    dead = model.softplus(model.head_dead(f))
    gdm = green + clover
    total = gdm + dead
    return total, gdm, green


class MultiModalStudentTeacher(nn.Module):
    """
    包装器：student = 现有图像模型（含 image-FiLM）；teacher = 共享参数，但用 text-FiLM 调制。
    训练期使用 forward(..., use_text=True) 产生三类正则；验证/推理使用 forward(..., use_text=False)。
    """
    def __init__(self, base_student: nn.Module, proj_dim: int, text_model_name: str):
        super().__init__()
        self.student = base_student
        feat_dim = base_student.feat_dim
        self.img_proj = nn.Linear(2 * feat_dim, proj_dim)  # g_img = W [fL; fR]
        self.txt_enc  = TextEncoder(text_model_name, proj_dim)
        self.txt_film_left  = TextFiLM(proj_dim, feat_dim)
        self.txt_film_right = TextFiLM(proj_dim, feat_dim)

        # 透传若干属性，便于外部日志/加载/冻结
        self.used_backbone_name = base_student.used_backbone_name
        self.input_res = base_student.input_res
        self.feat_dim = base_student.feat_dim
        self.backbone = base_student.backbone  # for freezing access

    # === 统一前向（支持 DataParallel）：use_text=True 时走多模态分支
    def forward(self, x_left, x_right, input_ids=None, attention_mask=None, use_text=False):
        if use_text:
            assert input_ids is not None and attention_mask is not None, "use_text=True 需要文本 tokens"
            return self._forward_with_text(x_left, x_right, input_ids, attention_mask)
        else:
            # 验证/推理：只走学生的 forward（纯图像）
            return self.student(x_left, x_right)

    # === 原 forward_with_text 逻辑（私有）
    def _forward_with_text(self, x_left, x_right, input_ids, attention_mask):
        # --- student path (image-FiLM)
        featL_s, featR_s, (gL_img, bL_img, gR_img, bR_img) = self._student_feats(x_left, x_right)
        total_s, gdm_s, green_s = _run_heads_from_features(self.student, featL_s, featR_s)

        # --- teacher path (text-FiLM)
        gtext = self.txt_enc(input_ids, attention_mask)               # (B, d)
        tilesL = self.student._tiles_backbone(x_left)                 # 复用 backbone
        tilesR = self.student._tiles_backbone(x_right)

        gL_txt, bL_txt = self.txt_film_left(gtext)                    # (B,F),(B,F)
        gR_txt, bR_txt = self.txt_film_right(gtext)

        featL_t = (tilesL * (1 + gL_txt.unsqueeze(1)) + bL_txt.unsqueeze(1)).mean(dim=1)
        featR_t = (tilesR * (1 + gR_txt.unsqueeze(1)) + bR_txt.unsqueeze(1)).mean(dim=1)
        total_t, gdm_t, green_t = _run_heads_from_features(self.student, featL_t, featR_t)

        # --- representations for A (InfoNCE)
        g_img  = self.img_proj(torch.cat([featL_s, featR_s], dim=1))   # (B, d)
        g_txt  = gtext                                                 # (B, d)

        out = {
            "student": (total_s, gdm_s, green_s),
            "teacher": (total_t, gdm_t, green_t),
            "img_proj": g_img, "txt_proj": g_txt,
            "film_img": (gL_img, bL_img, gR_img, bR_img),
            "film_txt": (gL_txt, bL_txt, gR_txt, bR_txt),
            "feats":    (featL_s, featR_s, featL_t, featR_t)
        }
        return out

    @torch.no_grad()
    def _student_feats(self, x_left, x_right):
        # 复用学生的 tiles & image-FiLM 调制来拿特征（tiled_film 变体）
        assert hasattr(self.student, "_tiles_backbone"), "Student must be tiled variant for this path."
        # left
        tilesL = self.student._tiles_backbone(x_left)        # (B,T,F)
        ctxL   = tilesL.mean(dim=1)                          # (B,F)
        gL, bL = self.student.film_left(ctxL)                # (B,F)
        featL  = (tilesL * (1 + gL.unsqueeze(1)) + bL.unsqueeze(1)).mean(dim=1)  # (B,F)
        # right
        tilesR = self.student._tiles_backbone(x_right)
        ctxR   = tilesR.mean(dim=1)
        gR, bR = self.student.film_right(ctxR)
        featR  = (tilesR * (1 + gR.unsqueeze(1)) + bR.unsqueeze(1)).mean(dim=1)
        return featL, featR, (gL, bL, gR, bR)


# --- multimodal losses ---
def info_nce_loss(g_img: torch.Tensor, g_txt: torch.Tensor, temperature: float = 0.07):
    g_img = F.normalize(g_img, dim=1)
    g_txt = F.normalize(g_txt, dim=1)
    logits = g_img @ g_txt.t() / temperature          # (B, B)
    labels = torch.arange(g_img.size(0), device=g_img.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


def _pack5(total, gdm, green):
    clover = gdm - green
    dead   = total - gdm
    return torch.cat([green, dead, clover, gdm, total], dim=1)


def distill_loss(student_tuple, teacher_tuple):
    total_s, gdm_s, green_s = student_tuple
    total_t, gdm_t, green_t = teacher_tuple
    yS = _pack5(total_s, gdm_s, green_s)
    with torch.no_grad():
        yT = _pack5(total_t, gdm_t, green_t)
    return F.mse_loss(yS, yT)


def film_consistency_loss(film_img, film_txt):
    gL_i, bL_i, gR_i, bR_i = film_img
    gL_t, bL_t, gR_t, bR_t = film_txt
    with torch.no_grad():
        gL_t = gL_t.detach(); bL_t = bL_t.detach()
        gR_t = gR_t.detach(); bR_t = bR_t.detach()
    return (F.mse_loss(gL_i, gL_t) + F.mse_loss(bL_i, bL_t)
          + F.mse_loss(gR_i, gR_t) + F.mse_loss(bR_i, bR_t)) * 0.25


# -----------------------------
# Build (wrapper)
# -----------------------------
def build_model():
    variant = str(CFG.model_variant).lower().strip()
    if variant == "plain":
        base = TwoStreamDINOPlain(dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
        variant_name = "plain"; grid = None
    elif variant == "tiled":
        base = TwoStreamDINOTiled(grid=CFG.tiled_grid, overlap=CFG.tiled_overlap,
                                  dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
        variant_name = "tiled"; grid = CFG.tiled_grid
    elif variant == "tiled_film":
        base = TwoStreamDINOTiledFiLM(grid=CFG.tiled_grid, overlap=CFG.tiled_overlap,
                                      dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
        variant_name = "tiled_film"; grid = CFG.tiled_grid
    else:
        raise ValueError(f"Unknown model_variant: {CFG.model_variant}")
    # 包装成“训练期多模态 + 推理期纯图像”的一体模型
    net = MultiModalStudentTeacher(base, proj_dim=CFG.text_proj_dim, text_model_name=CFG.text_model_name)
    return net, variant_name, grid


# -----------------------------
# Loss (线性空间)
# -----------------------------
class LinearPhysLoss(nn.Module):
    def __init__(self, weights: Dict[str, float], lam_cons1: float = 0.2, lam_cons2: float = 0.2):
        super().__init__()
        self.w = weights
        self.l1 = lam_cons1
        self.l2 = lam_cons2
        self.mse = nn.MSELoss()

    def forward(self, preds_tuple, targets_5: torch.Tensor):
        total, gdm, green = preds_tuple
        clover = gdm - green
        dead = total - gdm

        g_true = targets_5[:, 0:1]
        d_true = targets_5[:, 1:2]
        c_true = targets_5[:, 2:3]
        gdm_t  = targets_5[:, 3:4]
        tot_t  = targets_5[:, 4:5]

        loss_tgt = (
            self.w["Dry_Green_g"]  * self.mse(green,  g_true) +
            self.w["Dry_Dead_g"]   * self.mse(dead,   d_true)  +
            self.w["Dry_Clover_g"] * self.mse(clover, c_true)  +
            self.w["GDM_g"]        * self.mse(gdm,    gdm_t)   +
            self.w["Dry_Total_g"]  * self.mse(total,  tot_t)
        )

        cons1 = self.mse(green + clover, gdm)
        cons2 = self.mse(green + clover + dead, total)

        return loss_tgt + self.l1 * cons1 + self.l2 * cons2


# -----------------------------
# Metrics & Visualization
# -----------------------------
def _per_target_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]):
    per_mae, per_rmse = {}, {}
    for i, n in enumerate(names):
        diff = y_pred[:, i] - y_true[:, i]
        per_mae[n] = float(np.mean(np.abs(diff)))
        per_rmse[n] = float(np.sqrt(np.mean(diff**2)))
    return per_mae, per_rmse


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    names = CFG.ALL_TARGET_COLS
    w = CFG.METRIC_WEIGHTS
    per = {}
    total = 0.0
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0
        per[name] = float(r2)
        total += r2 * w.get(name, 0.0)
    return float(total), per


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def plot_curves(history_df: pd.DataFrame, save_prefix: str):
    if "train_loss" in history_df and "val_loss" in history_df:
        plt.figure()
        plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend()
        plt.grid(True); plt.tight_layout()
        plt.savefig(f"{save_prefix}_loss_curve.png", dpi=150)
        plt.close()

    if "val_wr2" in history_df:
        plt.figure()
        plt.plot(history_df["epoch"], history_df["val_wr2"], label="Val Weighted R2")
        plt.xlabel("Epoch"); plt.ylabel("Weighted R2"); plt.title("Weighted R2 Curve")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(f"{save_prefix}_r2_curve.png", dpi=150)
        plt.close()


def plot_per_target_r2_bar(per_r2: Dict[str, float], save_prefix: str):
    names = list(per_r2.keys())
    vals = [per_r2[n] for n in names]
    plt.figure()
    plt.bar(names, vals)
    plt.xticks(rotation=30)
    plt.ylabel("R2"); plt.title("Per-target R2")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_per_target_r2_bar.png", dpi=150)
    plt.close()


def plot_scatter_per_target(y_true: np.ndarray, y_pred: np.ndarray, save_prefix: str):
    names = CFG.ALL_TARGET_COLS
    for i, n in enumerate(names):
        plt.figure()
        plt.scatter(y_true[:, i], y_pred[:, i], s=6, alpha=0.5)
        lim_min = min(y_true[:, i].min(), y_pred[:, i].min())
        lim_max = max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0
        plt.title(f"{n} | R2={r2:.3f}")
        plt.xlabel("Ground Truth"); plt.ylabel("Prediction")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_scatter_{n}.png", dpi=150)
        plt.close()


# -----------------------------
# Checkpoint utils (DP 安全)
# -----------------------------
def save_model_state(model: nn.Module, path: str):
    """保存时若为 DataParallel，保存 module 的 state_dict，以免 'module.' 前缀污染"""
    if isinstance(model, nn.DataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()
    torch.save(state, path)


def _strip_module_prefix(state_dict: dict) -> dict:
    """若 key 以 'module.' 开头，剥离之"""
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_state_strict(model: nn.Module, path: str, map_location=None):
    state = torch.load(path, map_location=map_location)
    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) == 0 and len(unexpected) == 0:
        return
    raise RuntimeError(
        f"Strict load failed after prefix fix.\nMissing keys: {missing}\nUnexpected keys: {unexpected}"
    )


# -----------------------------
# Train / Valid loop
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch_desc="Training"):
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    amp_dtype = "cuda" if CFG.device.type == "cuda" else "cpu"
    for step, batch in enumerate(tqdm(loader, desc=epoch_desc, leave=False)):
        # 训练期：拿到文本 tokens
        xl, xr, tgt5, input_ids, attn_mask = batch
        xl = xl.to(CFG.device, non_blocking=True)
        xr = xr.to(CFG.device, non_blocking=True)
        # input_ids = input_ids.to(CFG.device, non_blocking=True)
        # attn_mask = attn_mask.to(CFG.device, non_blocking=True)

        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            # 统一 forward（DataParallel 友好）
            out = model(xl, xr, input_ids=input_ids, attention_mask=attn_mask, use_text=True)
            total_s, gdm_s, green_s = out["student"]
            # ✅ 关键：把 tgt5 移到 output 所在显卡（DP 汇总在 output_device）
            tgt5 = tgt5.to(total_s.device, non_blocking=True)
            # 主损失：物理一致 + 多目标
            loss = criterion((total_s, gdm_s, green_s), tgt5)

            if CFG.use_text_losses:
                # A: InfoNCE（图文对齐）
                loss_nce  = info_nce_loss(out["img_proj"], out["txt_proj"], CFG.nce_temperature)
                # B: 蒸馏（学生 mimic 老师）
                loss_dist = distill_loss(out["student"], out["teacher"])
                # C: FiLM 一致性正则
                loss_film = film_consistency_loss(out["film_img"], out["film_txt"])
                loss = loss + CFG.lambda_nce*loss_nce + CFG.lambda_dist*loss_dist + CFG.lambda_film*loss_film

        running += loss.item()
        loss = loss / CFG.grad_acc
        scaler.scale(loss).backward()

        if (step + 1) % CFG.grad_acc == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return running / len(loader)


@torch.no_grad()
def valid_one_epoch(model, loader, criterion, collect_oof: bool = False):
    model.eval()
    running = 0.0
    preds_list = []
    tgts_list = []

    amp_dtype = "cuda" if CFG.device.type == "cuda" else "cpu"
    for batch in tqdm(loader, desc="Validating", leave=False):
        # 验证/OOF：忽略文本（仍然会从 dataset 里拿到 tokens，但这里不用）
        if len(batch) == 5:
            xl, xr, tgt5, _, _ = batch
        else:
            xl, xr, tgt5 = batch

        xl = xl.to(CFG.device, non_blocking=True)
        xr = xr.to(CFG.device, non_blocking=True)

        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            total, gdm, green = model(xl, xr)  # 仅学生路径
            tgt5 = tgt5.to(CFG.device, non_blocking=True)
            loss = criterion((total, gdm, green), tgt5)

        running += loss.item()

        clover = gdm - green
        dead = total - gdm
        pred_5 = torch.cat([green, dead, clover, gdm, total], dim=1)  # (B,5)
        if collect_oof:
            preds_list.append(pred_5.float().cpu().numpy())
            tgts_list.append(tgt5.float().cpu().numpy())

    val_loss = running / len(loader)

    if collect_oof:
        y_pred = np.concatenate(preds_list, axis=0)
        y_true = np.concatenate(tgts_list, axis=0)
        wr2, per_r2 = weighted_r2(y_true, y_pred)
        per_mae, per_rmse = _per_target_mae_rmse(y_true, y_pred, CFG.ALL_TARGET_COLS)
        return val_loss, wr2, per_r2, per_mae, per_rmse, y_true, y_pred
    else:
        return val_loss, None, None, None, None, None, None


# -----------------------------
# Runner
# -----------------------------
def run_training():
    set_seed(CFG.seed)
    _ensure_dir(CFG.model_dir)

    # 配置信息
    LOGGER.info("✅ 配置加载完成（训练-only）")
    LOGGER.info(f"   - 设备 (Device): {CFG.device.type}")
    LOGGER.info(f"   - 图像最终分辨率 (Final Resolution): {CFG.final_res}x{CFG.final_res}（日志，实际按主干分辨率）")
    LOGGER.info(f"   - 批次大小 (Batch Size): {CFG.batch_size}")
    LOGGER.info(f"   - 梯度累积 (Grad Acc): {CFG.grad_acc}")
    LOGGER.info(f"   - 训练策略: Freeze {CFG.freeze_epochs} + Finetune {CFG.epochs - CFG.freeze_epochs}")
    LOGGER.info(f"   - 学习率: Head {CFG.head_lr} | All {CFG.finetune_lr}")
    LOGGER.info(f"   - 混合精度: {'启用' if CFG.mixed_precision else '禁用'}")
    LOGGER.info(f"   - 保存目录: {CFG.model_dir}")
    LOGGER.info(f"   - 模式: {CFG.model_variant} | 网格: {CFG.tiled_grid if CFG.model_variant!='plain' else '-'}")

    # 加载与分折
    LOGGER.info("📥 正在加载并转换数据...")
    df_wide = load_train_df()
    df_wide = add_folds(df_wide)
    counts = df_wide["fold"].value_counts().sort_index()
    LOGGER.info(f"✅ 数据转换成功，共 {len(df_wide)} 个独立样本。")
    LOGGER.info("✅ 已完成分层 K-Fold 数据划分。")
    LOGGER.info(f"   各 Fold 样本数量分布:\n{counts}")

    # OOF 容器
    oof_records = []  # (image_path, fold, [y_true_5], [y_pred_5])
    fold_best_scores = []

    for fold in range(CFG.n_splits):
        LOGGER.info(f"\n========================= FOLD {fold+1}/{CFG.n_splits} =========================")

        tr_idx = df_wide[df_wide["fold"] != fold].index
        va_idx = df_wide[df_wide["fold"] == fold].index
        tr_df = df_wide.iloc[tr_idx].reset_index(drop=True)
        va_df = df_wide.iloc[va_idx].reset_index(drop=True)

        # 构建模型（含变体/网格）
        net, variant_name, grid = build_model()
        backbone_res = int(getattr(net, "input_res", CFG.final_res))
        LOGGER.info(f"🧠 Backbone: {net.used_backbone_name} | 期望分辨率: {backbone_res}x{backbone_res} | 模式: {variant_name}")

        # 数据集/加载器（使用 backbone_res 做预处理）
        train_tf = _get_tf(backbone_res, is_train=True)
        valid_tf = _get_tf(backbone_res, is_train=False)
        train_ds = DualStreamDataset(tr_df, CFG.image_dir, transforms=train_tf, is_train=True)
        valid_ds = DualStreamDataset(va_df, CFG.image_dir, transforms=valid_tf, is_train=True)

        train_loader = DataLoader(
            train_ds, batch_size=CFG.batch_size, shuffle=True,
            num_workers=CFG.num_workers, pin_memory=True, drop_last=True
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=max(2, CFG.batch_size // 2), shuffle=False,
            num_workers=CFG.num_workers, pin_memory=True
        )

        # 模型信息
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        LOGGER.info(f"   - Params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")

        if torch.cuda.device_count() >= 2 and CFG.device.type == "cuda":
            LOGGER.info(f"🧩 使用多GPU训练: {torch.cuda.device_count()} x GPUs (DataParallel)")
            net = nn.DataParallel(net, device_ids=[0,1])
        net.to(CFG.device)

        # 损失
        criterion = LinearPhysLoss(CFG.METRIC_WEIGHTS).to(CFG.device)

        # 历史记录（用于可视化）
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_wr2": [],
        }
        per_target_keys = CFG.ALL_TARGET_COLS
        for k in per_target_keys:
            history[f"val_r2_{k}"] = []

        # ------- Phase 1: 冻结 backbone，仅训练 heads/投影/FiLM 等 -------
        for p in net.module.backbone.parameters() if isinstance(net, nn.DataParallel) else net.backbone.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=CFG.head_lr)
        scaler = torch.amp.GradScaler("cuda" if CFG.device.type == "cuda" else "cpu", enabled=CFG.mixed_precision)

        best_score = -1e9
        best_path = os.path.join(CFG.model_dir, f"{variant_name}_best_model_fold{fold+1}.pth")

        LOGGER.info(f"--- 阶段一: 训练Heads（{CFG.freeze_epochs} epochs） ---")
        for ep in range(1, CFG.freeze_epochs + 1):
            tr_loss = train_one_epoch(net, train_loader, optimizer, criterion, scaler,
                                      epoch_desc=f"Training P1 {ep}/{CFG.freeze_epochs}")
            va_loss, va_wr2, per_r2, per_mae, per_rmse, y_true, y_pred = valid_one_epoch(
                net, valid_loader, criterion, collect_oof=True
            )

            history["epoch"].append(ep)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["val_wr2"].append(va_wr2)
            for k in per_target_keys:
                history[f"val_r2_{k}"].append(per_r2[k])

            LOGGER.info(
                f"  Train Loss: {tr_loss:.4f} | Valid Loss: {va_loss:.4f} | WR²: {va_wr2:.4f} "
                f"| R2(green={per_r2['Dry_Green_g']:.3f}, dead={per_r2['Dry_Dead_g']:.3f}, "
                f"clover={per_r2['Dry_Clover_g']:.3f}, gdm={per_r2['GDM_g']:.3f}, total={per_r2['Dry_Total_g']:.3f})"
            )

            if va_wr2 > best_score:
                best_score = va_wr2
                save_model_state(net, best_path)
                LOGGER.info(f"  ✨ 提升，保存 -> {os.path.basename(best_path)}")

        # ------- Phase 2: 解冻，微调全网 -------
        for p in net.module.backbone.parameters() if isinstance(net, nn.DataParallel) else net.backbone.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(net.parameters(), lr=CFG.finetune_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(CFG.epochs - CFG.freeze_epochs), eta_min=1e-7
        )
        scaler = torch.amp.GradScaler("cuda" if CFG.device.type == "cuda" else "cpu", enabled=CFG.mixed_precision)

        LOGGER.info(f"--- 阶段二: 全量微调（{CFG.epochs - CFG.freeze_epochs} epochs） ---")
        patience = 6
        bad = 0
        for ep in range(CFG.freeze_epochs + 1, CFG.epochs + 1):
            tr_loss = train_one_epoch(net, train_loader, optimizer, criterion, scaler,
                                      epoch_desc=f"Training P2 {ep}/{CFG.epochs}")
            va_loss, va_wr2, per_r2, per_mae, per_rmse, y_true, y_pred = valid_one_epoch(
                net, valid_loader, criterion, collect_oof=True
            )
            scheduler.step()
            cur_lr = optimizer.param_groups[0]["lr"]
            LOGGER.info(
                f"  Train Loss: {tr_loss:.4f} | Valid Loss: {va_loss:.4f} | WR²: {va_wr2:.4f} | LR: {cur_lr:.2e}"
            )

            history["epoch"].append(ep)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["val_wr2"].append(va_wr2)
            for k in per_target_keys:
                history[f"val_r2_{k}"].append(per_r2[k])

            if va_wr2 > best_score:
                best_score = va_wr2
                bad = 0
                save_model_state(net, best_path)
                LOGGER.info(f"  ✨ 提升，保存 -> {os.path.basename(best_path)}")
            else:
                bad += 1
                if bad >= patience:
                    LOGGER.info("  ⏹️ 早停")
                    break

        # ============= OOF 评估/可视化（加载最佳） =============
        if isinstance(net, nn.DataParallel):
            used_backbone_name = net.module.used_backbone_name
            input_res = net.module.input_res
            del net
        else:
            used_backbone_name = net.used_backbone_name
            input_res = net.input_res
            del net

        net_eval, variant_eval, _ = build_model()
        assert variant_eval == variant_name, "变体不一致，无法加载最佳权重。"
        net_eval.to(CFG.device)
        load_state_strict(net_eval, best_path, map_location=CFG.device)
        net_eval.eval()
        LOGGER.info(f"🔁 已加载最佳权重进行 OOF | Backbone: {used_backbone_name} | input_res={input_res}")

        # 收集 fold OOF
        _, _, _, _, _, y_true, y_pred = valid_one_epoch(net_eval, valid_loader, criterion, collect_oof=True)

        # 保存 OOF (fold)
        oof_df = pd.DataFrame(y_true, columns=[f"true_{c}" for c in CFG.ALL_TARGET_COLS])
        for i, c in enumerate(CFG.ALL_TARGET_COLS):
            oof_df[f"pred_{c}"] = y_pred[:, i]
        oof_df["image_path"] = va_df["image_path"].values
        oof_path = os.path.join(CFG.model_dir, f"{variant_name}_oof_fold{fold+1}.csv")
        oof_df.to_csv(oof_path, index=False)
        LOGGER.info(f"📝 OOF 保存: {oof_path}")

        # 可视化：历史曲线、柱状图、散点图
        hist_df = pd.DataFrame(history)
        prefix = os.path.join(CFG.model_dir, f"{variant_name}_fold{fold+1}")
        if CFG.save_history_csv:
            hist_csv = f"{prefix}_history.csv"
            hist_df.to_csv(hist_csv, index=False)
            LOGGER.info(f"🧾 训练历史保存: {hist_csv}")
        if CFG.plot_history:
            plot_curves(hist_df, prefix)
            plot_per_target_r2_bar({k: hist_df[f"val_r2_{k}"].iloc[-1] for k in per_target_keys}, prefix)
        if CFG.plot_scatter:
            plot_scatter_per_target(y_true, y_pred, prefix)

        # 统计 fold 最佳
        fold_best_scores.append(best_score)

        # 汇总记录
        for ip, yt, yp in zip(va_df["image_path"].values, y_true, y_pred):
            oof_records.append((ip, fold, yt.tolist(), yp.tolist()))

        # 清理
        del net_eval, optimizer, scheduler, train_loader, valid_loader, train_ds, valid_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总：整体 OOF
    all_df = []
    for ip, fd, yt, yp in oof_records:
        row = {"image_path": ip, "fold": fd}
        for i, c in enumerate(CFG.ALL_TARGET_COLS):
            row[f"true_{c}"] = yt[i]
            row[f"pred_{c}"] = yp[i]
        all_df.append(row)
    all_df = pd.DataFrame(all_df)
    oof_all_path = os.path.join(CFG.model_dir, f"{CFG.model_variant}_oof_all.csv")
    all_df.to_csv(oof_all_path, index=False)
    LOGGER.info(f"🧾 汇总 OOF 保存: {oof_all_path}")

    # 计算整体 WR2
    y_true_all = all_df[[f"true_{c}" for c in CFG.ALL_TARGET_COLS]].values
    y_pred_all = all_df[[f"pred_{c}" for c in CFG.ALL_TARGET_COLS]].values
    wr2_all, per_all = weighted_r2(y_true_all, y_pred_all)

    LOGGER.info("\n========================= 训练完成 =========================")
    LOGGER.info(f"各 Fold 最佳 WR²：{[f'{s:.4f}' for s in fold_best_scores]}")
    LOGGER.info(f"平均最佳 WR²：{np.mean(fold_best_scores):.4f}")
    LOGGER.info(f"整体 OOF WR²：{wr2_all:.4f}")
    LOGGER.info(f"整体每目标 R²：{ {k: round(v, 4) for k, v in per_all.items()} }")


# -----------------------------
# Main
# -----------------------------
def main():
    run_training()


if __name__ == "__main__":
    main()

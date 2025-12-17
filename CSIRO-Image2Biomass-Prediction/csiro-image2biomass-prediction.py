# =============================================================================
# CSIRO Image2Biomass - Two-Stream DINO (v3→v2 fallback) + Plain/Tiled/Tiled-FiLM
# =============================================================================
# - 三变体可选：plain / tiled / tiled_film（由 CFG.model_variant 控制）
# - DINOv2 固定输入修复：切块后每 tile 统一插值到 self.input_res（例如 518×518）
# - DataParallel 兼容：保存/加载时自动去/加 "module." 前缀，避免 Missing/Unexpected keys
# - 两阶段：冻结 backbone 训练 heads → 全量微调；AMP + DataParallel
# - 指标与可视化：WR2、每目标 R2/MAE/RMSE、OOF CSV、曲线/散点/柱状图
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
    epochs = 50
    freeze_epochs = 4
    head_lr = 5e-4
    finetune_lr = 1e-5
    grad_acc = 4

    # 运行参数
    batch_size = 4
    num_workers = 4
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
def load_train_df() -> pd.DataFrame:
    df_long = pd.read_csv(CFG.train_csv_path)
    df_wide = df_long.pivot_table(
        index="image_path",
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None
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
    左/右两路输入；返回 (left_tensor, right_tensor, targets_5)
    targets_5 顺序与 CFG.ALL_TARGET_COLS 一致：
    [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """
    def __init__(self, df: pd.DataFrame, image_dir: str, transforms, is_train: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_train = is_train

        self.img_paths = self.df["image_path"].values
        if self.is_train:
            self.targets_5 = self.df[CFG.ALL_TARGET_COLS].values.astype(np.float32)

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

        if self.is_train:
            tgt = torch.tensor(self.targets_5[idx], dtype=torch.float32)
        else:
            # 保持接口一致（当前无推理专用逻辑）
            tgt = torch.tensor(self.targets_5[idx], dtype=torch.float32)

        return left_t, right_t, tgt


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
        # 输入已被 dataloader resize 到 input_res
        f_l = self.backbone(x_left)
        f_r = self.backbone(x_right)
        return self._merge_heads(f_l, f_r)


# --- Tiled helpers ---
def _make_edges(L: int, parts: int):
    """把 [0, L) 均分为 parts 份，返回边界 [(s, e), ...]"""
    step = L // parts
    edges = []
    start = 0
    for i in range(parts - 1):
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
        # x: (B, C, H, W) —— 虽已被全图 resize 到 input_res，但切块后尺寸会更小
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)

        feats = []
        for (rs, re) in rows:
            for (cs, ce) in cols:
                xt = x[:, :, rs:re, cs:ce]  # (B, C, h, w)
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, size=(self.input_res, self.input_res),
                                       mode="bilinear", align_corners=False)
                ft = self.backbone(xt)       # (B, F)
                feats.append(ft)
        feats = torch.stack(feats, dim=0).permute(1, 0, 2)  # (B, T, F)
        feat_stream = feats.mean(dim=1)                     # (B, F)
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
        # context: (B, F)
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
        # 返回所有 tile 的特征 (B, T, F)
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)

        feats = []
        for (rs, re) in rows:
            for (cs, ce) in cols:
                xt = x[:, :, rs:re, cs:ce]  # (B, C, h, w)
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
        f_l = self._encode_stream(x_left, self.film_left)   # (B, F)
        f_r = self._encode_stream(x_right, self.film_right) # (B, F)
        return self._merge_heads(f_l, f_r)


def build_model():
    variant = str(CFG.model_variant).lower().strip()
    if variant == "plain":
        net = TwoStreamDINOPlain(dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
        variant_name = "plain"
        grid = None
    elif variant == "tiled":
        net = TwoStreamDINOTiled(grid=CFG.tiled_grid, overlap=CFG.tiled_overlap,
                                 dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
        variant_name = "tiled"
        grid = CFG.tiled_grid
    elif variant == "tiled_film":
        net = TwoStreamDINOTiledFiLM(grid=CFG.tiled_grid, overlap=CFG.tiled_overlap,
                                     dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
        variant_name = "tiled_film"
        grid = CFG.tiled_grid
    else:
        raise ValueError(f"Unknown model_variant: {CFG.model_variant}")
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
    # 我们希望严格匹配，但先做前缀校正，再人工检查
    if len(missing) == 0 and len(unexpected) == 0:
        return
    # 若还有错误，抛出详细异常，便于定位
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
    for step, (xl, xr, tgt5) in enumerate(tqdm(loader, desc=epoch_desc, leave=False)):
        xl = xl.to(CFG.device, non_blocking=True)
        xr = xr.to(CFG.device, non_blocking=True)
        tgt5 = tgt5.to(CFG.device, non_blocking=True)

        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            preds = model(xl, xr)
            loss = criterion(preds, tgt5)

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
    for xl, xr, tgt5 in tqdm(loader, desc="Validating", leave=False):
        xl = xl.to(CFG.device, non_blocking=True)
        xr = xr.to(CFG.device, non_blocking=True)
        tgt5 = tgt5.to(CFG.device, non_blocking=True)

        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            total, gdm, green = model(xl, xr)
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
            net = nn.DataParallel(net)
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

        # ------- Phase 1: 冻结 backbone，仅训练 heads -------
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
        # 重新构建同构模型以确保 state_dict 结构一致，然后严格加载
        # （注意：此时不要 DataParallel 包裹，以免再次引入 'module.' 前缀）
        if isinstance(net, nn.DataParallel):
            # 从 wrapper 中取出用于日志/尺寸信息
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

        # 汇总记录（可选进一步用）
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

# =============================================================================
# CSIRO Image2Biomass - v4 CrossPVT T2T Mamba (Two-Stream DINOv2, Pure Image)
# -----------------------------------------------------------------------------
# - 左右 half 两路输入；DINOv2 作为 tile encoder，small grid (4x4) + big grid (2x2)
# - T2T soft re-tokenization 将 4x4 → 2x2，并引入局部 attention
# - CrossViT 式小/大分支 cross-attention 融合
# - PVT 金字塔：MobileViT stage → SRA + local Mamba → global Mamba + MHSA
# - 轻量左右交互（cross-gating / cross-attention），三路 MLP 头 + Softplus
# - 物理一致加权 MSE + Pairwise 排序/差分 + 可选 Stage2 辅助头
# - SwanLab 全量集成：配置、指标、图像/特征可视化、断点恢复
# - 所有输出/日志/ckpt 都写入本目录（experiments/v4_crosspvt_t2t_mamba）
# =============================================================================

import argparse
import gc
import json
import logging
import math
import os
import random
import subprocess
import time
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

# SwanLab（训练记录）
try:
    import swanlab
except ImportError:
    swanlab = None


EXPERIMENT_ROOT = Path(__file__).resolve().parent


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    # 数据与路径
    seed: int = 42
    data_path: str = "/home/aaa/Kaggle-Series-Competition/CSIRO-Image2Biomass-Prediction/csiro-biomass"
    train_csv_path: str = ""
    image_dir: str = ""
    experiment_dir: str = str(EXPERIMENT_ROOT)

    # KFold
    n_splits: int = 5
    stratify_col: str = "Dry_Total_g"
    stratify_bins: int = 10

    # 训练
    epochs: int = 60
    freeze_epochs: int = 5
    batch_size: int = 4
    num_workers: int = 4
    grad_accum: int = 3
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.05
    mixed_precision: bool = True

    # 模型
    dropout: float = 0.1
    hidden_ratio: float = 0.35
    dino_candidates: Tuple[str, ...] = field(
        default_factory=lambda: (
            "vit_base_patch14_dinov2",
            "vit_base_patch14_reg4_dinov2",
            "vit_small_patch14_dinov2",
        )
    )
    small_grid: Tuple[int, int] = (4, 4)
    big_grid: Tuple[int, int] = (2, 2)
    t2t_depth: int = 2
    cross_layers: int = 2
    cross_heads: int = 6
    pyramid_dims: Tuple[int, int, int] = (384, 512, 640)
    mobilevit_heads: int = 4
    mobilevit_depth: int = 2
    sra_heads: int = 8
    sra_ratio: int = 2
    mamba_depth: int = 3
    mamba_kernel: int = 5
    aux_head: bool = True
    aux_loss_weight: float = 0.4

    # 目标
    TRAIN_TARGET_COLS: Tuple[str, ...] = (
        "Dry_Total_g",
        "GDM_g",
        "Dry_Green_g",
    )
    ALL_TARGET_COLS: Tuple[str, ...] = (
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    )
    METRIC_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "Dry_Green_g": 0.1,
            "Dry_Dead_g": 0.1,
            "Dry_Clover_g": 0.1,
            "GDM_g": 0.2,
            "Dry_Total_g": 0.5,
        }
    )

    # Pairwise
    use_pairwise: bool = True
    lambda_pair_rank: float = 0.2
    lambda_pair_diff: float = 0.1
    pair_margin: float = 5.0

    # Batch-level aug
    p_cutmix: float = 0.0
    p_fmix: float = 0.0
    p_fda: float = 0.0
    fmix_alpha: float = 1.0
    fmix_decay: float = 3.0
    fda_beta: float = 0.02

    # SwanLab / logging
    project: str = "csiro-img2biomass"
    experiment_name: str = "v4_crosspvt_t2t_mamba"
    comment: str = ""
    log_image_every: int = 1
    log_image_limit: int = 2
    save_history_csv: bool = True

    def __post_init__(self):
        if not self.train_csv_path:
            self.train_csv_path = os.path.join(self.data_path, "train.csv")
        if not self.image_dir:
            self.image_dir = os.path.join(self.data_path, "train")
        self.experiment_dir = str(Path(self.experiment_dir))


CFG = CFG()


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EXPERIMENT_ROOT.parent, timeout=5)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def setup_logger(log_path: Path):
    logger = logging.getLogger("train_v4")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


_ensure_dir(Path(CFG.experiment_dir))
LOGGER = setup_logger(Path(CFG.experiment_dir) / "train.log")


# -----------------------------
# Data
# -----------------------------
def _get_tf(res: int, is_train: bool):
    if is_train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.0, 0.05),
                    rotate=(-18, 18),
                    shear=0,
                    p=0.35,
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.45),
                A.RandomBrightnessContrast(p=0.25),
                A.Resize(res, res, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(res, res, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def load_train_df() -> "pd.DataFrame":
    import pandas as pd  # 延迟导入，避免在未安装时阻塞其他流程

    df_long = pd.read_csv(CFG.train_csv_path)
    meta_cols = [
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]
    df_meta = df_long[meta_cols].drop_duplicates("image_path").reset_index(drop=True)

    df_wide = (
        df_long.pivot_table(index="image_path", columns="target_name", values="target", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    df_wide = df_wide.merge(df_meta, on="image_path", how="left")
    return df_wide


def add_folds(df_wide: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    df_wide = df_wide.copy()
    df_wide["stratify_bin"] = pd.qcut(
        df_wide[CFG.stratify_col], q=CFG.stratify_bins, labels=False, duplicates="drop"
    )
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    df_wide["fold"] = -1
    for fold_id, (_, val_idx) in enumerate(skf.split(df_wide, df_wide["stratify_bin"])):
        df_wide.loc[val_idx, "fold"] = fold_id
    return df_wide.drop(columns=["stratify_bin"])


class DualStreamDataset(Dataset):
    """
    返回 left/right tensor + targets_5
    targets_5 顺序: [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """

    def __init__(self, df, image_dir: str, transforms, is_train: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_train = is_train
        self.img_paths = self.df["image_path"].values
        self.targets = (
            self.df[list(CFG.ALL_TARGET_COLS)].values.astype(np.float32) if "Dry_Total_g" in self.df.columns else None
        )

    def __len__(self):
        return len(self.df)

    def _apply_tf(self, img):
        out = self.transforms(image=img)
        if out is None or "image" not in out:
            raise RuntimeError("Albumentations returned invalid output.")
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
        if self.targets is not None:
            tgt = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            tgt = torch.zeros(5, dtype=torch.float32)
        return left_t, right_t, tgt


# -----------------------------
# Augmentations (tensor-level)
# -----------------------------
def _rand_bbox(H, W, lam):
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def apply_cutmix(xl, xr, y, p=0.5):
    if random.random() > p:
        return xl, xr, y
    B, C, H, W = xl.size()
    perm = torch.randperm(B, device=xl.device)
    lam = np.random.beta(1.0, 1.0)
    x1, y1, x2, y2 = _rand_bbox(H, W, lam)
    xl[:, :, y1:y2, x1:x2] = xl[perm, :, y1:y2, x1:x2]
    xr[:, :, y1:y2, x1:x2] = xr[perm, :, y1:y2, x1:x2]
    lam_area = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
    y = lam_area * y + (1 - lam_area) * y[perm]
    return xl, xr, y


def _sample_fmix_mask(H, W, alpha=1.0, decay=3.0, device="cpu", dtype=torch.float32):
    freqs_y = torch.fft.fftfreq(H, d=1.0).to(device)
    freqs_x = torch.fft.fftfreq(W, d=1.0).to(device)
    fy, fx = torch.meshgrid(freqs_y, freqs_x, indexing="ij")
    spectrum_decay = (fy**2 + fx**2).pow(-decay / 2.0)
    spectrum_decay[0, 0] = 0
    phase = torch.rand(H, W, device=device) * 2 * math.pi
    real = spectrum_decay * torch.cos(phase)
    imag = spectrum_decay * torch.sin(phase)
    four = torch.complex(real, imag)
    field = torch.fft.ifft2(four).real
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    thresh = torch.distributions.Beta(alpha, alpha).sample().to(device)
    mask = (field > thresh).to(dtype)
    return mask


def apply_fmix(xl, xr, y, p=0.5, alpha=1.0, decay=3.0):
    if random.random() > p:
        return xl, xr, y
    B, C, H, W = xl.size()
    perm = torch.randperm(B, device=xl.device)
    mask = _sample_fmix_mask(H, W, alpha=alpha, decay=decay, device=xl.device, dtype=xl.dtype)
    mask = mask.unsqueeze(0).unsqueeze(0)
    xl = xl * mask + xl[perm] * (1 - mask)
    xr = xr * mask + xr[perm] * (1 - mask)
    lam = mask.mean()
    y = lam * y + (1 - lam) * y[perm]
    return xl, xr, y


def _low_freq_amp(img):
    fft = torch.fft.fft2(img, dim=(-2, -1))
    amp, ph = torch.abs(fft), torch.angle(fft)
    return amp, ph


def apply_fda(xl, xr, p=0.5, beta=0.02):
    if random.random() > p:
        return xl, xr
    B, C, H, W = xl.size()
    perm = torch.randperm(B, device=xl.device)

    def fda_once(x, x_perm):
        amp, ph = _low_freq_amp(x)
        amp2, _ = _low_freq_amp(x_perm)
        b = int(min(H, W) * beta)
        cy, cx = H // 2, W // 2
        amp[:, :, cy - b : cy + b, cx - b : cx + b] = amp2[:, :, cy - b : cy + b, cx - b : cx + b]
        fft = amp * torch.exp(1j * ph)
        x_back = torch.fft.ifft2(fft, dim=(-2, -1)).real
        return x_back

    xl2 = fda_once(xl, xl[perm])
    xr2 = fda_once(xr, xr[perm])
    return xl2, xr2


# -----------------------------
# Metrics / loss helpers
# -----------------------------
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


def _per_target_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]):
    per_mae, per_rmse = {}, {}
    for i, n in enumerate(names):
        diff = y_pred[:, i] - y_true[:, i]
        per_mae[n] = float(np.mean(np.abs(diff)))
        per_rmse[n] = float(np.sqrt(np.mean(diff**2)))
    return per_mae, per_rmse


class PhysicalLoss(nn.Module):
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
        gdm_t = targets_5[:, 3:4]
        tot_t = targets_5[:, 4:5]

        loss_tgt = (
            self.w["Dry_Green_g"] * self.mse(green, g_true)
            + self.w["Dry_Dead_g"] * self.mse(dead, d_true)
            + self.w["Dry_Clover_g"] * self.mse(clover, c_true)
            + self.w["GDM_g"] * self.mse(gdm, gdm_t)
            + self.w["Dry_Total_g"] * self.mse(total, tot_t)
        )

        cons1 = self.mse(green + clover, gdm)
        cons2 = self.mse(green + clover + dead, total)
        return loss_tgt + self.l1 * cons1 + self.l2 * cons2


def pairwise_diff_losses(score_head: nn.Module, feat_concat: torch.Tensor, targets_5: torch.Tensor, margin=5.0):
    B = feat_concat.size(0)
    if B < 2:
        z = feat_concat.new_tensor(0.0)
        return z, z

    idx_i = torch.randint(0, B, (B // 2,), device=feat_concat.device)
    idx_j = torch.randint(0, B, (B // 2,), device=feat_concat.device)

    fi, fj = feat_concat[idx_i], feat_concat[idx_j]
    yi = targets_5[idx_i][:, 4]
    yj = targets_5[idx_j][:, 4]
    dy = (yi - yj).detach()

    si = score_head(fi).squeeze(-1)
    sj = score_head(fj).squeeze(-1)

    y = torch.sign(dy).clamp(min=-1, max=1)
    y[y == 0] = 1

    rank_loss = F.margin_ranking_loss(si, sj, y, margin=margin)
    k = 0.1
    diff_loss = F.mse_loss((si - sj), k * dy)
    return rank_loss, diff_loss


# -----------------------------
# Model blocks
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    轻量 MobileViT：局部 CNN + 小型 Transformer（token 化和 fold back）
    """

    def __init__(self, dim, heads=4, depth=2, patch=(2, 2), dropout=0.0):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        self.patch = patch
        self.transformer = nn.ModuleList(
            [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)]
        )
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        local_feat = self.local(x)
        B, C, H, W = local_feat.shape
        ph, pw = self.patch
        new_h = math.ceil(H / ph) * ph
        new_w = math.ceil(W / pw) * pw
        if new_h != H or new_w != W:
            local_feat = F.interpolate(local_feat, size=(new_h, new_w), mode="bilinear", align_corners=False)
            H, W = new_h, new_w
        tokens = local_feat.unfold(2, ph, ph).unfold(3, pw, pw)  # B,C,nh,nw,ph,pw
        tokens = tokens.contiguous().view(B, C, -1, ph, pw)
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        for blk in self.transformer:
            tokens = blk(tokens)
        feat = tokens.view(B, -1, ph * pw, C).permute(0, 3, 1, 2)
        nh = H // ph
        nw = W // pw
        feat = feat.view(B, C, nh, nw, ph, pw).permute(0, 1, 2, 4, 3, 5)
        feat = feat.reshape(B, C, H, W)
        if feat.shape[-2:] != x.shape[-2:]:
            feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = self.fuse(torch.cat([x, feat], dim=1))
        return out


class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, heads=8, sr_ratio=2, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, hw: Tuple[int, int]):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        if self.sr is not None:
            H, W = hw
            feat = x.transpose(1, 2).reshape(B, C, H, W)
            feat = self.sr(feat)
            feat = feat.reshape(B, C, -1).transpose(1, 2)
            feat = self.norm(feat)
        else:
            feat = x
        kv = self.kv(feat)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 3, 1)
        v = v.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


class PVTBlock(nn.Module):
    def __init__(self, dim, heads=8, sr_ratio=2, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.sra = SpatialReductionAttention(dim, heads=heads, sr_ratio=sr_ratio, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x, hw: Tuple[int, int]):
        x = x + self.sra(self.norm1(x), hw)
        x = x + self.ff(self.norm2(x))
        return x


class LocalMambaBlock(nn.Module):
    """
    简化版 local Mamba：DW-Conv + gating + 线性映射
    """

    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = (x * g).transpose(1, 2)  # B, C, N
        x = self.dwconv(x).transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class T2TRetokenizer(nn.Module):
    """
    将 4x4 tile token 做局部 attention + 下采样到 2x2
    """

    def __init__(self, dim, depth=2, heads=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)])

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]):
        B, T, C = tokens.shape
        H, W = grid_hw
        feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)
        seq = feat_map.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            seq = blk(seq)
        seq_map = seq.transpose(1, 2).reshape(B, C, H, W)
        pooled = F.adaptive_avg_pool2d(seq_map, (2, 2))
        retokens = pooled.flatten(2).transpose(1, 2)
        return retokens, seq_map


class CrossScaleFusion(nn.Module):
    def __init__(self, dim, heads=6, dropout=0.0, layers=2):
        super().__init__()
        self.layers_s = nn.ModuleList([AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(layers)])
        self.layers_b = nn.ModuleList([AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(layers)])
        self.cross_s = nn.ModuleList(
            [
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim)
                for _ in range(layers)
            ]
        )
        self.cross_b = nn.ModuleList(
            [
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim)
                for _ in range(layers)
            ]
        )
        self.norm_s = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(self, tok_s: torch.Tensor, tok_b: torch.Tensor):
        # tok_s / tok_b 不含 cls；在这里附加 cls
        B, Ts, C = tok_s.shape
        Tb = tok_b.shape[1]
        cls_s = tok_s.new_zeros(B, 1, C)
        cls_b = tok_b.new_zeros(B, 1, C)
        tok_s = torch.cat([cls_s, tok_s], dim=1)
        tok_b = torch.cat([cls_b, tok_b], dim=1)

        for ls, lb, cs, cb in zip(self.layers_s, self.layers_b, self.cross_s, self.cross_b):
            tok_s = ls(tok_s)
            tok_b = lb(tok_b)
            q_s = self.norm_s(tok_s[:, :1])
            q_b = self.norm_b(tok_b[:, :1])
            cls_s_upd, _ = cs(q_s, torch.cat([tok_b, q_b], dim=1), torch.cat([tok_b, q_b], dim=1), need_weights=False)
            cls_b_upd, _ = cb(q_b, torch.cat([tok_s, q_s], dim=1), torch.cat([tok_s, q_s], dim=1), need_weights=False)
            tok_s = torch.cat([tok_s[:, :1] + cls_s_upd, tok_s[:, 1:]], dim=1)
            tok_b = torch.cat([tok_b[:, :1] + cls_b_upd, tok_b[:, 1:]], dim=1)

        tokens = torch.cat([tok_s[:, :1], tok_b[:, :1], tok_s[:, 1:], tok_b[:, 1:]], dim=1)
        return tokens  # shape ~ (B, 2 + Ts + Tb, C)


class TileEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, input_res: int):
        super().__init__()
        self.backbone = backbone
        self.input_res = input_res

    def forward(self, x: torch.Tensor, grid: Tuple[int, int]):
        B, C, H, W = x.shape
        r, c = grid
        hs = torch.linspace(0, H, steps=r + 1, device=x.device).round().long()
        ws = torch.linspace(0, W, steps=c + 1, device=x.device).round().long()
        tiles = []
        for i in range(r):
            for j in range(c):
                rs, re = hs[i].item(), hs[i + 1].item()
                cs, ce = ws[j].item(), ws[j + 1].item()
                xt = x[:, :, rs:re, cs:ce]
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, size=(self.input_res, self.input_res), mode="bilinear", align_corners=False)
                tiles.append(xt)
        tiles = torch.stack(tiles, dim=1)  # (B, T, C, H, W)
        flat = tiles.view(-1, C, self.input_res, self.input_res)
        feats = self.backbone(flat)
        feats = feats.view(B, -1, feats.shape[-1])
        return feats


class PyramidMixer(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dims: Tuple[int, int, int],
        mobilevit_heads: int = 4,
        mobilevit_depth: int = 2,
        sra_heads: int = 6,
        sra_ratio: int = 2,
        mamba_depth: int = 3,
        mamba_kernel: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3 = dims
        self.proj1 = nn.Linear(dim_in, c1)
        self.mobilevit = MobileViTBlock(c1, heads=mobilevit_heads, depth=mobilevit_depth, dropout=dropout)
        self.proj2 = nn.Linear(c1, c2)
        self.pvt = PVTBlock(c2, heads=sra_heads, sr_ratio=sra_ratio, dropout=dropout, mlp_ratio=3.0)
        self.mamba_local = LocalMambaBlock(c2, kernel_size=mamba_kernel, dropout=dropout)
        self.proj3 = nn.Linear(c2, c3)
        self.mamba_global = nn.ModuleList([LocalMambaBlock(c3, kernel_size=mamba_kernel, dropout=dropout) for _ in range(mamba_depth)])
        self.final_attn = AttentionBlock(c3, heads=min(8, c3 // 64 + 1), dropout=dropout, mlp_ratio=2.0)

    def _tokens_to_map(self, tokens: torch.Tensor, target_hw: Tuple[int, int]):
        B, N, C = tokens.shape
        H, W = target_hw
        need = H * W
        if N < need:
            pad = tokens.new_zeros(B, need - N, C)
            tokens = torch.cat([tokens, pad], dim=1)
        tokens = tokens[:, : need, :]
        feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)
        return feat_map

    @staticmethod
    def _fit_hw(n_tokens: int) -> Tuple[int, int]:
        """选择一个接近方形、满足 h*w>=n_tokens 的网格。"""
        h = int(math.sqrt(n_tokens))
        w = h
        while h * w < n_tokens:
            w += 1
            if h * w < n_tokens:
                h += 1
        return h, w

    def forward(self, tokens: torch.Tensor):
        # 约 10 tokens -> 3x4 map
        B, N, C = tokens.shape
        map_hw = (3, 4)
        feat_map = self._tokens_to_map(tokens, map_hw)
        t1 = self.proj1(tokens)
        m1 = self._tokens_to_map(t1, map_hw)
        m1 = self.mobilevit(m1)
        t1_out = m1.flatten(2).transpose(1, 2)[:, :N]

        # Stage2: 下采样 token 数量（平均池化）
        t2 = self.proj2(t1_out)
        new_len = max(4, N // 2)
        t2 = t2[:, :new_len] + F.adaptive_avg_pool1d(t2.transpose(1, 2), new_len).transpose(1, 2)
        hw2 = self._fit_hw(t2.size(1))
        if t2.size(1) < hw2[0] * hw2[1]:
            pad = t2.new_zeros(B, hw2[0] * hw2[1] - t2.size(1), t2.size(2))
            t2 = torch.cat([t2, pad], dim=1)
        t2 = self.pvt(t2, hw2)
        t2 = self.mamba_local(t2)

        # Stage3: 全局
        t3 = self.proj3(t2)
        pooled = torch.stack([t3.mean(dim=1), t3.max(dim=1).values], dim=1)  # (B,2,C)
        t3 = pooled
        for blk in self.mamba_global:
            t3 = blk(t3)
        t3 = self.final_attn(t3)
        global_feat = t3.mean(dim=1)
        return global_feat, {"stage1_map": m1.detach(), "stage2_tokens": t2.detach(), "stage3_tokens": t3.detach()}


class CrossPVT_T2T_MambaDINO(nn.Module):
    def __init__(self, dropout: float = 0.1, hidden_ratio: float = 0.35):
        super().__init__()
        self.backbone, self.feat_dim, self.backbone_name, self.input_res = self._build_dino_backbone()
        self.tile_encoder = TileEncoder(self.backbone, self.input_res)
        self.t2t = T2TRetokenizer(self.feat_dim, depth=CFG.t2t_depth, heads=CFG.cross_heads, dropout=dropout)
        self.cross = CrossScaleFusion(
            self.feat_dim, heads=CFG.cross_heads, dropout=dropout, layers=CFG.cross_layers
        )
        self.pyramid = PyramidMixer(
            dim_in=self.feat_dim,
            dims=CFG.pyramid_dims,
            mobilevit_heads=CFG.mobilevit_heads,
            mobilevit_depth=CFG.mobilevit_depth,
            sra_heads=CFG.sra_heads,
            sra_ratio=CFG.sra_ratio,
            mamba_depth=CFG.mamba_depth,
            mamba_kernel=CFG.mamba_kernel,
            dropout=dropout,
        )

        combined = CFG.pyramid_dims[-1] * 2
        self.combined_dim = combined
        hidden = max(32, int(combined * hidden_ratio))

        def head():
            return nn.Sequential(
                nn.Linear(combined, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        self.head_green = head()
        self.head_clover = head()
        self.head_dead = head()
        self.score_head = nn.Sequential(nn.LayerNorm(combined), nn.Linear(combined, 1))
        self.aux_head = (
            nn.Sequential(nn.LayerNorm(CFG.pyramid_dims[1]), nn.Linear(CFG.pyramid_dims[1], 5))
            if CFG.aux_head
            else None
        )
        self.softplus = nn.Softplus(beta=1.0)

        self.cross_gate_left = nn.Linear(CFG.pyramid_dims[-1], CFG.pyramid_dims[-1])
        self.cross_gate_right = nn.Linear(CFG.pyramid_dims[-1], CFG.pyramid_dims[-1])

    def _build_dino_backbone(self):
        last_err = None
        for name in CFG.dino_candidates:
            for gp in ["token", "avg", "__default__"]:
                try:
                    if gp == "__default__":
                        m = timm.create_model(name, pretrained=True, num_classes=0)
                        gp_str = "default"
                    else:
                        m = timm.create_model(name, pretrained=True, num_classes=0, global_pool=gp)
                        gp_str = gp
                    feat = m.num_features
                    input_res = self._infer_input_res(m)
                    LOGGER.info(f"✅ 使用 DINO 主干: {name} | global_pool={gp_str} | feat_dim={feat} | input_res={input_res}")
                    if hasattr(m, "set_grad_checkpointing"):
                        m.set_grad_checkpointing(True)
                    return m, feat, name, int(input_res)
                except Exception as e:
                    last_err = e
                    continue
        raise RuntimeError(f"无法创建任何 DINO 主干。最后错误: {last_err}")

    @staticmethod
    def _infer_input_res(m) -> int:
        if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "img_size"):
            isz = m.patch_embed.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])
        if hasattr(m, "img_size"):
            isz = m.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])
        dc = getattr(m, "default_cfg", {}) or {}
        ins = dc.get("input_size", None)
        if ins:
            if isinstance(ins, (tuple, list)) and len(ins) >= 2:
                return int(ins[1])
            return int(ins if isinstance(ins, (int, float)) else 224)
        return 518

    def _half_forward(self, x_half: torch.Tensor):
        tiles_small = self.tile_encoder(x_half, CFG.small_grid)
        tiles_big = self.tile_encoder(x_half, CFG.big_grid)

        t2, stage1_map = self.t2t(tiles_small, CFG.small_grid)
        fused = self.cross(t2, tiles_big)
        feat, feat_maps = self.pyramid(fused)
        # 返回 stage2 token 供辅助头
        feat_maps["stage1_map"] = stage1_map
        return feat, feat_maps

    def _merge_heads(self, f_l: torch.Tensor, f_r: torch.Tensor):
        g_l = torch.sigmoid(self.cross_gate_left(f_r))
        g_r = torch.sigmoid(self.cross_gate_right(f_l))
        f_l = f_l * g_l
        f_r = f_r * g_r
        f = torch.cat([f_l, f_r], dim=1)
        green_pos = self.softplus(self.head_green(f))
        clover_pos = self.softplus(self.head_clover(f))
        dead_pos = self.softplus(self.head_dead(f))
        gdm = green_pos + clover_pos
        total = gdm + dead_pos
        return total, gdm, green_pos, f

    def _param_device_dtype(self):
        try:
            ref = next(self.parameters())
            return ref.device, ref.dtype
        except StopIteration:
            return torch.device("cpu"), torch.float32

    def _empty_forward_output(self, device=None, dtype=None, return_features: bool = False):
        if device is None or dtype is None:
            device_p, dtype_p = self._param_device_dtype()
            if device is None:
                device = device_p
            if dtype is None:
                dtype = dtype_p

        zero = torch.zeros(0, 1, device=device, dtype=dtype)
        out = {
            "total": zero,
            "gdm": zero,
            "green": zero,
            "score_feat": torch.zeros(0, self.combined_dim, device=device, dtype=dtype),
        }
        if self.aux_head is not None:
            out["aux"] = torch.zeros(0, len(CFG.ALL_TARGET_COLS), device=device, dtype=dtype)
        if return_features:
            out["feature_maps"] = {}
        return out

    def forward(self, *inputs, x_left=None, x_right=None, return_features: bool = False):
        # 兼容多种调用方式（单 tensor、双 tensor、元组，以及 DataParallel 空输入场景）
        if inputs:
            if len(inputs) == 1:
                first = inputs[0]
                if isinstance(first, (tuple, list)):
                    if len(first) >= 1:
                        x_left = first[0]
                    if len(first) >= 2:
                        x_right = first[1]
                else:
                    x_left = first
            else:
                x_left = inputs[0]
                x_right = inputs[1]

        # DataParallel 可能在某些设备上给到空输入，此时直接返回零 batch 输出，避免 TypeError
        if x_left is None:
            return self._empty_forward_output(return_features=return_features)
        if isinstance(x_left, torch.Tensor) and x_left.shape[0] == 0:
            return self._empty_forward_output(return_features=return_features)

        if x_right is None:
            if isinstance(x_left, torch.Tensor):
                if x_left.shape[1] % 2 != 0:
                    raise ValueError("无法从单个张量推断左右分支，请显式提供 x_right。")
                x_left, x_right = torch.chunk(x_left, 2, dim=1)
            else:
                raise ValueError("缺少 x_right 输入。")

        feat_l, feats_l = self._half_forward(x_left)
        feat_r, feats_r = self._half_forward(x_right)
        total, gdm, green, f_concat = self._merge_heads(feat_l, feat_r)
        out = {
            "total": total,
            "gdm": gdm,
            "green": green,
            "score_feat": f_concat,
        }
        if self.aux_head is not None:
            aux_tokens = torch.cat([feats_l["stage2_tokens"], feats_r["stage2_tokens"]], dim=1)
            aux_pred = self.softplus(self.aux_head(aux_tokens.mean(dim=1)))
            out["aux"] = aux_pred  # 顺序与 CFG.ALL_TARGET_COLS 对齐
        if return_features:
            out["feature_maps"] = {
                "stage1_left": feats_l.get("stage1_map"),
                "stage1_right": feats_r.get("stage1_map"),
                "stage3_left": feats_l.get("stage3_tokens"),
                "stage3_right": feats_r.get("stage3_tokens"),
            }
        return out


# -----------------------------
# Checkpoint utils
# -----------------------------
def save_checkpoint(state: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location=None):
    return torch.load(path, map_location=map_location)


def find_checkpoint(fold_dir: Path, resume_mode: str = "auto") -> Optional[Path]:
    """
    查找可用的 checkpoint
    
    Args:
        fold_dir: fold 目录
        resume_mode: "auto" (自动检测), "last" (仅 last.pt), "best_wr2" (仅 best_wr2.pt), 
                    "best_loss" (仅 best_loss.pt), "none" (不续训)
    
    Returns:
        checkpoint 路径，如果未找到则返回 None
    """
    ckpt_dir = fold_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    
    if resume_mode == "none":
        return None
    elif resume_mode == "last":
        ckpt_path = ckpt_dir / "last.pt"
        return ckpt_path if ckpt_path.exists() else None
    elif resume_mode == "best_wr2":
        ckpt_path = ckpt_dir / "best_wr2.pt"
        return ckpt_path if ckpt_path.exists() else None
    elif resume_mode == "best_loss":
        ckpt_path = ckpt_dir / "best_loss.pt"
        return ckpt_path if ckpt_path.exists() else None
    elif resume_mode == "auto":
        # 优先级: last.pt > best_wr2.pt > best_loss.pt
        for ckpt_name in ["last.pt", "best_wr2.pt", "best_loss.pt"]:
            ckpt_path = ckpt_dir / ckpt_name
            if ckpt_path.exists():
                return ckpt_path
        return None
    else:
        LOGGER.warning(f"未知的 resume_mode: {resume_mode}，使用 auto")
        return find_checkpoint(fold_dir, "auto")


# -----------------------------
# Training / Validation
# -----------------------------
def _pack5(total, gdm, green):
    clover = gdm - green
    dead = total - gdm
    return torch.cat([green, dead, clover, gdm, total], dim=1)


def dp_unwrap(m: nn.Module):
    return m.module if isinstance(m, nn.DataParallel) else m


def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch_idx, sw_run=None):
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    amp_dtype = "cuda" if CFG.mixed_precision and torch.cuda.is_available() else "cpu"
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch_idx} Train", dynamic_ncols=True)
    
    for step, batch in pbar:
        xl, xr, tgt5 = batch
        xl = xl.to(CFG.device, non_blocking=True)
        xr = xr.to(CFG.device, non_blocking=True)
        tgt5 = tgt5.to(CFG.device, non_blocking=True)

        if CFG.p_fda > 0:
            xl, xr = apply_fda(xl, xr, p=CFG.p_fda, beta=CFG.fda_beta)
        if CFG.p_fmix > 0:
            xl, xr, tgt5 = apply_fmix(xl, xr, tgt5, p=CFG.p_fmix, alpha=CFG.fmix_alpha, decay=CFG.fmix_decay)
        if CFG.p_cutmix > 0:
            xl, xr, tgt5 = apply_cutmix(xl, xr, tgt5, p=CFG.p_cutmix)

        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            # DataParallel 稳健做法：拼接为单 Tensor 传入，避免 scatter 异常
            x_cat = torch.cat([xl, xr], dim=1)
            out = model(x_cat, return_features=False)
            loss = criterion((out["total"], out["gdm"], out["green"]), tgt5)
            if CFG.aux_head and "aux" in out:
                loss_aux = F.mse_loss(out["aux"], tgt5)
                loss = loss + CFG.aux_loss_weight * loss_aux
            if CFG.use_pairwise:
                # 获取实际的模型（去掉 DataParallel）
                actual_model = dp_unwrap(model)
                rank_loss, diff_loss = pairwise_diff_losses(
                    actual_model.score_head, out["score_feat"], tgt5, margin=CFG.pair_margin
                )
                loss = loss + CFG.lambda_pair_rank * rank_loss + CFG.lambda_pair_diff * diff_loss

        current_loss = loss.item()
        running += current_loss
        
        # SwanLab Step Logging
        if sw_run is not None:
            global_step = (epoch_idx - 1) * len(loader) + step + 1
            swanlab.log({"train/step_loss": current_loss}, step=global_step)
            
        loss = loss / CFG.grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % CFG.grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    return running / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, sw_run=None, epoch_idx=0, log_images=False):
    # 为避免 DataParallel 在验证时出现 batch 不均导致的 gather 错误,
    # 临时将模型切换到单 GPU 模式
    is_dp = isinstance(model, nn.DataParallel)
    if is_dp:
        # 获取主设备(通常是 cuda:0)
        main_device = model.output_device if hasattr(model, 'output_device') else torch.device('cuda:0')
        # 解包 DataParallel,移到主设备
        actual_model = model.module.to(main_device)
        val_model = actual_model
        val_device = main_device
    else:
        val_model = model
        val_device = CFG.device
    
    val_model.eval()
    running = 0.0
    preds_list = []
    tgts_list = []
    amp_dtype = "cuda" if CFG.mixed_precision and torch.cuda.is_available() else "cpu"
    first_batch_imgs = None
    first_feats = None

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch_idx} Val", dynamic_ncols=True)

    for step, batch in pbar:
        xl, xr, tgt5 = batch
        xl = xl.to(val_device, non_blocking=True)
        xr = xr.to(val_device, non_blocking=True)
        tgt5 = tgt5.to(val_device, non_blocking=True)
        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            # 拼接为单 Tensor 传入
            x_cat = torch.cat([xl, xr], dim=1)
            out = val_model(x_cat, return_features=log_images and step == 0)
            loss = criterion((out["total"], out["gdm"], out["green"]), tgt5)
        running += loss.item()
        pred_5 = _pack5(out["total"], out["gdm"], out["green"])
        preds_list.append(pred_5.float().cpu().numpy())
        tgts_list.append(tgt5.float().cpu().numpy())
        if first_batch_imgs is None and log_images and out.get("feature_maps") is not None:
            first_batch_imgs = (xl.detach().cpu(), xr.detach().cpu(), tgt5.detach().cpu(), pred_5.detach().cpu())
            first_feats = out["feature_maps"]
        
        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    val_loss = running / len(loader)
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(tgts_list, axis=0)
    wr2, per_r2 = weighted_r2(y_true, y_pred)
    per_mae, per_rmse = _per_target_mae_rmse(y_true, y_pred, CFG.ALL_TARGET_COLS)

    if log_images and sw_run is not None and first_batch_imgs is not None:
        log_images_to_swanlab(sw_run, first_batch_imgs, first_feats, epoch_idx)
    
    # 验证结束后,如果之前是 DataParallel,需要将模型重新包装回去
    # 注意:这里不需要显式操作,因为外部传入的 model 引用仍指向 DataParallel 包装器
    # 我们只是临时用了 model.module,不影响外部的 model 对象

    return val_loss, wr2, per_r2, per_mae, per_rmse, y_true, y_pred


def log_images_to_swanlab(sw_run, batch_imgs, feat_maps, epoch_idx: int):
    if sw_run is None or swanlab is None:
        return
    xl, xr, tgt, pred = batch_imgs
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _denorm(x):
        return torch.clamp(x * std + mean, 0, 1)

    def _to_chw(img_t: torch.Tensor):
        # 期望输出 (3,H,W)
        if img_t.is_sparse:
            img_t = img_t.to_dense()
        while img_t.dim() > 3:
            img_t = img_t[0]
        if img_t.dim() == 3:
            return img_t
        return None

    limit = min(CFG.log_image_limit, xl.size(0))
    for i in range(limit):
        xli = _to_chw(xl[i])
        xri = _to_chw(xr[i])
        if xli is None or xri is None:
            continue
        left = (_denorm(xli).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        right = (_denorm(xri).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        full = np.concatenate([left, right], axis=1)
        swanlab.log({f"val/full_epoch{epoch_idx}_idx{i}": swanlab.Image(full)})
        gt = tgt[i].numpy()
        pd = pred[i].numpy()
        txt = (
            f"GT[G,Dead,Clover,GDM,Total]={gt.round(2).tolist()} | "
            f"PD={pd.round(2).tolist()}"
        )
        swanlab.log({"val/text_epoch": txt})

    def _feat_to_img(feat: torch.Tensor):
        # feat: (B,C,H,W) or (B, L, C)
        if feat is None:
            return None
        if feat.is_sparse:
            feat = feat.to_dense()
        while feat.dim() > 4:
            feat = feat[0]
        if feat.dim() == 3:
            # (B, tokens, C) -> pseudo map
            feat = feat.permute(0, 2, 1).reshape(feat.size(0), feat.size(2), -1, 1)
        if feat.dim() == 4 and feat.size(1) not in (1, 3):
            # 取第一张后均值
            feat = feat[0:1]
        fmap = feat[0].mean(dim=0, keepdim=True)
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
        fmap = (fmap.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        fmap = cv2.applyColorMap(fmap, cv2.COLORMAP_VIRIDIS)
        return cv2.cvtColor(fmap, cv2.COLOR_BGR2RGB)

    for key, fmap in feat_maps.items():
        img = _feat_to_img(fmap)
        if img is not None:
            swanlab.log({f"features/{key}_epoch{epoch_idx}": swanlab.Image(img)})


def build_loaders(tr_df, va_df, input_res: int):
    train_tf = _get_tf(input_res, is_train=True)
    valid_tf = _get_tf(input_res, is_train=False)
    train_ds = DualStreamDataset(tr_df, CFG.image_dir, transforms=train_tf, is_train=True)
    valid_ds = DualStreamDataset(va_df, CFG.image_dir, transforms=valid_tf, is_train=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=max(2, CFG.batch_size // 2),
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    return train_loader, valid_loader


def run_fold(fold: int, df_wide, sw_project: str, resume_path: Optional[Path] = None, resume_mode: str = "auto"):
    fold_dir = _ensure_dir(Path(CFG.experiment_dir) / f"fold_{fold}")
    ckpt_dir = _ensure_dir(fold_dir / "checkpoints")
    metrics_path = fold_dir / "metrics.csv"
    swanlab_info_path = fold_dir / "swanlab_info.json"

    tr_df = df_wide[df_wide["fold"] != fold].reset_index(drop=True)
    va_df = df_wide[df_wide["fold"] == fold].reset_index(drop=True)
    LOGGER.info(f"[Fold {fold}] 训练集: {len(tr_df)} 样本 | 验证集: {len(va_df)} 样本")

    model = CrossPVT_T2T_MambaDINO(dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
    backbone_res = model.input_res
    LOGGER.info(f"[Fold {fold}] Backbone: {model.backbone_name} | 输入分辨率: {backbone_res}")
    train_loader, valid_loader = build_loaders(tr_df, va_df, backbone_res)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"[Fold {fold}] Params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")

    model = model.to(CFG.device)
    if torch.cuda.device_count() >= 2 and CFG.device.type == "cuda":
        LOGGER.info(f"[Fold {fold}] 使用多 GPU (DataParallel)")
        # DataParallel 直接包装模型，支持多个位置参数
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    criterion = PhysicalLoss(CFG.METRIC_WEIGHTS).to(CFG.device)
    optimizer = None
    scheduler = None
    scaler = torch.amp.GradScaler("cuda" if CFG.device.type == "cuda" else "cpu", enabled=CFG.mixed_precision)

    start_epoch = 1
    best_wr2 = -1e9
    best_loss = 1e9
    stage_loaded = 1
    opt_state = None
    sch_state = None
    scaler_state = None
    history_rows = []
    swanlab_run_id = None
    
    # 自动检测或使用指定的 checkpoint
    if resume_path is None:
        resume_path = find_checkpoint(fold_dir, resume_mode)
        if resume_path:
            LOGGER.info(f"[Fold {fold}] 自动检测到 checkpoint: {resume_path}")
    
    if resume_path and resume_path.exists():
        try:
            LOGGER.info(f"[Fold {fold}] 正在加载 checkpoint: {resume_path}")
            state = load_checkpoint(resume_path, map_location="cpu")
            
            # 加载模型状态
            model_state = state.get("model_state")
            if model_state:
                # 检查 checkpoint 中的 key 是否有 module. 前缀（DataParallel 保存的格式）
                first_key = next(iter(model_state.keys())) if model_state else None
                has_module_prefix = first_key and first_key.startswith("module.")
                is_dp_model = isinstance(model, nn.DataParallel)
                
                # 处理 key 前缀不匹配的情况
                if has_module_prefix and not is_dp_model:
                    # checkpoint 有 module. 前缀，但当前模型没有包装 DataParallel，需要去掉前缀
                    LOGGER.info(f"[Fold {fold}] 检测到 DataParallel 格式的 checkpoint，正在移除 'module.' 前缀")
                    new_model_state = {}
                    for k, v in model_state.items():
                        if k.startswith("module."):
                            new_k = k[7:]  # 移除 "module." 前缀
                            new_model_state[new_k] = v
                        else:
                            new_model_state[k] = v
                    model_state = new_model_state
                elif not has_module_prefix and is_dp_model:
                    # checkpoint 没有 module. 前缀，但当前模型已包装 DataParallel，需要添加前缀
                    LOGGER.info(f"[Fold {fold}] 检测到非 DataParallel 格式的 checkpoint，正在添加 'module.' 前缀")
                    new_model_state = {}
                    for k, v in model_state.items():
                        new_model_state[f"module.{k}"] = v
                    model_state = new_model_state
                
                # 加载权重
                # 如果模型是 DataParallel，需要访问内部的模型
                if is_dp_model:
                    actual_model = model.module
                    
                    if has_module_prefix:
                        # checkpoint 有 module. 前缀，但我们已经访问到了实际模型，需要去掉前缀
                        new_state = {}
                        for k, v in model_state.items():
                            if k.startswith("module."):
                                new_state[k[7:]] = v
                            else:
                                new_state[k] = v
                        actual_model.load_state_dict(new_state, strict=False)
                    else:
                        actual_model.load_state_dict(model_state, strict=False)
                else:
                    model.load_state_dict(model_state, strict=False)
                LOGGER.info(f"[Fold {fold}] ✓ 模型权重已加载")
            
            # 加载训练状态
            opt_state = state.get("optimizer_state")
            sch_state = state.get("scheduler_state")
            scaler_state = state.get("scaler_state")
            start_epoch = state.get("epoch", 0) + 1
            best_wr2 = state.get("best_wr2", best_wr2)
            best_loss = state.get("best_loss", best_loss)
            stage_loaded = state.get("stage", 1)
            swanlab_run_id = state.get("swanlab_run_id")
            
            # 加载历史记录
            if metrics_path.exists():
                try:
                    import pandas as pd
                    hist_df = pd.read_csv(metrics_path)
                    history_rows = hist_df.to_dict("records")
                    LOGGER.info(f"[Fold {fold}] ✓ 已加载历史记录 ({len(history_rows)} epochs)")
                except Exception as e:
                    LOGGER.warning(f"[Fold {fold}] 加载历史记录失败: {e}")
            
            # 检查是否已经训练完成
            saved_epoch = state.get("epoch", 0)
            if saved_epoch >= CFG.epochs:
                LOGGER.info(
                    f"[Fold {fold}] ⚠️  该 fold 已完成训练 (Epoch {saved_epoch}/{CFG.epochs})，跳过训练"
                )
                LOGGER.info(f"[Fold {fold}] 最佳指标: WR2={best_wr2:.4f}, Loss={best_loss:.4f}")
                return  # 直接返回，不进行训练
            
            LOGGER.info(
                f"[Fold {fold}] ✓ 断点恢复成功 | "
                f"Epoch: {saved_epoch} → {start_epoch} | "
                f"Best WR2: {best_wr2:.4f} | Best Loss: {best_loss:.4f} | "
                f"Stage: {stage_loaded}"
            )
        except Exception as e:
            LOGGER.error(f"[Fold {fold}] ✗ 加载 checkpoint 失败: {e}")
            LOGGER.info(f"[Fold {fold}] 将从头开始训练")
            resume_path = None
            start_epoch = 1
    else:
        LOGGER.info(f"[Fold {fold}] 未找到 checkpoint，从头开始训练")

    def set_stage(stage: int, load_opt=None, load_sch=None):
        nonlocal optimizer, scheduler
        # 获取实际的模型（去掉 DataParallel）
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        backbone = actual_model.backbone
        if stage == 1:
            for p in backbone.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CFG.lr_head,
                weight_decay=CFG.weight_decay,
            )
            scheduler = None
        else:
            for p in backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=CFG.lr_backbone,
                weight_decay=CFG.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, CFG.epochs - CFG.freeze_epochs), eta_min=CFG.lr_backbone * 0.1
            )
            if load_sch:
                try:
                    scheduler.load_state_dict(load_sch)
                except Exception as e:
                    LOGGER.warning(f"[Fold {fold}] 加载 scheduler 状态失败，使用新 scheduler。原因: {e}")
        if load_opt:
            try:
                optimizer.load_state_dict(load_opt)
            except Exception as e:
                LOGGER.warning(f"[Fold {fold}] 加载 optimizer 状态失败，使用新优化器。原因: {e}")

    set_stage(stage_loaded, opt_state, sch_state)
    if scaler_state:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception:
            LOGGER.warning(f"[Fold {fold}] 恢复 scaler 失败，使用新 scaler。")

    # SwanLab 初始化（支持接续）
    run = None
    if swanlab is not None:
        tags = []
        if resume_path:
            tags.append("resume")
            if "best" in str(resume_path):
                tags.append("from_best")
        
        # 尝试恢复之前的 run_id
        if swanlab_run_id and swanlab_info_path.exists():
            try:
                with open(swanlab_info_path, "r") as f:
                    swanlab_info = json.load(f)
                    saved_run_id = swanlab_info.get("run_id")
                    if saved_run_id == swanlab_run_id:
                        LOGGER.info(f"[Fold {fold}] 尝试接续 SwanLab run: {swanlab_run_id}")
                        # 注意：SwanLab 可能不支持直接接续，这里先尝试
                        run = swanlab.init(
                            project=sw_project,
                            experiment=f"{CFG.experiment_name}_fold{fold}",
                            config={**asdict(CFG), "git_commit": get_git_commit()},
                            tags=tags,
                        )
                        # 如果接续失败，会创建新 run
                    else:
                        LOGGER.warning(f"[Fold {fold}] SwanLab run_id 不匹配，创建新 run")
                        run = swanlab.init(
                            project=sw_project,
                            experiment=f"{CFG.experiment_name}_fold{fold}",
                            config={**asdict(CFG), "git_commit": get_git_commit()},
                            tags=tags,
                        )
            except Exception as e:
                LOGGER.warning(f"[Fold {fold}] 读取 SwanLab 信息失败: {e}，创建新 run")
                run = swanlab.init(
                    project=sw_project,
                    experiment=f"{CFG.experiment_name}_fold{fold}",
                    config={**asdict(CFG), "git_commit": get_git_commit()},
                    tags=tags,
                )
        else:
            run = swanlab.init(
                project=sw_project,
                experiment=f"{CFG.experiment_name}_fold{fold}",
                config={**asdict(CFG), "git_commit": get_git_commit()},
                tags=tags,
            )
        
        # 保存 run_id 供下次使用
        if run is not None:
            try:
                run_id = getattr(run, "run_id", None) or getattr(run, "id", None)
                if run_id:
                    swanlab_info = {"run_id": run_id, "experiment": f"{CFG.experiment_name}_fold{fold}"}
                    with open(swanlab_info_path, "w") as f:
                        json.dump(swanlab_info, f, indent=2)
                    LOGGER.info(f"[Fold {fold}] SwanLab run_id 已保存: {run_id}")
            except Exception as e:
                LOGGER.warning(f"[Fold {fold}] 保存 SwanLab run_id 失败: {e}")

    def _log_epoch(ep, tr_loss, va_loss, wr2, per_r2, per_mae=None, per_rmse=None, lr=None, global_step=None):
        payload = {
            "train/loss": tr_loss,
            "val/loss": va_loss,
            "val/wr2": wr2,
        }
        if per_r2:
            for k, v in per_r2.items():
                payload[f"val/r2_{k}"] = v
        if per_mae:
            for k, v in per_mae.items():
                payload[f"val/mae_{k}"] = v
        if per_rmse:
            for k, v in per_rmse.items():
                payload[f"val/rmse_{k}"] = v
        if lr is not None:
            payload["train/lr"] = lr
        if run is not None:
            swanlab.log(payload, step=global_step if global_step is not None else ep)

    current_stage = stage_loaded
    max_epoch = CFG.epochs
    LOGGER.info(f"[Fold {fold}] 训练开始，起始 epoch={start_epoch}，stage={current_stage}")
    for ep in range(start_epoch, max_epoch + 1):
        stage = 1 if ep <= CFG.freeze_epochs else 2
        if stage != current_stage:
            LOGGER.info(f"[Fold {fold}] 进入 Stage {stage}（{'解冻' if stage==2 else '冻结'}）")
            set_stage(stage)
            current_stage = stage
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, ep, sw_run=run)
        va_loss, wr2, per_r2, per_mae, per_rmse, y_true, y_pred = validate(
            model, valid_loader, criterion, sw_run=run, epoch_idx=ep, log_images=(ep % CFG.log_image_every == 0)
        )
        if scheduler is not None and stage == 2:
            scheduler.step()
        lr_cur = optimizer.param_groups[0]["lr"] if optimizer is not None else None
        _log_epoch(ep, tr_loss, va_loss, wr2, per_r2, per_mae, per_rmse, lr_cur, global_step=ep * len(train_loader))
        lr_str = f"{lr_cur:.2e}" if lr_cur is not None else "0"
        LOGGER.info(
            f"[Fold {fold}] Epoch {ep}/{max_epoch} | Stage {stage} | "
            f"TrainLoss {tr_loss:.4f} | ValLoss {va_loss:.4f} | WR2 {wr2:.4f} | "
            f"R2_total {per_r2.get('Dry_Total_g', 0):.3f} | R2_gdm {per_r2.get('GDM_g', 0):.3f} | "
            f"LR {lr_str}"
        )
        history_rows.append(
            {
                "epoch": ep,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_wr2": wr2,
                **{f"val_r2_{k}": v for k, v in per_r2.items()},
            }
        )
        # 保存 SwanLab run_id
        current_run_id = None
        if run is not None:
            try:
                current_run_id = getattr(run, "run_id", None) or getattr(run, "id", None)
            except Exception:
                pass
        
        # 获取实际的模型状态（去掉 DataParallel 和包装器）
        if isinstance(model, nn.DataParallel):
            actual_model = model.module
            model_state = actual_model.state_dict()
        else:
            model_state = model.state_dict()
        
        state = {
            "epoch": ep,
            "stage": stage,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else {},
            "scaler_state": scaler.state_dict(),
            "best_wr2": best_wr2,
            "best_loss": best_loss,
            "swanlab_run_id": current_run_id or swanlab_run_id,
            "cfg": asdict(CFG),
        }
        save_checkpoint(state, ckpt_dir / "last.pt")
        if wr2 > best_wr2:
            best_wr2 = wr2
            save_checkpoint(state, ckpt_dir / "best_wr2.pt")
            LOGGER.info(f"[Fold {fold}] ✓ 新的最佳 WR2: {best_wr2:.4f} (Epoch {ep})")
        if va_loss < best_loss:
            best_loss = va_loss
            save_checkpoint(state, ckpt_dir / "best_loss.pt")
            LOGGER.info(f"[Fold {fold}] ✓ 新的最佳 Loss: {best_loss:.4f} (Epoch {ep})")

    # 保存历史记录
    if CFG.save_history_csv:
        import pandas as pd
        pd.DataFrame(history_rows).to_csv(metrics_path, index=False)
        LOGGER.info(f"[Fold {fold}] ✓ 训练历史已保存: {metrics_path}")
    
    # 训练完成总结
    LOGGER.info("=" * 80)
    LOGGER.info(f"[Fold {fold}] 训练完成总结")
    LOGGER.info(f"  总 Epochs: {max_epoch}")
    LOGGER.info(f"  最佳 WR2: {best_wr2:.4f}")
    LOGGER.info(f"  最佳 Loss: {best_loss:.4f}")
    if history_rows:
        final_metrics = history_rows[-1]
        LOGGER.info(f"  最终 Epoch {final_metrics.get('epoch', '?')} 指标:")
        LOGGER.info(f"    Train Loss: {final_metrics.get('train_loss', 'N/A'):.4f}")
        LOGGER.info(f"    Val Loss: {final_metrics.get('val_loss', 'N/A'):.4f}")
        LOGGER.info(f"    Val WR2: {final_metrics.get('val_wr2', 'N/A'):.4f}")
        for k, v in final_metrics.items():
            if k.startswith("val_r2_"):
                LOGGER.info(f"    {k}: {v:.4f}")
    LOGGER.info(f"  Checkpoints:")
    LOGGER.info(f"    last.pt: {ckpt_dir / 'last.pt'}")
    LOGGER.info(f"    best_wr2.pt: {ckpt_dir / 'best_wr2.pt'}")
    LOGGER.info(f"    best_loss.pt: {ckpt_dir / 'best_loss.pt'}")
    LOGGER.info("=" * 80)
    
    if run is not None:
        try:
            run.finish()
            LOGGER.info(f"[Fold {fold}] SwanLab run 已结束")
        except Exception as e:
            LOGGER.warning(f"[Fold {fold}] 结束 SwanLab run 时出错: {e}")


def export_config():
    cfg_path = Path(CFG.experiment_dir) / "config.yaml"
    cfg_dict = asdict(CFG)
    cfg_dict["git_commit"] = get_git_commit()
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    LOGGER.info(f"配置已写入 {cfg_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="CSIRO v4 CrossPVT T2T Mamba")
    parser.add_argument("--fold", type=int, default=-1, help="若指定，仅训练该 fold，否则全量 k-fold")
    parser.add_argument("--resume", type=str, default="", help="断点 ckpt 路径（优先级高于 --resume-mode）")
    parser.add_argument(
        "--resume-mode",
        type=str,
        default="auto",
        choices=["auto", "last", "best_wr2", "best_loss", "none"],
        help="断点续训模式: auto(自动检测), last(仅last.pt), best_wr2(仅best_wr2.pt), best_loss(仅best_loss.pt), none(不续训)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(CFG.seed)
    CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_config()

    LOGGER.info("=== 启动 v4 CrossPVT T2T Mamba 训练 ===")
    LOGGER.info(f"设备: {CFG.device}, 混合精度: {CFG.mixed_precision}")
    LOGGER.info(f"实验目录: {CFG.experiment_dir}")
    df_wide = load_train_df()
    df_wide = add_folds(df_wide)
    LOGGER.info(f"数据样本: {len(df_wide)}")

    folds = [args.fold] if args.fold >= 0 else list(range(CFG.n_splits))
    resume_path = Path(args.resume) if args.resume else None
    resume_mode = args.resume_mode
    
    LOGGER.info(f"训练配置: Folds={folds}, Resume Mode={resume_mode}")
    if resume_path:
        LOGGER.info(f"指定 Resume Path: {resume_path}")
    
    for fold in folds:
        LOGGER.info("=" * 80)
        LOGGER.info(f"开始训练 Fold {fold}/{CFG.n_splits - 1}")
        LOGGER.info("=" * 80)
        
        try:
            run_fold(fold, df_wide, CFG.project, resume_path=resume_path, resume_mode=resume_mode)
            LOGGER.info(f"✓ Fold {fold} 训练完成")
        except KeyboardInterrupt:
            LOGGER.warning(f"Fold {fold} 训练被用户中断")
            raise
        except Exception as e:
            LOGGER.error(f"✗ Fold {fold} 训练失败: {e}", exc_info=True)
            LOGGER.info("继续下一个 fold...")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if fold < folds[-1]:
            LOGGER.info(f"等待 3 秒后继续下一个 fold...")
            time.sleep(3)
    
    LOGGER.info("=" * 80)
    LOGGER.info("所有 Folds 训练完成！")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()

# =============================================================================
# CSIRO Image2Biomass - CNN Tile+FiLM (Two-Stream with Feature-wise Linear Modulation)
# -----------------------------------------------------------------------------
# - 左右 half 两路输入；CNN 主干（ConvNeXt/EfficientNetV2等）作为 tile encoder
# - Tile 切块编码 + FiLM 调制（全局上下文调节每个 tile 的特征）
# - 三路 MLP 头 + Softplus 约束正值输出
# - 物理一致加权 MSE 损失
# - SwanLab 全量集成：配置、指标、图像可视化、断点恢复
# - 所有输出/日志/ckpt 都写入本目录（CNN/Tile+FiLM）
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
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
    epochs: int = 80
    freeze_epochs: int = 20
    batch_size: int = 8
    num_workers: int = 4
    grad_accum: int = 2
    lr_backbone: float = 1e-5
    lr_head: float = 5e-4
    weight_decay: float = 0.01
    mixed_precision: bool = True
    patience: int = 15

    # 模型
    dropout: float = 0.2
    hidden_ratio: float = 0.25
    
    # CNN主干候选（从大到小）
    backbone_candidates: Tuple[str, ...] = field(
        default_factory=lambda: (
            "convnext_base.fb_in22k_ft_in1k",  # ConvNeXt-Base (~89M params)
            "convnext_small.fb_in22k_ft_in1k",  # ConvNeXt-Small (~50M params)
            "efficientnetv2_rw_m.agc_in1k",     # EfficientNetV2-M (~54M params)
            "efficientnetv2_rw_s.ra2_in1k",     # EfficientNetV2-S (~24M params)
            "efficientnet_b4.ra2_in1k",         # EfficientNet-B4 (~19M params)
            "resnet50.a1_in1k",                 # ResNet50 fallback
        )
    )
    
    # Tile + FiLM 配置
    tile_grid: Tuple[int, int] = (2, 2)  # (rows, cols)
    tile_overlap: int = 0  # 重叠像素
    input_size: int = 384  # 每个tile的输入尺寸

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

    # SwanLab / logging
    project: str = "csiro-img2biomass"
    experiment_name: str = "cnn_tile_film"
    comment: str = ""
    log_image_every: int = 5
    log_image_limit: int = 2
    save_history_csv: bool = True

    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __post_init__(self):
        if not self.train_csv_path:
            self.train_csv_path = os.path.join(self.data_path, "train.csv")
        if not self.image_dir:
            self.image_dir = os.path.join(self.data_path, "train")


CFG = CFG()


# -----------------------------
# Logger
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
        
        # 文件日志
        fh = logging.FileHandler(Path(CFG.experiment_dir) / "train.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


LOGGER = setup_logger()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8]
    except Exception:
        return "unknown"


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


def _get_transforms(res: int, is_train: bool):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
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
    targets_5 顺序与 CFG.ALL_TARGET_COLS 一致
    """
    def __init__(self, df: pd.DataFrame, image_dir: str, transforms, is_train: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_train = is_train
        self.img_paths = self.df["image_path"].values
        self.targets_5 = self.df[list(CFG.ALL_TARGET_COLS)].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def _apply_tf(self, img):
        out = self.transforms(image=img)
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
        tgt = torch.tensor(self.targets_5[idx], dtype=torch.float32)
        
        return left_t, right_t, tgt


# -----------------------------
# Model Components
# -----------------------------
def build_backbone():
    """尝试创建CNN主干"""
    last_err = None
    for name in CFG.backbone_candidates:
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0)
            feat_dim = model.num_features
            
            # 推断输入尺寸
            dc = getattr(model, 'default_cfg', {}) or {}
            input_size = dc.get('input_size', (3, 224, 224))
            if isinstance(input_size, (tuple, list)) and len(input_size) >= 2:
                input_res = int(input_size[1])
            else:
                input_res = 224
            
            LOGGER.info(f"✅ 使用 CNN 主干: {name} | feat_dim={feat_dim} | input_res={input_res}")
            return model, feat_dim, name, input_res
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"无法创建任何 CNN 主干: {last_err}")


def _make_edges(L: int, parts: int) -> List[Tuple[int, int]]:
    """把 [0, L) 均分为 parts 份，返回边界 [(s, e), ...]"""
    step = L // parts
    edges = []
    start = 0
    for i in range(parts - 1):
        edges.append((start, start + step))
        start += step
    edges.append((start, L))
    return edges


class FiLM(nn.Module):
    """Feature-wise Linear Modulation：全局上下文 → 每 tile 特征的 γ/β"""
    def __init__(self, in_dim: int):
        super().__init__()
        hid = max(64, in_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, in_dim * 2)  # 输出 concat [gamma, beta]
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # context: (B, F)
        gb = self.mlp(context)  # (B, 2F)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return gamma, beta


class TwoStreamCNNTileFiLM(nn.Module):
    """
    Two-Stream CNN with Tile + FiLM
    - 每路图像切分为 tile_grid 个 tiles
    - 每个 tile 通过 CNN backbone 编码
    - 使用 FiLM 调制各 tile 特征
    - 融合后通过三个回归头预测
    """
    def __init__(self, dropout: float = 0.2, hidden_ratio: float = 0.25):
        super().__init__()
        self.backbone, feat_dim, used_name, input_res = build_backbone()
        self.used_backbone_name = used_name
        self.input_res = input_res
        self.feat_dim = feat_dim
        self.combined_dim = feat_dim * 2
        self.grid = CFG.tile_grid
        
        # FiLM 模块（每路一个）
        self.film_left = FiLM(feat_dim)
        self.film_right = FiLM(feat_dim)
        
        # 融合层
        hidden = max(64, int(self.combined_dim * hidden_ratio))
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        
        # 三个回归头
        def head():
            return nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden // 2, 1)
            )
        
        self.head_green = head()
        self.head_clover = head()
        self.head_dead = head()
        self.softplus = nn.Softplus(beta=1.0)

    def _extract_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取所有 tiles 的特征
        Args:
            x: (B, C, H, W)
        Returns:
            tiles_feat: (B, T, F) where T = grid[0] * grid[1]
        """
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)
        
        feats = []
        for (rs, re) in rows:
            for (cs, ce) in cols:
                xt = x[:, :, rs:re, cs:ce]  # (B, C, h, w)
                # 如果 tile 尺寸与预期不符，插值调整
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, size=(self.input_res, self.input_res),
                                      mode="bilinear", align_corners=False)
                ft = self.backbone(xt)  # (B, F)
                feats.append(ft)
        
        feats = torch.stack(feats, dim=1)  # (B, T, F)
        return feats

    def _encode_stream(self, x: torch.Tensor, film: FiLM) -> torch.Tensor:
        """
        编码单路图像
        Args:
            x: (B, C, H, W)
            film: FiLM module
        Returns:
            feat: (B, F)
        """
        tiles = self._extract_tiles(x)  # (B, T, F)
        context = tiles.mean(dim=1)  # (B, F) 全局上下文
        gamma, beta = film(context)  # (B, F), (B, F)
        
        # FiLM 调制
        tiles = tiles * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        
        # 聚合
        feat = tiles.mean(dim=1)  # (B, F)
        return feat

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        f_l = self._encode_stream(x_left, self.film_left)   # (B, F)
        f_r = self._encode_stream(x_right, self.film_right) # (B, F)
        
        # 融合
        f = torch.cat([f_l, f_r], dim=1)  # (B, 2F)
        f = self.fusion(f)  # (B, hidden)
        
        # 三个回归头（Softplus 保证正值）
        green_pos = self.softplus(self.head_green(f))
        clover_pos = self.softplus(self.head_clover(f))
        dead_pos = self.softplus(self.head_dead(f))
        
        # 计算组合指标
        gdm = green_pos + clover_pos
        total = gdm + dead_pos
        
        return total, gdm, green_pos


def build_model():
    net = TwoStreamCNNTileFiLM(dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
    return net


# -----------------------------
# Loss
# -----------------------------
class LinearPhysLoss(nn.Module):
    """物理一致加权 MSE 损失"""
    def __init__(self, weights: Dict[str, float], lam_cons1: float = 0.2, lam_cons2: float = 0.2):
        super().__init__()
        self.w = weights
        self.l1 = lam_cons1
        self.l2 = lam_cons2
        self.smooth_l1 = nn.SmoothL1Loss()

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
            self.w["Dry_Green_g"] * self.smooth_l1(green, g_true) +
            self.w["Dry_Dead_g"] * self.smooth_l1(dead, d_true) +
            self.w["Dry_Clover_g"] * self.smooth_l1(clover, c_true) +
            self.w["GDM_g"] * self.smooth_l1(gdm, gdm_t) +
            self.w["Dry_Total_g"] * self.smooth_l1(total, tot_t)
        )
        
        # 物理一致性约束
        cons1 = self.smooth_l1(green + clover, gdm)
        cons2 = self.smooth_l1(green + clover + dead, total)
        
        return loss_tgt + self.l1 * cons1 + self.l2 * cons2


# -----------------------------
# Metrics
# -----------------------------
def _per_target_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]):
    per_mae, per_rmse = {}, {}
    for i, n in enumerate(names):
        diff = y_pred[:, i] - y_true[:, i]
        per_mae[n] = float(np.mean(np.abs(diff)))
        per_rmse[n] = float(np.sqrt(np.mean(diff ** 2)))
    return per_mae, per_rmse


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    names = list(CFG.ALL_TARGET_COLS)
    w = CFG.METRIC_WEIGHTS
    per = {}
    total_score = 0.0
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0
        per[name] = float(r2)
        total_score += r2 * w.get(name, 0.0)
    return float(total_score), per


# -----------------------------
# Training & Validation
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, sw_run=None):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    amp_dtype = "cuda" if CFG.device.type == "cuda" else "cpu"
    pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)
    
    for step, (xl, xr, tgt5) in enumerate(pbar):
        xl = xl.to(CFG.device, non_blocking=True)
        xr = xr.to(CFG.device, non_blocking=True)
        tgt5 = tgt5.to(CFG.device, non_blocking=True)
        
        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            preds = model(xl, xr)
            loss = criterion(preds, tgt5)
        
        running_loss += loss.item()
        loss = loss / CFG.grad_accum
        scaler.scale(loss).backward()
        
        if (step + 1) % CFG.grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        pbar.set_postfix({"loss": f"{loss.item() * CFG.grad_accum:.4f}"})
    
    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, sw_run=None, epoch_idx=0, log_images=False):
    model.eval()
    running_loss = 0.0
    preds_list = []
    tgts_list = []
    
    amp_dtype = "cuda" if CFG.device.type == "cuda" else "cpu"
    pbar = tqdm(loader, desc="Validating", leave=False)
    
    for xl, xr, tgt5 in pbar:
        xl = xl.to(CFG.device, non_blocking=True)
        xr = xr.to(CFG.device, non_blocking=True)
        tgt5 = tgt5.to(CFG.device, non_blocking=True)
        
        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            total, gdm, green = model(xl, xr)
            loss = criterion((total, gdm, green), tgt5)
        
        running_loss += loss.item()
        
        clover = gdm - green
        dead = total - gdm
        pred_5 = torch.cat([green, dead, clover, gdm, total], dim=1)
        preds_list.append(pred_5.float().cpu().numpy())
        tgts_list.append(tgt5.float().cpu().numpy())
    
    val_loss = running_loss / len(loader)
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(tgts_list, axis=0)
    
    wr2, per_r2 = weighted_r2(y_true, y_pred)
    per_mae, per_rmse = _per_target_mae_rmse(y_true, y_pred, list(CFG.ALL_TARGET_COLS))
    
    return val_loss, wr2, per_r2, per_mae, per_rmse, y_true, y_pred


# -----------------------------
# Checkpoint Utils
# -----------------------------
def save_checkpoint(state: dict, path: Path):
    torch.save(state, path)


def load_checkpoint(path: Path, model, device):
    state = torch.load(path, map_location=device)
    
    # 处理 DataParallel 前缀
    model_state = state.get("model_state", {})
    if list(model_state.keys())[0].startswith("module."):
        model_state = {k[7:]: v for k, v in model_state.items()}
    
    model.load_state_dict(model_state, strict=False)
    return state


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# -----------------------------
# Run Fold
# -----------------------------
def run_fold(fold: int, df_wide: pd.DataFrame, sw_project: str, resume_path: Path = None, resume_mode: str = "auto"):
    ckpt_dir = Path(CFG.experiment_dir) / "checkpoints" / f"fold{fold}"
    _ensure_dir(ckpt_dir)
    
    metrics_path = Path(CFG.experiment_dir) / f"fold{fold}_metrics.csv"
    swanlab_info_path = ckpt_dir / "swanlab_info.json"
    
    # 数据准备
    tr_idx = df_wide[df_wide["fold"] != fold].index
    va_idx = df_wide[df_wide["fold"] == fold].index
    tr_df = df_wide.iloc[tr_idx].reset_index(drop=True)
    va_df = df_wide.iloc[va_idx].reset_index(drop=True)
    
    LOGGER.info(f"[Fold {fold}] Train: {len(tr_df)}, Valid: {len(va_df)}")
    
    # 构建模型
    model = build_model()
    input_res = model.input_res
    
    # 数据加载器
    train_tf = _get_transforms(input_res, is_train=True)
    valid_tf = _get_transforms(input_res, is_train=False)
    
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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"[Fold {fold}] 模型: {model.used_backbone_name}")
    LOGGER.info(f"[Fold {fold}] Params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")
    LOGGER.info(f"[Fold {fold}] Input size: {input_res}x{input_res}, Grid: {CFG.tile_grid}")
    
    # 多GPU支持
    if torch.cuda.device_count() >= 2 and CFG.device.type == "cuda":
        LOGGER.info(f"[Fold {fold}] 使用多GPU训练: {torch.cuda.device_count()} x GPUs")
        model = nn.DataParallel(model)
    model.to(CFG.device)
    
    # 损失函数
    criterion = LinearPhysLoss(CFG.METRIC_WEIGHTS).to(CFG.device)
    
    # 优化器和调度器
    optimizer = None
    scheduler = None
    scaler = torch.amp.GradScaler("cuda" if CFG.device.type == "cuda" else "cpu", enabled=CFG.mixed_precision)
    
    # 断点恢复
    start_epoch = 1
    best_wr2 = -float('inf')
    best_loss = float('inf')
    stage_loaded = 1
    swanlab_run_id = None
    opt_state, sch_state, scaler_state = None, None, None
    history_rows = []
    
    # 检查checkpoint
    if resume_path is None and resume_mode != "none":
        if resume_mode == "auto":
            for name in ["last.pt", "best_wr2.pt", "best_loss.pt"]:
                if (ckpt_dir / name).exists():
                    resume_path = ckpt_dir / name
                    break
        elif resume_mode in ["last", "best_wr2", "best_loss"]:
            candidate = ckpt_dir / f"{resume_mode}.pt"
            if candidate.exists():
                resume_path = candidate
    
    if resume_path and resume_path.exists():
        LOGGER.info(f"[Fold {fold}] 加载 checkpoint: {resume_path}")
        try:
            state = load_checkpoint(resume_path, model.module if isinstance(model, nn.DataParallel) else model, CFG.device)
            start_epoch = state.get("epoch", 0) + 1
            stage_loaded = state.get("stage", 1)
            best_wr2 = state.get("best_wr2", -float('inf'))
            best_loss = state.get("best_loss", float('inf'))
            opt_state = state.get("optimizer_state")
            sch_state = state.get("scheduler_state")
            scaler_state = state.get("scaler_state")
            swanlab_run_id = state.get("swanlab_run_id")
            
            if start_epoch > CFG.epochs:
                LOGGER.info(f"[Fold {fold}] 训练已完成 (Epoch {start_epoch-1}/{CFG.epochs})，跳过")
                return
            
            LOGGER.info(f"[Fold {fold}] ✓ 断点恢复成功 | Epoch: {start_epoch} | Best WR2: {best_wr2:.4f}")
        except Exception as e:
            LOGGER.error(f"[Fold {fold}] 加载 checkpoint 失败: {e}")
            start_epoch = 1
    
    def set_stage(stage: int, load_opt=None, load_sch=None):
        nonlocal optimizer, scheduler
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
                except Exception:
                    pass
        
        if load_opt:
            try:
                optimizer.load_state_dict(load_opt)
            except Exception:
                pass
    
    set_stage(stage_loaded, opt_state, sch_state)
    if scaler_state:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception:
            pass
    
    # SwanLab 初始化
    run = None
    if swanlab is not None:
        tags = [model.module.used_backbone_name if isinstance(model, nn.DataParallel) else model.used_backbone_name]
        if resume_path:
            tags.append("resume")
        
        run = swanlab.init(
            project=sw_project,
            experiment_name=f"{CFG.experiment_name}_fold{fold}",
            config={**asdict(CFG), "git_commit": get_git_commit()},
        )
        
        if run is not None:
            try:
                run_id = getattr(run, "run_id", None) or getattr(run, "id", None)
                if run_id:
                    with open(swanlab_info_path, "w") as f:
                        json.dump({"run_id": run_id}, f)
                    LOGGER.info(f"[Fold {fold}] SwanLab run_id: {run_id}")
            except Exception:
                pass
    
    def _log_epoch(ep, tr_loss, va_loss, wr2, per_r2, per_mae=None, per_rmse=None, lr=None):
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
        if lr is not None:
            payload["train/lr"] = lr
        if run is not None:
            swanlab.log(payload, step=ep)
    
    # 训练循环
    current_stage = stage_loaded
    epochs_without_improvement = 0
    
    LOGGER.info(f"[Fold {fold}] 开始训练，起始 epoch={start_epoch}, stage={current_stage}")
    
    for ep in range(start_epoch, CFG.epochs + 1):
        stage = 1 if ep <= CFG.freeze_epochs else 2
        if stage != current_stage:
            LOGGER.info(f"[Fold {fold}] 进入 Stage {stage}（{'解冻主干' if stage==2 else '冻结主干'}）")
            set_stage(stage)
            current_stage = stage
        
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, ep, sw_run=run)
        va_loss, wr2, per_r2, per_mae, per_rmse, y_true, y_pred = validate(
            model, valid_loader, criterion, sw_run=run, epoch_idx=ep,
            log_images=(ep % CFG.log_image_every == 0)
        )
        
        if scheduler is not None and stage == 2:
            scheduler.step()
        
        lr_cur = optimizer.param_groups[0]["lr"] if optimizer else None
        _log_epoch(ep, tr_loss, va_loss, wr2, per_r2, per_mae, per_rmse, lr_cur)
        
        lr_str = f"{lr_cur:.2e}" if lr_cur else "N/A"
        LOGGER.info(
            f"[Fold {fold}] Epoch {ep}/{CFG.epochs} | Stage {stage} | "
            f"TrainLoss {tr_loss:.4f} | ValLoss {va_loss:.4f} | WR2 {wr2:.4f} | "
            f"R2_total {per_r2.get('Dry_Total_g', 0):.3f} | R2_gdm {per_r2.get('GDM_g', 0):.3f} | "
            f"LR {lr_str}"
        )
        
        history_rows.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_wr2": wr2,
            **{f"val_r2_{k}": v for k, v in per_r2.items()},
        })
        
        # 获取模型状态
        if isinstance(model, nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        current_run_id = None
        if run is not None:
            try:
                current_run_id = getattr(run, "run_id", None) or getattr(run, "id", None)
            except Exception:
                pass
        
        state = {
            "epoch": ep,
            "stage": stage,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict() if optimizer else {},
            "scheduler_state": scheduler.state_dict() if scheduler else {},
            "scaler_state": scaler.state_dict(),
            "best_wr2": best_wr2,
            "best_loss": best_loss,
            "swanlab_run_id": current_run_id or swanlab_run_id,
            "cfg": asdict(CFG),
        }
        save_checkpoint(state, ckpt_dir / "last.pt")
        
        if wr2 > best_wr2:
            best_wr2 = wr2
            state["best_wr2"] = best_wr2
            save_checkpoint(state, ckpt_dir / "best_wr2.pt")
            LOGGER.info(f"[Fold {fold}] ✓ 新的最佳 WR2: {best_wr2:.4f} (Epoch {ep})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if va_loss < best_loss:
            best_loss = va_loss
            state["best_loss"] = best_loss
            save_checkpoint(state, ckpt_dir / "best_loss.pt")
            LOGGER.info(f"[Fold {fold}] ✓ 新的最佳 Loss: {best_loss:.4f} (Epoch {ep})")
        
        # 早停
        if epochs_without_improvement >= CFG.patience:
            LOGGER.info(f"[Fold {fold}] 早停: {CFG.patience} epochs 无改善")
            break
    
    # 保存历史记录
    if CFG.save_history_csv:
        pd.DataFrame(history_rows).to_csv(metrics_path, index=False)
        LOGGER.info(f"[Fold {fold}] ✓ 训练历史已保存: {metrics_path}")
    
    # 训练完成总结
    LOGGER.info("=" * 80)
    LOGGER.info(f"[Fold {fold}] 训练完成总结")
    LOGGER.info(f"  最佳 WR2: {best_wr2:.4f}")
    LOGGER.info(f"  最佳 Loss: {best_loss:.4f}")
    LOGGER.info(f"  Checkpoints: {ckpt_dir}")
    LOGGER.info("=" * 80)
    
    if run is not None:
        try:
            run.finish()
            LOGGER.info(f"[Fold {fold}] SwanLab run 已结束")
        except Exception as e:
            LOGGER.warning(f"[Fold {fold}] 结束 SwanLab run 时出错: {e}")


def export_config():
    cfg_path = Path(CFG.experiment_dir) / "config.json"
    cfg_dict = asdict(CFG)
    cfg_dict["git_commit"] = get_git_commit()
    cfg_dict["device"] = str(CFG.device)
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    LOGGER.info(f"配置已写入 {cfg_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="CSIRO CNN Tile+FiLM Training")
    parser.add_argument("--fold", type=int, default=-1, help="若指定，仅训练该 fold，否则全量 k-fold")
    parser.add_argument("--resume", type=str, default="", help="断点 ckpt 路径")
    parser.add_argument(
        "--resume-mode",
        type=str,
        default="auto",
        choices=["auto", "last", "best_wr2", "best_loss", "none"],
        help="断点续训模式",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(CFG.seed)
    CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_config()
    
    LOGGER.info("=" * 80)
    LOGGER.info("=== 启动 CNN Tile+FiLM 训练 ===")
    LOGGER.info(f"设备: {CFG.device}, 混合精度: {CFG.mixed_precision}")
    LOGGER.info(f"实验目录: {CFG.experiment_dir}")
    LOGGER.info(f"主干候选: {CFG.backbone_candidates}")
    LOGGER.info(f"Tile Grid: {CFG.tile_grid}, Input Size: {CFG.input_size}")
    LOGGER.info("=" * 80)
    
    df_wide = load_train_df()
    df_wide = add_folds(df_wide)
    LOGGER.info(f"数据样本: {len(df_wide)}")
    
    folds = [args.fold] if args.fold >= 0 else list(range(CFG.n_splits))
    resume_path = Path(args.resume) if args.resume else None
    resume_mode = args.resume_mode
    
    LOGGER.info(f"训练配置: Folds={folds}, Resume Mode={resume_mode}")
    
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
            LOGGER.info("等待 3 秒后继续下一个 fold...")
            time.sleep(3)
    
    LOGGER.info("=" * 80)
    LOGGER.info("所有 Folds 训练完成！")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()


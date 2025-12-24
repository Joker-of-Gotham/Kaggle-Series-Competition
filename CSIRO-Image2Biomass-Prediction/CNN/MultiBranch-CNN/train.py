# =============================================================================
# CSIRO Image2Biomass - Multi-Branch CNN + Token Fusion
# -----------------------------------------------------------------------------
# - Multi-expert CNN backbones with multi-scale pooling
# - Expert tokens + optional metadata token -> transformer fusion
# - Physical constraints via parameterization (Green/Clover/Dead -> GDM/Total)
# - Expert DropPath + aux heads to prevent expert collapse
# - SwanLab full logging (metrics, images, feature maps, attention)
# - Outputs/logs/ckpt saved in this directory (CNN/MultiBranch-CNN)
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
from typing import Any, Dict, List, Optional, Tuple

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
    # Data
    seed: int = 42
    data_path: str = "/home/aaa/Kaggle-Series-Competition/CSIRO-Image2Biomass-Prediction/csiro-biomass"
    train_csv_path: str = ""
    image_dir: str = ""
    experiment_dir: str = str(EXPERIMENT_ROOT)

    # KFold
    n_splits: int = 5
    stratify_col: str = "Dry_Total_g"
    stratify_bins: int = 10

    # Training
    epochs: int = 130
    freeze_epochs: int = 30
    batch_size: int = 4
    num_workers: int = 4
    grad_accum: int = 2
    lr_backbone: float = 4e-5
    lr_head: float = 8e-4
    weight_decay: float = 0.0025
    mixed_precision: bool = True
    patience: int = 20

    # Input
    input_size: int = 384

    # Model
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.2
    hidden_ratio: float = 0.5
    num_scales: int = 3
    use_meta_token: bool = True
    meta_embed_dim: int = 48
    meta_dropout: float = 0.1

    # Expert drop path + aux loss
    expert_drop_path_max: float = 0.35
    expert_drop_warmup: int = 20
    aux_loss_weight: float = 0.2

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    use_ema_eval: bool = True

    # Expert configs (name + candidate backbones + aug profile)
    expert_configs: Tuple[Dict[str, Any], ...] = field(
        default_factory=lambda: (
            {
                "name": "convnextv2",
                "candidates": (
                    "convnextv2_base.fcmae_ft_in1k",
                    "convnextv2_base",
                    "convnext_base.fb_in22k_ft_in1k",
                    "convnext_small.fb_in22k_ft_in1k",
                    "resnet50.a1_in1k",
                ),
                "aug": "mild",
            },
            {
                "name": "internimage",
                "candidates": (
                    "internimage_t_1k_224",
                    "internimage_s_1k_224",
                    "convnext_small.fb_in22k_ft_in1k",
                    "resnet50.a1_in1k",
                ),
                "aug": "geo",
            },
            {
                "name": "unireplknet",
                "candidates": (
                    "unireplknet_base",
                    "unireplknet_tiny",
                    "repvit_m1",
                    "efficientnetv2_rw_s.ra2_in1k",
                ),
                "aug": "blur",
            },
            {
                "name": "repvit",
                "candidates": (
                    "repvit_m1",
                    "repvit_m2",
                    "fasternet_s",
                    "mobilenetv3_large_100",
                ),
                "aug": "strong",
            },
        )
    )

    # Targets
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

    # Logging
    project: str = "csiro-img2biomass"
    experiment_name: str = "cnn_multibranch"
    log_image_every: int = 5
    log_image_limit: int = 2
    log_attn_every: int = 5
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
# Data + Meta
# -----------------------------
META_INFO: Dict[str, Any] = {}


def load_train_df() -> pd.DataFrame:
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
    df_wide = df_long.pivot_table(
        index="image_path",
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.merge(df_meta, on="image_path", how="left")
    return df_wide


def prepare_meta(df_wide: pd.DataFrame) -> pd.DataFrame:
    df = df_wide.copy()
    df["State"] = df["State"].fillna("unknown")
    df["Species"] = df["Species"].fillna("unknown")
    df["Sampling_Date"] = df["Sampling_Date"].fillna("unknown")

    df["ndvi"] = pd.to_numeric(df["Pre_GSHH_NDVI"], errors="coerce")
    df["height"] = pd.to_numeric(df["Height_Ave_cm"], errors="coerce")
    dt = pd.to_datetime(df["Sampling_Date"], errors="coerce")
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day

    for col in ["ndvi", "height", "year", "month", "day"]:
        if df[col].isna().all():
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].median())

    df["month_sin"] = np.sin(2 * math.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * math.pi * df["month"] / 12.0)
    df["day_sin"] = np.sin(2 * math.pi * df["day"] / 31.0)
    df["day_cos"] = np.cos(2 * math.pi * df["day"] / 31.0)

    numeric_cols = ["ndvi", "height", "year"]
    stats = {}
    for col in numeric_cols:
        mean = float(df[col].mean())
        std = float(df[col].std())
        if std < 1e-6:
            std = 1.0
        stats[col] = (mean, std)
        df[f"{col}_norm"] = (df[col] - mean) / std

    states = sorted(df["State"].unique().tolist())
    species = sorted(df["Species"].unique().tolist())
    if "unknown" not in states:
        states = ["unknown"] + states
    if "unknown" not in species:
        species = ["unknown"] + species
    state2idx = {s: i for i, s in enumerate(states)}
    species2idx = {s: i for i, s in enumerate(species)}

    df["state_idx"] = df["State"].map(state2idx).fillna(0).astype(int)
    df["species_idx"] = df["Species"].map(species2idx).fillna(0).astype(int)

    META_INFO.update(
        {
            "meta_num_cols": [f"{c}_norm" for c in numeric_cols] + ["month_sin", "month_cos", "day_sin", "day_cos"],
            "state2idx": state2idx,
            "species2idx": species2idx,
            "num_states": len(state2idx),
            "num_species": len(species2idx),
            "stats": stats,
        }
    )

    return df


def add_folds(df_wide: pd.DataFrame) -> pd.DataFrame:
    df_wide = df_wide.copy()
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
    return df_wide.drop(columns=["stratify_bin"])


def build_aug_profiles(res: int, is_train: bool) -> Dict[str, Any]:
    if not is_train:
        base = A.Compose(
            [
                A.Resize(res, res, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        return {"mild": base, "geo": base, "blur": base, "strong": base}

    return {
        "mild": A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Resize(res, res, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        "geo": A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5),
                A.Perspective(scale=(0.03, 0.08), p=0.4),
                A.RandomResizedCrop(res, res, scale=(0.75, 1.0), ratio=(0.9, 1.1), p=0.6),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.Resize(res, res, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        "blur": A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=10, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                A.MotionBlur(blur_limit=5, p=0.2),
                A.RandomBrightnessContrast(p=0.25),
                A.Resize(res, res, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        "strong": A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.6),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.6),
                A.RandomBrightnessContrast(p=0.4),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.CoarseDropout(max_holes=8, max_height=24, max_width=24, p=0.2),
                A.Resize(res, res, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
    }


class MultiBranchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transforms_list: List[Any], is_train: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms_list = transforms_list
        self.is_train = is_train

        self.img_paths = self.df["image_path"].values
        self.targets = self.df[list(CFG.ALL_TARGET_COLS)].values.astype(np.float32)

        self.meta_num = self.df[META_INFO["meta_num_cols"]].values.astype(np.float32)
        self.state_idx = self.df["state_idx"].values.astype(np.int64)
        self.species_idx = self.df["species_idx"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def _apply_tf(self, img, tf):
        out = tf(image=img)
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

        views = [self._apply_tf(img, tf) for tf in self.transforms_list]
        views = torch.stack(views, dim=0)

        tgt = torch.tensor(self.targets[idx], dtype=torch.float32)
        meta_num = torch.tensor(self.meta_num[idx], dtype=torch.float32)
        state_idx = torch.tensor(self.state_idx[idx], dtype=torch.long)
        species_idx = torch.tensor(self.species_idx[idx], dtype=torch.long)

        return views, meta_num, state_idx, species_idx, tgt


# -----------------------------
# Model Components
# -----------------------------
def _infer_input_res(model) -> int:
    dc = getattr(model, "default_cfg", {}) or {}
    input_size = dc.get("input_size", (3, 224, 224))
    if isinstance(input_size, (tuple, list)) and len(input_size) >= 2:
        return int(input_size[1])
    return 224


def build_backbone(candidates: Tuple[str, ...], num_scales: int):
    last_err = None
    for name in candidates:
        try:
            model = timm.create_model(
                name, pretrained=True, features_only=True, out_indices=(1, 2, 3)
            )
            channels = model.feature_info.channels()
            if len(channels) >= 1:
                input_res = _infer_input_res(model)
                use_features = True
                if len(channels) > num_scales:
                    channels = channels[-num_scales:]
                LOGGER.info(f"✅ Expert backbone (features): {name} | channels={channels}")
                return model, name, use_features, channels, None, input_res
        except Exception as e:
            last_err = e
            try:
                model = timm.create_model(name, pretrained=True, num_classes=0)
                feat_dim = getattr(model, "num_features", None)
                if feat_dim is None:
                    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
                    feat_dim = getattr(classifier, "in_features", None)
                if feat_dim is None:
                    raise RuntimeError("Failed to infer feature dim.")
                input_res = _infer_input_res(model)
                LOGGER.info(f"✅ Expert backbone: {name} | feat_dim={feat_dim}")
                return model, name, False, [feat_dim], feat_dim, input_res
            except Exception as e2:
                last_err = e2
                continue
    raise RuntimeError(f"Unable to create backbone from candidates: {last_err}")


class MultiScaleSummary(nn.Module):
    def __init__(self, in_channels: List[int], out_dim: int, dropout: float):
        super().__init__()
        self.in_channels = in_channels
        self.proj = nn.Sequential(
            nn.Linear(sum(2 * c for c in in_channels), out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        feats = feats[-len(self.in_channels):]
        pooled = []
        for f in feats:
            gap = f.mean(dim=(2, 3))
            gmp = f.amax(dim=(2, 3))
            pooled.append(torch.cat([gap, gmp], dim=1))
        x = torch.cat(pooled, dim=1)
        return self.proj(x)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float, hidden_ratio: float):
        super().__init__()
        hidden = max(64, int(in_dim * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.out = nn.Linear(hidden // 2, 3)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.softplus(self.out(x))


class ExpertBranch(nn.Module):
    def __init__(self, name: str, candidates: Tuple[str, ...], d_model: int, dropout: float):
        super().__init__()
        self.name = name
        self.backbone, used_name, use_features, channels, feat_dim, input_res = build_backbone(
            candidates, num_scales=CFG.num_scales
        )
        self.used_backbone_name = used_name
        self.use_features = use_features
        self.input_res = input_res

        if use_features:
            self.summary = MultiScaleSummary(channels, d_model, dropout)
        else:
            self.summary = nn.Sequential(
                nn.Linear(channels[-1], d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.aux_head = RegressionHead(d_model, dropout=dropout, hidden_ratio=CFG.hidden_ratio)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feature_maps = {}
        if self.use_features:
            feats = self.backbone(x)
            if not isinstance(feats, (list, tuple)):
                feats = [feats]
            feats = feats[-CFG.num_scales:]
            token = self.summary(feats)
            if return_features:
                for i, f in enumerate(feats):
                    feature_maps[f"{self.name}/s{i}"] = f
        else:
            feat = self.backbone(x)
            token = self.summary(feat)
        aux = self.aux_head(token)
        return token, aux, feature_maps


class MetaEncoder(nn.Module):
    def __init__(self, num_states: int, num_species: int, num_numeric: int, d_model: int):
        super().__init__()
        self.state_embed = nn.Embedding(num_states, CFG.meta_embed_dim)
        self.species_embed = nn.Embedding(num_species, CFG.meta_embed_dim)
        in_dim = num_numeric + CFG.meta_embed_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(CFG.meta_dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, meta_num: torch.Tensor, state_idx: torch.Tensor, species_idx: torch.Tensor):
        state_e = self.state_embed(state_idx)
        species_e = self.species_embed(species_idx)
        x = torch.cat([meta_num, state_e, species_e], dim=1)
        return self.mlp(x)


class FusionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm, need_weights=return_attn, average_attn_weights=False
        )
        x = x + self.drop(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class TokenFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FusionLayer(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn)
        return x, attn_maps


class MultiBranchCNN(nn.Module):
    def __init__(self, expert_cfgs: Tuple[Dict[str, Any], ...]):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                ExpertBranch(
                    name=cfg["name"],
                    candidates=tuple(cfg["candidates"]),
                    d_model=CFG.d_model,
                    dropout=CFG.dropout,
                )
                for cfg in expert_cfgs
            ]
        )
        self.num_experts = len(self.experts)
        self.expert_embed = nn.Embedding(self.num_experts, CFG.d_model)
        self.null_tokens = nn.Parameter(torch.zeros(self.num_experts, CFG.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, CFG.d_model))

        self.meta_encoder = None
        if CFG.use_meta_token:
            self.meta_encoder = MetaEncoder(
                META_INFO["num_states"],
                META_INFO["num_species"],
                len(META_INFO["meta_num_cols"]),
                CFG.d_model,
            )

        self.fusion = TokenFusion(
            d_model=CFG.d_model,
            n_heads=CFG.n_heads,
            n_layers=CFG.n_layers,
            mlp_ratio=CFG.mlp_ratio,
            dropout=CFG.dropout,
        )
        self.head = RegressionHead(CFG.d_model, dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
        self.drop_prob = CFG.expert_drop_path_max

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.null_tokens, std=0.02)
        nn.init.trunc_normal_(self.expert_embed.weight, std=0.02)

    def set_drop_prob(self, p: float):
        self.drop_prob = p

    def _apply_expert_drop(self, tokens: torch.Tensor):
        if not self.training or self.drop_prob <= 0:
            return tokens, torch.zeros(1, device=tokens.device)
        bsz, n, _ = tokens.shape
        keep = torch.rand(bsz, n, device=tokens.device) > self.drop_prob
        # ensure at least one expert kept per sample
        for i in range(bsz):
            if keep[i].sum() == 0:
                keep[i, torch.randint(0, n, (1,), device=tokens.device)] = True
        mask = keep.float().unsqueeze(-1)
        null = self.null_tokens.unsqueeze(0).expand(bsz, -1, -1)
        tokens = tokens * mask + null * (1 - mask)
        drop_ratio = (1.0 - keep.float().mean()).view(1)
        return tokens, drop_ratio

    def forward(
        self,
        views: torch.Tensor,
        meta_num: Optional[torch.Tensor] = None,
        state_idx: Optional[torch.Tensor] = None,
        species_idx: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        return_features: bool = False,
    ):
        if isinstance(views, torch.Tensor):
            views_list = [views[:, i] for i in range(self.num_experts)]
        else:
            views_list = views

        tokens = []
        aux_preds = []
        feature_maps = {}
        for i, (expert, x) in enumerate(zip(self.experts, views_list)):
            tok, aux, fmap = expert(x, return_features=return_features)
            tok = tok + self.expert_embed.weight[i].unsqueeze(0)
            tokens.append(tok)
            aux_preds.append(aux)
            if return_features and fmap:
                feature_maps.update(fmap)

        tokens = torch.stack(tokens, dim=1)
        aux_stack = torch.stack(aux_preds, dim=1)
        tokens, drop_ratio = self._apply_expert_drop(tokens)

        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        token_list = [cls]
        if self.meta_encoder is not None and meta_num is not None:
            meta_tok = self.meta_encoder(meta_num, state_idx, species_idx).unsqueeze(1)
            token_list.append(meta_tok)
        token_list.append(tokens)
        x = torch.cat(token_list, dim=1)

        x, attn_maps = self.fusion(x, return_attn=return_attn)
        cls_out = x[:, 0]
        fused_gcd = self.head(cls_out)
        green = fused_gcd[:, 0:1]
        clover = fused_gcd[:, 1:2]
        dead = fused_gcd[:, 2:3]

        out = {
            "green": green,
            "clover": clover,
            "dead": dead,
            "aux": aux_stack,
            "drop_ratio": drop_ratio,
        }
        if return_attn:
            out["attn"] = attn_maps
        if return_features:
            out["feature_maps"] = feature_maps
        return out


def build_model():
    return MultiBranchCNN(CFG.expert_configs)


# -----------------------------
# Loss + Metrics
# -----------------------------
def pack_pred5(green: torch.Tensor, clover: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
    gdm = green + clover
    total = gdm + dead
    return torch.cat([green, dead, clover, gdm, total], dim=1)


class WeightedHuberLoss(nn.Module):
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = torch.tensor(
            [
                weights["Dry_Green_g"],
                weights["Dry_Dead_g"],
                weights["Dry_Clover_g"],
                weights["GDM_g"],
                weights["Dry_Total_g"],
            ],
            dtype=torch.float32,
        )
        self.huber = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred_5: torch.Tensor, target_5: torch.Tensor):
        w = self.weights.to(pred_5.device).view(1, -1)
        loss = self.huber(pred_5, target_5)
        return (loss * w).mean()


class MultiBranchLoss(nn.Module):
    def __init__(self, weights: Dict[str, float], aux_weight: float):
        super().__init__()
        self.base = WeightedHuberLoss(weights)
        self.aux_weight = aux_weight

    def forward(self, out: Dict[str, Any], targets: torch.Tensor):
        fused_pred5 = pack_pred5(out["green"], out["clover"], out["dead"])
        loss_fused = self.base(fused_pred5, targets)
        aux_losses = []
        aux = out.get("aux", None)
        if isinstance(aux, torch.Tensor):
            # aux shape: (B, num_experts, 3)
            for i in range(aux.size(1)):
                aux_pred5 = pack_pred5(aux[:, i, 0:1], aux[:, i, 1:2], aux[:, i, 2:3])
                aux_losses.append(self.base(aux_pred5, targets))
        if aux_losses:
            loss_aux = torch.stack(aux_losses).mean()
        else:
            loss_aux = torch.tensor(0.0, device=targets.device)
        loss = loss_fused + self.aux_weight * loss_aux
        return loss, {"loss_fused": loss_fused, "loss_aux": loss_aux, "aux_losses": aux_losses}


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
# EMA
# -----------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register(model)

    def register(self, model: nn.Module):
        self.shadow = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        self.backup = {}
        state = model.state_dict()
        for k in self.shadow:
            self.backup[k] = state[k].clone()
            state[k].copy_(self.shadow[k])
        model.load_state_dict(state, strict=False)

    def restore(self, model: nn.Module):
        state = model.state_dict()
        for k in self.backup:
            state[k].copy_(self.backup[k])
        model.load_state_dict(state, strict=False)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, Any]):
        self.decay = state.get("decay", self.decay)
        self.shadow = state.get("shadow", self.shadow)


# -----------------------------
# Training & Validation
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, ema=None, sw_run=None):
    model.train()
    running_loss = 0.0
    aux_running = [0.0 for _ in range(model.module.num_experts if isinstance(model, nn.DataParallel) else model.num_experts)]
    drop_running = 0.0
    drop_count = 0
    optimizer.zero_grad(set_to_none=True)

    amp_dtype = "cuda" if CFG.device.type == "cuda" else "cpu"
    pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)

    for step, (views, meta_num, state_idx, species_idx, tgt5) in enumerate(pbar):
        views = views.to(CFG.device, non_blocking=True)
        meta_num = meta_num.to(CFG.device, non_blocking=True)
        state_idx = state_idx.to(CFG.device, non_blocking=True)
        species_idx = species_idx.to(CFG.device, non_blocking=True)
        tgt5 = tgt5.to(CFG.device, non_blocking=True)

        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            out = model(views, meta_num, state_idx, species_idx)
            loss, loss_dict = criterion(out, tgt5)

        running_loss += loss.item()
        if loss_dict["aux_losses"]:
            for i, aux_l in enumerate(loss_dict["aux_losses"]):
                aux_running[i] += aux_l.item()
        drop_ratio = out.get("drop_ratio")
        if drop_ratio is not None:
            if isinstance(drop_ratio, torch.Tensor):
                drop_running += float(drop_ratio.detach().cpu().float().mean().item())
            else:
                drop_running += float(drop_ratio)
            drop_count += 1

        if sw_run is not None and swanlab is not None:
            global_step = (epoch - 1) * len(loader) + step + 1
            swanlab.log({"train/step_loss": loss.item()}, step=global_step)

        loss = loss / CFG.grad_accum
        scaler.scale(loss).backward()

        if (step + 1) % CFG.grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model.module if isinstance(model, nn.DataParallel) else model)

        pbar.set_postfix({"loss": f"{loss.item() * CFG.grad_accum:.4f}"})

    mean_aux = [v / max(1, len(loader)) for v in aux_running]
    mean_drop = drop_running / max(1, drop_count)
    return running_loss / len(loader), mean_aux, mean_drop


@torch.no_grad()
def validate(model, loader, criterion, sw_run=None, epoch_idx=0, log_images=False, log_attn=False, ema=None):
    is_dp = isinstance(model, nn.DataParallel)
    if is_dp:
        val_model = model.module
        val_device = torch.device("cuda:0") if CFG.device.type == "cuda" else CFG.device
        val_model.to(val_device)
    else:
        val_model = model
        val_device = CFG.device

    if ema is not None and CFG.use_ema_eval:
        ema.apply_to(val_model)

    val_model.eval()
    running_loss = 0.0
    preds_list = []
    tgts_list = []
    first_batch = None
    first_feats = None
    first_attn = None

    amp_dtype = "cuda" if CFG.device.type == "cuda" else "cpu"
    pbar = tqdm(loader, desc="Validating", leave=False)

    for step, (views, meta_num, state_idx, species_idx, tgt5) in enumerate(pbar):
        views = views.to(val_device, non_blocking=True)
        meta_num = meta_num.to(val_device, non_blocking=True)
        state_idx = state_idx.to(val_device, non_blocking=True)
        species_idx = species_idx.to(val_device, non_blocking=True)
        tgt5 = tgt5.to(val_device, non_blocking=True)

        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            out = val_model(
                views,
                meta_num,
                state_idx,
                species_idx,
                return_attn=log_attn and step == 0,
                return_features=log_images and step == 0,
            )
            loss, _ = criterion(out, tgt5)

        running_loss += loss.item()

        pred5 = pack_pred5(out["green"], out["clover"], out["dead"])
        preds_list.append(pred5.float().cpu().numpy())
        tgts_list.append(tgt5.float().cpu().numpy())

        if step == 0 and log_images:
            first_batch = (views.detach().cpu(), tgt5.detach().cpu(), pred5.detach().cpu())
            first_feats = out.get("feature_maps", None)
        if step == 0 and log_attn:
            first_attn = out.get("attn", None)

    val_loss = running_loss / len(loader)
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(tgts_list, axis=0)

    wr2, per_r2 = weighted_r2(y_true, y_pred)
    per_mae, per_rmse = _per_target_mae_rmse(y_true, y_pred, list(CFG.ALL_TARGET_COLS))

    if log_images and sw_run is not None and first_batch is not None:
        token_labels = ["CLS"]
        if CFG.use_meta_token:
            token_labels.append("META")
        token_labels += [f"E{i+1}" for i in range(len(CFG.expert_configs))]
        log_images_to_swanlab(sw_run, first_batch, first_feats, first_attn, token_labels, epoch_idx)

    if ema is not None and CFG.use_ema_eval:
        ema.restore(val_model)

    return val_loss, wr2, per_r2, per_mae, per_rmse, y_true, y_pred


def log_images_to_swanlab(sw_run, batch_data, feat_maps, attn_maps, token_labels, epoch_idx: int):
    if sw_run is None or swanlab is None:
        return
    views, tgt, pred = batch_data
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

    def _denorm(x):
        return torch.clamp(x * std + mean, 0, 1)

    views = _denorm(views).permute(0, 1, 3, 4, 2).numpy()
    limit = min(CFG.log_image_limit, views.shape[0])
    for i in range(limit):
        imgs = [(views[i, j] * 255).astype(np.uint8) for j in range(views.shape[1])]
        full = np.concatenate(imgs, axis=1)
        swanlab.log({f"val/aug_views_epoch{epoch_idx}_idx{i}": swanlab.Image(full)})
        gt = tgt[i].numpy()
        pd = pred[i].numpy()
        txt = f"GT={gt.round(2).tolist()} | PD={pd.round(2).tolist()}"
        swanlab.log({f"val/text_epoch{epoch_idx}_idx{i}": txt})

    def _feat_to_img(feat: torch.Tensor):
        if feat is None:
            return None
        if feat.is_sparse:
            feat = feat.to_dense()
        while feat.dim() > 4:
            feat = feat[0]
        if feat.dim() == 3:
            feat = feat.permute(0, 2, 1).reshape(feat.size(0), feat.size(2), -1, 1)
        if feat.dim() == 4:
            fmap = feat[0].mean(dim=0, keepdim=True)
        else:
            return None
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
        fmap = (fmap.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        fmap = cv2.applyColorMap(fmap, cv2.COLORMAP_VIRIDIS)
        return cv2.cvtColor(fmap, cv2.COLOR_BGR2RGB)

    if feat_maps:
        for key, fmap in feat_maps.items():
            img = _feat_to_img(fmap)
            if img is not None:
                swanlab.log({f"features/{key}_epoch{epoch_idx}": swanlab.Image(img)})

    if attn_maps:
        attn = attn_maps[-1]
        if attn is not None and attn.numel() > 0:
            attn = attn[0].mean(dim=0).cpu().numpy()
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)
            heat = (attn * 255).astype(np.uint8)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            swanlab.log({f"attn/heat_epoch{epoch_idx}": swanlab.Image(heat)})
            swanlab.log({f"attn/tokens_epoch{epoch_idx}": ", ".join(token_labels)})


# -----------------------------
# Checkpoints
# -----------------------------
def save_checkpoint(state: dict, path: Path):
    torch.save(state, path)


def load_checkpoint(path: Path, model, device):
    state = torch.load(path, map_location=device)
    model_state = state.get("model_state", state)
    if model_state:
        first_key = next(iter(model_state.keys()))
        if first_key.startswith("module.") and not isinstance(model, nn.DataParallel):
            model_state = {k[7:]: v for k, v in model_state.items()}
        elif not first_key.startswith("module.") and isinstance(model, nn.DataParallel):
            model_state = {f"module.{k}": v for k, v in model_state.items()}
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

    tr_idx = df_wide[df_wide["fold"] != fold].index
    va_idx = df_wide[df_wide["fold"] == fold].index
    tr_df = df_wide.iloc[tr_idx].reset_index(drop=True)
    va_df = df_wide.iloc[va_idx].reset_index(drop=True)

    LOGGER.info(f"[Fold {fold}] Train: {len(tr_df)}, Valid: {len(va_df)}")

    # Data loaders
    train_profiles = build_aug_profiles(CFG.input_size, is_train=True)
    valid_profiles = build_aug_profiles(CFG.input_size, is_train=False)
    train_tfs = [train_profiles.get(cfg["aug"], train_profiles["mild"]) for cfg in CFG.expert_configs]
    valid_tfs = [valid_profiles.get(cfg["aug"], valid_profiles["mild"]) for cfg in CFG.expert_configs]

    train_ds = MultiBranchDataset(tr_df, CFG.image_dir, transforms_list=train_tfs, is_train=True)
    valid_ds = MultiBranchDataset(va_df, CFG.image_dir, transforms_list=valid_tfs, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=max(2, CFG.batch_size // 2), shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True
    )

    # Model
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    used_names = [ex.used_backbone_name for ex in model.experts]
    LOGGER.info(f"[Fold {fold}] Experts: {used_names}")
    LOGGER.info(f"[Fold {fold}] Params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")
    LOGGER.info(f"[Fold {fold}] Input size: {CFG.input_size}x{CFG.input_size}")

    if torch.cuda.device_count() >= 2 and CFG.device.type == "cuda":
        LOGGER.info(f"[Fold {fold}] Using DataParallel: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(CFG.device)

    # Loss, optimizer, scheduler
    criterion = MultiBranchLoss(CFG.METRIC_WEIGHTS, aux_weight=CFG.aux_loss_weight).to(CFG.device)
    scaler = torch.amp.GradScaler("cuda" if CFG.device.type == "cuda" else "cpu", enabled=CFG.mixed_precision)
    optimizer = None
    scheduler = None

    # EMA
    ema = None
    if CFG.use_ema:
        ema = ModelEMA(model.module if isinstance(model, nn.DataParallel) else model, decay=CFG.ema_decay)

    # Resume
    start_epoch = 1
    best_wr2 = -float("inf")
    best_loss = float("inf")
    stage_loaded = 1
    swanlab_run_id = None
    opt_state, sch_state, scaler_state, ema_state = None, None, None, None
    history_rows = []

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
        LOGGER.info(f"[Fold {fold}] Loading checkpoint: {resume_path}")
        try:
            state = load_checkpoint(resume_path, model, CFG.device)
            start_epoch = state.get("epoch", 0) + 1
            stage_loaded = state.get("stage", 1)
            best_wr2 = state.get("best_wr2", -float("inf"))
            best_loss = state.get("best_loss", float("inf"))
            opt_state = state.get("optimizer_state")
            sch_state = state.get("scheduler_state")
            scaler_state = state.get("scaler_state")
            ema_state = state.get("ema_state")
            swanlab_run_id = state.get("swanlab_run_id")
            if start_epoch > CFG.epochs:
                LOGGER.info(f"[Fold {fold}] Training already finished (Epoch {start_epoch-1}/{CFG.epochs}).")
                return
            LOGGER.info(f"[Fold {fold}] Resume success | Epoch: {start_epoch} | Best WR2: {best_wr2:.4f}")
        except Exception as e:
            LOGGER.error(f"[Fold {fold}] Resume failed: {e}")
            start_epoch = 1

    def set_stage(stage: int, load_opt=None, load_sch=None):
        nonlocal optimizer, scheduler
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        for ex in actual_model.experts:
            for p in ex.backbone.parameters():
                p.requires_grad = stage != 1

        if stage == 1:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CFG.lr_head,
                weight_decay=CFG.weight_decay,
            )
            scheduler = None
        else:
            backbone_params = []
            head_params = []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if "backbone" in n:
                    backbone_params.append(p)
                else:
                    head_params.append(p)
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": CFG.lr_backbone},
                    {"params": head_params, "lr": CFG.lr_head},
                ],
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
    if ema is not None and ema_state:
        try:
            ema.load_state_dict(ema_state)
        except Exception:
            pass

    # SwanLab init
    run = None
    if swanlab is not None:
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

    def _log_epoch(
        ep,
        tr_loss,
        va_loss,
        wr2,
        per_r2,
        per_mae=None,
        per_rmse=None,
        lr=None,
        aux_loss=None,
        aux_losses=None,
        drop_ratio=None,
    ):
        payload = {
            "train/loss": tr_loss,
            "val/loss": va_loss,
            "val/wr2": wr2,
        }
        if aux_loss is not None:
            payload["train/aux_loss"] = aux_loss
        if aux_losses:
            for i, v in enumerate(aux_losses):
                payload[f"train/aux_loss_e{i+1}"] = v
        if drop_ratio is not None:
            payload["train/expert_drop_ratio"] = drop_ratio
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
            swanlab.log(payload, step=ep)

    current_stage = stage_loaded
    epochs_without_improvement = 0

    LOGGER.info(f"[Fold {fold}] Start training, epoch={start_epoch}, stage={current_stage}")

    for ep in range(start_epoch, CFG.epochs + 1):
        stage = 1 if ep <= CFG.freeze_epochs else 2
        if stage != current_stage:
            LOGGER.info(f"[Fold {fold}] Switch to Stage {stage}")
            set_stage(stage)
            current_stage = stage

        drop_p = CFG.expert_drop_path_max * min(1.0, ep / max(1, CFG.expert_drop_warmup))
        if isinstance(model, nn.DataParallel):
            model.module.set_drop_prob(drop_p)
        else:
            model.set_drop_prob(drop_p)

        tr_loss, aux_losses, drop_ratio = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, ep, ema=ema, sw_run=run
        )
        va_loss, wr2, per_r2, per_mae, per_rmse, y_true, y_pred = validate(
            model,
            valid_loader,
            criterion,
            sw_run=run,
            epoch_idx=ep,
            log_images=(ep % CFG.log_image_every == 0),
            log_attn=(ep % CFG.log_attn_every == 0),
            ema=ema,
        )

        if scheduler is not None and stage == 2:
            scheduler.step()

        lr_cur = optimizer.param_groups[0]["lr"] if optimizer else None
        aux_mean = float(np.mean(aux_losses)) if aux_losses else None
        _log_epoch(
            ep,
            tr_loss,
            va_loss,
            wr2,
            per_r2,
            per_mae,
            per_rmse,
            lr_cur,
            aux_loss=aux_mean,
            aux_losses=aux_losses,
            drop_ratio=drop_ratio,
        )

        lr_str = f"{lr_cur:.2e}" if lr_cur else "N/A"
        LOGGER.info(
            f"[Fold {fold}] Epoch {ep}/{CFG.epochs} | Stage {stage} | "
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

        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        model_state = actual_model.state_dict()
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
            "ema_state": ema.state_dict() if ema is not None else {},
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
            LOGGER.info(f"[Fold {fold}] New best WR2: {best_wr2:.4f} (Epoch {ep})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if va_loss < best_loss:
            best_loss = va_loss
            state["best_loss"] = best_loss
            save_checkpoint(state, ckpt_dir / "best_loss.pt")
            LOGGER.info(f"[Fold {fold}] New best Loss: {best_loss:.4f} (Epoch {ep})")

        if epochs_without_improvement >= CFG.patience:
            LOGGER.info(f"[Fold {fold}] Early stop: {CFG.patience} epochs without improvement")
            break

    if CFG.save_history_csv:
        pd.DataFrame(history_rows).to_csv(metrics_path, index=False)
        LOGGER.info(f"[Fold {fold}] History saved: {metrics_path}")

    LOGGER.info("=" * 80)
    LOGGER.info(f"[Fold {fold}] Training summary")
    LOGGER.info(f"  Best WR2: {best_wr2:.4f}")
    LOGGER.info(f"  Best Loss: {best_loss:.4f}")
    LOGGER.info(f"  Checkpoints: {ckpt_dir}")
    LOGGER.info("=" * 80)

    if run is not None:
        try:
            run.finish()
            LOGGER.info(f"[Fold {fold}] SwanLab run finished")
        except Exception as e:
            LOGGER.warning(f"[Fold {fold}] SwanLab finish error: {e}")


def export_config():
    cfg_path = Path(CFG.experiment_dir) / "config.json"
    cfg_dict = asdict(CFG)
    cfg_dict["git_commit"] = get_git_commit()
    cfg_dict["device"] = str(CFG.device)
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    LOGGER.info(f"Config saved to {cfg_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="CSIRO Multi-Branch CNN Training")
    parser.add_argument("--fold", type=int, default=-1, help="Train only one fold if specified")
    parser.add_argument("--resume", type=str, default="", help="Resume checkpoint path")
    parser.add_argument(
        "--resume-mode",
        type=str,
        default="auto",
        choices=["auto", "last", "best_wr2", "best_loss", "none"],
        help="Resume mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(CFG.seed)
    CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_config()

    LOGGER.info("=" * 80)
    LOGGER.info("=== Multi-Branch CNN Training ===")
    LOGGER.info(f"Device: {CFG.device}, Mixed Precision: {CFG.mixed_precision}")
    LOGGER.info(f"Experiment dir: {CFG.experiment_dir}")
    LOGGER.info(f"Experts: {[c['name'] for c in CFG.expert_configs]}")
    LOGGER.info(f"Input Size: {CFG.input_size}")
    LOGGER.info("=" * 80)

    df_wide = load_train_df()
    df_wide = prepare_meta(df_wide)
    df_wide = add_folds(df_wide)
    LOGGER.info(f"Dataset size: {len(df_wide)}")

    folds = [args.fold] if args.fold >= 0 else list(range(CFG.n_splits))
    resume_path = Path(args.resume) if args.resume else None
    resume_mode = args.resume_mode

    LOGGER.info(f"Training folds: {folds}, Resume Mode: {resume_mode}")

    for fold in folds:
        LOGGER.info("=" * 80)
        LOGGER.info(f"Start Fold {fold}/{CFG.n_splits - 1}")
        LOGGER.info("=" * 80)
        try:
            run_fold(fold, df_wide, CFG.project, resume_path=resume_path, resume_mode=resume_mode)
            LOGGER.info(f"✓ Fold {fold} done")
        except KeyboardInterrupt:
            LOGGER.warning(f"Fold {fold} interrupted")
            raise
        except Exception as e:
            LOGGER.error(f"✗ Fold {fold} failed: {e}", exc_info=True)
            LOGGER.info("Continue to next fold...")

        gc.collect()
        torch.cuda.empty_cache()
        if fold < folds[-1]:
            LOGGER.info("Wait 3 seconds for next fold...")
            time.sleep(3)

    LOGGER.info("=" * 80)
    LOGGER.info("All folds finished")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()

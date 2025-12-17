# =============================================================================
# CSIRO Image2Biomass - v4 CrossPVT T2T Mamba Inference
# -----------------------------------------------------------------------------
# - 5-fold ensemble + TTA (原图/水平翻转/垂直翻转)
# - 从 fold_X/checkpoints/best_wr2.pt 加载权重
# - 自动处理 DataParallel 的 module. 前缀
# - 输出 submission.csv
# =============================================================================

import os
import gc
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 导入训练代码中的模型和配置
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import CrossPVT_T2T_MambaDINO, CFG as TrainCFG

# =============================================================================
# 配置
# =============================================================================
class INF_CFG:
    # 数据路径（Kaggle 默认，本地测试可修改）
    # BASE_PATH = "/kaggle/input/csiro-biomass"
    BASE_PATH = "/home/aaa/Kaggle-Series-Competition/CSIRO-Image2Biomass-Prediction/csiro-biomass"
    TEST_CSV = os.path.join(BASE_PATH, "test.csv")
    TEST_IMAGE_DIR = os.path.join(BASE_PATH, "test")
    
    # 实验目录（checkpoint 所在位置）
    EXPERIMENT_DIR = str(Path(__file__).parent)
    
    # Checkpoint 路径（5-fold）
    # 支持两种路径格式：
    # 1. fold_0/checkpoints/best_wr2.pt (训练代码保存的格式)
    # 2. fold0/checkpoints/best_wr2.pt (备用格式)
    CKPT_PATTERN_FOLD_X = os.path.join(EXPERIMENT_DIR, "fold_{fold}", "checkpoints", "best_wr2.pt")
    CKPT_PATTERN_FOLDX = os.path.join(EXPERIMENT_DIR, "fold{fold}", "checkpoints", "best_wr2.pt")
    N_FOLDS = 5
    
    # 输出
    SUBMISSION_FILE = "submission.csv"
    
    # 推理设置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    MIXED_PRECISION = True
    
    # TTA 设置
    USE_TTA = True
    TTA_TRANSFORMS = ["original", "hflip", "vflip"]  # 原图、水平翻转、垂直翻转
    
    # 目标列顺序（与训练一致）
    ALL_TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


print(f"Device: {INF_CFG.DEVICE}")
print(f"Experiment Dir: {INF_CFG.EXPERIMENT_DIR}")


# =============================================================================
# 数据集
# =============================================================================
class TestBiomassDataset(Dataset):
    """测试数据集：左右两路输入"""
    
    def __init__(self, df: pd.DataFrame, transform, image_dir: str):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_dir = image_dir
        self.paths = self.df["image_path"].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = os.path.basename(self.paths[idx])
        full_path = os.path.join(self.image_dir, filename)
        
        img = cv2.imread(full_path)
        if img is None:
            # 容错：若读图失败，用黑图占位
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 左右切半（与训练一致）
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        
        left_t = self.transform(image=left)["image"]
        right_t = self.transform(image=right)["image"]
        
        return left_t, right_t


# =============================================================================
# TTA 变换
# =============================================================================
def get_tta_transforms(img_size: int) -> List[A.Compose]:
    """生成 TTA 变换列表"""
    base = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    transforms = []
    
    # 原图
    transforms.append(
        A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            *base
        ])
    )
    
    # 水平翻转
    transforms.append(
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            *base
        ])
    )
    
    # 垂直翻转
    transforms.append(
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            *base
        ])
    )
    
    return transforms


# =============================================================================
# 权重加载
# =============================================================================
def strip_module_prefix(state_dict: dict) -> dict:
    """移除 DataParallel 的 module. 前缀"""
    if not state_dict:
        return state_dict
    
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(path: str) -> dict:
    """加载 checkpoint"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    # 兼容 PyTorch 2.6 的 weights_only 变更
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    
    return state


def load_model_from_checkpoint(ckpt_path: str) -> nn.Module:
    """从 checkpoint 加载模型"""
    print(f"\n加载 checkpoint: {os.path.basename(ckpt_path)}")
    
    state = load_checkpoint(ckpt_path)
    
    # 提取模型状态
    model_state = state.get("model_state")
    if model_state is None:
        # 如果 checkpoint 直接是 state_dict
        model_state = state
    
    # 移除 module. 前缀
    model_state = strip_module_prefix(model_state)
    
    # 从 checkpoint 中读取配置（如果存在）
    cfg_dict = state.get("cfg", {})
    dropout = cfg_dict.get("dropout", TrainCFG.dropout)
    hidden_ratio = cfg_dict.get("hidden_ratio", TrainCFG.hidden_ratio)
    
    # 创建模型（使用训练时的配置）
    model = CrossPVT_T2T_MambaDINO(dropout=dropout, hidden_ratio=hidden_ratio)
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    
    if missing_keys:
        print(f"  ⚠️  缺失的键: {len(missing_keys)} 个")
        if len(missing_keys) <= 10:
            for k in missing_keys[:10]:
                print(f"    - {k}")
    
    if unexpected_keys:
        print(f"  ⚠️  意外的键: {len(unexpected_keys)} 个")
        if len(unexpected_keys) <= 10:
            for k in unexpected_keys[:10]:
                print(f"    - {k}")
    
    model.to(INF_CFG.DEVICE)
    model.eval()
    
    # 获取输入分辨率
    input_res = getattr(model, "input_res", 518)
    backbone_name = getattr(model, "backbone_name", "unknown")
    
    print(f"  ✓ 模型加载成功 | backbone={backbone_name} | input_res={input_res}")
    
    return model


# =============================================================================
# 推理
# =============================================================================
def pack5_targets(total: torch.Tensor, gdm: torch.Tensor, green: torch.Tensor) -> torch.Tensor:
    """将 total, gdm, green 打包为 5 个目标"""
    clover = gdm - green
    dead = total - gdm
    return torch.cat([green, dead, clover, gdm, total], dim=1)


@torch.no_grad()
def predict_one_view(models: List[nn.Module], loader: DataLoader) -> np.ndarray:
    """对单个 TTA 视角进行预测"""
    preds_list = []
    amp_dtype = "cuda" if INF_CFG.DEVICE.type == "cuda" else "cpu"
    
    for xl, xr in tqdm(loader, desc="  Predicting", leave=False):
        xl = xl.to(INF_CFG.DEVICE, non_blocking=True)
        xr = xr.to(INF_CFG.DEVICE, non_blocking=True)
        
        # 拼接为单 tensor（与训练时的 DataParallel 调用方式一致）
        x_cat = torch.cat([xl, xr], dim=1)
        
        per_model_preds = []
        
        with torch.amp.autocast(amp_dtype, enabled=INF_CFG.MIXED_PRECISION):
            for model in models:
                out = model(x_cat, return_features=False)
                
                total = out["total"]
                gdm = out["gdm"]
                green = out["green"]
                
                # 打包为 5 个目标
                five = pack5_targets(total, gdm, green)
                
                # 非负约束
                five = torch.clamp(five, min=0.0)
                
                per_model_preds.append(five.float().cpu())
        
        # 5-fold ensemble 平均
        stacked = torch.mean(torch.stack(per_model_preds, dim=0), dim=0)
        preds_list.append(stacked.numpy())
    
    return np.concatenate(preds_list, axis=0)


def run_inference(test_df: pd.DataFrame, image_dir: str) -> np.ndarray:
    """运行完整推理流程（5-fold ensemble + TTA）"""
    print("\n" + "=" * 80)
    print("开始推理")
    print("=" * 80)
    
    # 加载所有 fold 的模型
    print("\n加载模型 (5-fold)...")
    models = []
    input_res = None
    
    for fold in range(INF_CFG.N_FOLDS):
        # 尝试两种路径格式
        ckpt_path = INF_CFG.CKPT_PATTERN_FOLD_X.format(fold=fold)
        if not os.path.exists(ckpt_path):
            ckpt_path = INF_CFG.CKPT_PATTERN_FOLDX.format(fold=fold)
        
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  Fold {fold} checkpoint 不存在，尝试路径:")
            print(f"    - {INF_CFG.CKPT_PATTERN_FOLD_X.format(fold=fold)}")
            print(f"    - {INF_CFG.CKPT_PATTERN_FOLDX.format(fold=fold)}")
            continue
        
        model = load_model_from_checkpoint(ckpt_path)
        models.append(model)
        
        # 使用第一个模型确定输入分辨率
        if input_res is None:
            input_res = getattr(model, "input_res", 518)
            print(f"  输入分辨率: {input_res}")
    
    if len(models) == 0:
        print("\n❌ 错误：没有找到任何可用的 checkpoint！")
        print(f"   请检查实验目录: {INF_CFG.EXPERIMENT_DIR}")
        print(f"   Checkpoint 应该位于:")
        for fold in range(INF_CFG.N_FOLDS):
            print(f"     - {INF_CFG.CKPT_PATTERN_FOLD_X.format(fold=fold)}")
            print(f"     - {INF_CFG.CKPT_PATTERN_FOLDX.format(fold=fold)}")
        raise RuntimeError("没有找到任何可用的 checkpoint！")
    
    print(f"\n✓ 成功加载 {len(models)} 个模型")
    
    # TTA 推理
    if INF_CFG.USE_TTA:
        tta_transforms = get_tta_transforms(input_res)
        print(f"\n使用 TTA: {len(tta_transforms)} 个视角")
        
        per_view_preds = []
        
        for i, transform in enumerate(tta_transforms):
            view_name = INF_CFG.TTA_TRANSFORMS[i] if i < len(INF_CFG.TTA_TRANSFORMS) else f"view_{i+1}"
            print(f"\n--- TTA 视角 {i+1}/{len(tta_transforms)}: {view_name} ---")
            
            ds = TestBiomassDataset(test_df, transform, image_dir)
            dl = DataLoader(
                ds,
                batch_size=INF_CFG.BATCH_SIZE,
                shuffle=False,
                num_workers=INF_CFG.NUM_WORKERS,
                pin_memory=True
            )
            
            view_pred = predict_one_view(models, dl)
            per_view_preds.append(view_pred)
        
        # TTA 平均
        final_pred = np.mean(per_view_preds, axis=0)
        print(f"\n✓ TTA 完成，最终预测形状: {final_pred.shape}")
    else:
        # 不使用 TTA
        transform = get_tta_transforms(input_res)[0]
        ds = TestBiomassDataset(test_df, transform, image_dir)
        dl = DataLoader(
            ds,
            batch_size=INF_CFG.BATCH_SIZE,
            shuffle=False,
            num_workers=INF_CFG.NUM_WORKERS,
            pin_memory=True
        )
        final_pred = predict_one_view(models, dl)
    
    return final_pred


# =============================================================================
# 生成提交文件
# =============================================================================
def create_submission(final_pred: np.ndarray, test_long: pd.DataFrame, test_unique: pd.DataFrame) -> pd.DataFrame:
    """生成提交文件"""
    print("\n" + "=" * 80)
    print("生成提交文件")
    print("=" * 80)
    
    # 提取各目标
    green = final_pred[:, 0]
    dead = final_pred[:, 1]
    clover = final_pred[:, 2]
    gdm = final_pred[:, 3]
    total = final_pred[:, 4]
    
    # 最终非负裁剪与 NaN/Inf 处理
    def clean(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(0, x)
    
    green, dead, clover, gdm, total = map(clean, [green, dead, clover, gdm, total])
    
    # 构建宽表
    wide = pd.DataFrame({
        "image_path": test_unique["image_path"],
        "Dry_Green_g": green,
        "Dry_Dead_g": dead,
        "Dry_Clover_g": clover,
        "GDM_g": gdm,
        "Dry_Total_g": total,
    })
    
    # 转换为长表
    long_preds = wide.melt(
        id_vars=["image_path"],
        value_vars=INF_CFG.ALL_TARGET_COLS,
        var_name="target_name",
        value_name="target"
    )
    
    # 合并到测试集
    sub = pd.merge(
        test_long[["sample_id", "image_path", "target_name"]],
        long_preds,
        on=["image_path", "target_name"],
        how="left"
    )[["sample_id", "target"]]
    
    # 最终清理
    sub["target"] = np.nan_to_num(sub["target"], nan=0.0, posinf=0.0, neginf=0.0)
    
    # 保存
    sub.to_csv(INF_CFG.SUBMISSION_FILE, index=False)
    
    print(f"\n✓ 提交文件已保存: {INF_CFG.SUBMISSION_FILE}")
    print(f"  样本数: {len(sub)}")
    print(f"  预测统计:")
    print(f"    Dry_Green_g:   mean={green.mean():.2f}, std={green.std():.2f}, min={green.min():.2f}, max={green.max():.2f}")
    print(f"    Dry_Dead_g:    mean={dead.mean():.2f}, std={dead.std():.2f}, min={dead.min():.2f}, max={dead.max():.2f}")
    print(f"    Dry_Clover_g:  mean={clover.mean():.2f}, std={clover.std():.2f}, min={clover.min():.2f}, max={clover.max():.2f}")
    print(f"    GDM_g:         mean={gdm.mean():.2f}, std={gdm.std():.2f}, min={gdm.min():.2f}, max={gdm.max():.2f}")
    print(f"    Dry_Total_g:   mean={total.mean():.2f}, std={total.std():.2f}, min={total.min():.2f}, max={total.max():.2f}")
    print(f"\n前 10 行预览:")
    print(sub.head(10).to_string())
    
    return sub


# =============================================================================
# 主函数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CSIRO v4 CrossPVT T2T Mamba Inference")
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="测试集 CSV 路径（默认: INF_CFG.TEST_CSV）"
    )
    parser.add_argument(
        "--test-image-dir",
        type=str,
        default=None,
        help="测试图像目录（默认: INF_CFG.TEST_IMAGE_DIR）"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="实验目录（checkpoint 所在位置，默认: INF_CFG.EXPERIMENT_DIR）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认: INF_CFG.SUBMISSION_FILE）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批次大小（默认: INF_CFG.BATCH_SIZE）"
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="禁用 TTA"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 更新配置（如果提供了命令行参数）
    if args.test_csv:
        INF_CFG.TEST_CSV = args.test_csv
    if args.test_image_dir:
        INF_CFG.TEST_IMAGE_DIR = args.test_image_dir
    if args.experiment_dir:
        INF_CFG.EXPERIMENT_DIR = args.experiment_dir
        # 更新 checkpoint 路径模式
        INF_CFG.CKPT_PATTERN_FOLD_X = os.path.join(INF_CFG.EXPERIMENT_DIR, "fold_{fold}", "checkpoints", "best_wr2.pt")
        INF_CFG.CKPT_PATTERN_FOLDX = os.path.join(INF_CFG.EXPERIMENT_DIR, "fold{fold}", "checkpoints", "best_wr2.pt")
    if args.output:
        INF_CFG.SUBMISSION_FILE = args.output
    if args.batch_size:
        INF_CFG.BATCH_SIZE = args.batch_size
    if args.no_tta:
        INF_CFG.USE_TTA = False
    
    print("=" * 80)
    print("CSIRO Image2Biomass - v4 CrossPVT T2T Mamba Inference")
    print("=" * 80)
    print(f"测试 CSV: {INF_CFG.TEST_CSV}")
    print(f"测试图像目录: {INF_CFG.TEST_IMAGE_DIR}")
    print(f"实验目录: {INF_CFG.EXPERIMENT_DIR}")
    print(f"输出文件: {INF_CFG.SUBMISSION_FILE}")
    print(f"批次大小: {INF_CFG.BATCH_SIZE}")
    print(f"使用 TTA: {INF_CFG.USE_TTA}")
    
    # 加载测试数据
    print("\n加载测试数据...")
    if not os.path.exists(INF_CFG.TEST_CSV):
        raise FileNotFoundError(f"测试 CSV 不存在: {INF_CFG.TEST_CSV}")
    
    test_long = pd.read_csv(INF_CFG.TEST_CSV)
    test_unique = test_long.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    print(f"✓ 找到 {len(test_unique)} 张独立测试图像")
    print(f"  总测试样本数: {len(test_long)}")
    
    # 运行推理
    final_pred = run_inference(test_unique, INF_CFG.TEST_IMAGE_DIR)
    
    # 生成提交文件
    submission = create_submission(final_pred, test_long, test_unique)
    
    print("\n" + "=" * 80)
    print("推理完成！")
    print("=" * 80)
    
    # 清理
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


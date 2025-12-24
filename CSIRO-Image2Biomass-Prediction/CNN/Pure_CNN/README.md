# CNN Pure 生物量预测模型

基于 CNN 主干（ConvNeXt/EfficientNetV2）的直接整图输入生物量预测模型。

## 架构特点

### 1. Direct Full Image Input
- 直接将完整图像（2000×1000）作为输入
- 无 Tile 切分，无 FiLM 调制
- 简化架构，更快的推理速度

### 2. CNN 主干
支持多种预训练主干（按优先级尝试）：
- ConvNeXt-Base (~89M params)
- ConvNeXt-Small (~50M params)
- EfficientNetV2-M (~54M params)
- EfficientNetV2-S (~24M params)
- EfficientNet-B4 (~19M params)
- ResNet50 (fallback)

### 3. 三路回归头
- `head_green`: 预测 Dry_Green_g
- `head_clover`: 预测 Dry_Clover_g
- `head_dead`: 预测 Dry_Dead_g
- 使用 Softplus 约束输出为正值
- 组合计算：`GDM = green + clover`, `Total = GDM + dead`

### 4. 物理一致损失
- 加权 Smooth L1 损失
- 物理约束：`green + clover ≈ GDM`, `green + clover + dead ≈ Total`

## 使用方法

### 训练所有 Folds
```bash
python train.py
```

### 训练指定 Fold
```bash
python train.py --fold 0
```

### 断点续训
```bash
# 自动检测最近的 checkpoint
python train.py --resume-mode auto

# 从指定 checkpoint 恢复
python train.py --resume /path/to/checkpoint.pt

# 从最佳 WR2 恢复
python train.py --resume-mode best_wr2
```

## 配置

主要配置在 `CFG` dataclass 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 80 | 总训练轮数 |
| `freeze_epochs` | 20 | 冻结主干的轮数 |
| `batch_size` | 8 | 批次大小 |
| `input_size` | 384 | 输入图像尺寸 |
| `dropout` | 0.2 | Dropout 比例 |
| `lr_backbone` | 1e-5 | 主干学习率 |
| `lr_head` | 5e-4 | 头部学习率 |
| `patience` | 15 | 早停耐心值 |

## 输出

训练输出保存在 `CNN/Pure_CNN/` 目录：

```
CNN/Pure_CNN/
├── train.py              # 训练脚本
├── config.json           # 配置文件
├── train.log             # 训练日志
├── fold{i}_metrics.csv   # 每个 fold 的训练历史
└── checkpoints/
    └── fold{i}/
        ├── last.pt       # 最后一个 epoch
        ├── best_wr2.pt   # 最佳 WR2
        └── best_loss.pt  # 最佳 Loss
```

## SwanLab 集成

如果安装了 `swanlab`，训练过程会自动记录到 SwanLab：

```bash
pip install swanlab
```

记录的指标包括：
- `train/loss`: 训练损失
- `val/loss`: 验证损失
- `val/wr2`: 加权 R² 分数
- `val/r2_{target}`: 每个目标的 R² 分数
- `val/mae_{target}`: 每个目标的 MAE
- `train/lr`: 当前学习率

## 依赖

```
torch>=2.0
timm>=0.9.0
albumentations>=1.3.0
pandas
numpy
scikit-learn
tqdm
swanlab (optional)
```

## 目标说明

| 目标 | 说明 | 权重 |
|------|------|------|
| Dry_Green_g | 绿色植被（不含三叶草）| 0.1 |
| Dry_Dead_g | 枯死物质 | 0.1 |
| Dry_Clover_g | 三叶草 | 0.1 |
| GDM_g | 绿色干物质 = Green + Clover | 0.2 |
| Dry_Total_g | 总干物质 = GDM + Dead | 0.5 |

#!/usr/bin/env python3
"""
CSIRO Image2Biomass数据处理脚本
将train.csv中的长格式数据转换为宽格式数据
使用Python内置模块，不依赖外部库
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

def process_train_data():
    """处理训练数据，将target_name和target列转换为五列格式"""

    # 定义输入输出路径
    input_path = "/home/aaa/Kaggle-Series-Competition/CSIRO-Image2Biomass-Prediction/csiro-biomass/train.csv"
    output_dir = "/home/aaa/Kaggle-Series-Competition/CSIRO-Image2Biomass-Prediction/data_analysis/processed_train_data"
    output_path = os.path.join(output_dir, "train_processed.csv")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 读取数据
    print("正在读取训练数据...")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    print(f"原始数据行数: {len(data)}")
    if data:
        print(f"原始数据列名: {list(data[0].keys())}")

    # 解析sample_id，提取真实的sample_id和target_name
    print("正在解析sample_id...")
    processed_data = {}
    target_types = set()

    for row in data:
        sample_id_full = row['sample_id']
        parts = sample_id_full.split('__')
        if len(parts) != 2:
            print(f"警告: 无法解析sample_id: {sample_id_full}")
            continue

        sample_id_clean = parts[0]
        target_name = parts[1]
        target_types.add(target_name)

        # 如果是新的sample_id，初始化记录
        if sample_id_clean not in processed_data:
            processed_data[sample_id_clean] = {
                'sample_id': sample_id_clean,
                'image_path': row['image_path'],
                'Sampling_Date': row['Sampling_Date'],
                'State': row['State'],
                'Species': row['Species'],
                'Pre_GSHH_NDVI': row['Pre_GSHH_NDVI'],
                'Height_Ave_cm': row['Height_Ave_cm']
            }

        # 添加target值
        processed_data[sample_id_clean][target_name] = row['target']

    # 验证target_name的唯一值
    expected_targets = {'Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g'}
    print(f"发现的target_name: {sorted(target_types)}")
    print(f"期望的target_name: {sorted(expected_targets)}")

    if target_types != expected_targets:
        raise ValueError("target_name不匹配期望值！")

    # 转换为列表格式用于保存
    final_data = []
    for sample_id, record in processed_data.items():
        final_data.append(record)

    # 确保所有记录都有所有target列，如果缺失则设为空字符串
    for record in final_data:
        for target in expected_targets:
            if target not in record:
                record[target] = ''

    # 保存处理后的数据
    print(f"正在保存处理后的数据到: {output_path}")

    if final_data:
        fieldnames = ['sample_id', 'image_path', 'Sampling_Date', 'State', 'Species',
                     'Pre_GSHH_NDVI', 'Height_Ave_cm', 'Dry_Clover_g', 'Dry_Dead_g',
                     'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_data)

    print(f"处理完成！")
    print(f"最终数据行数: {len(final_data)}")
    print(f"最终数据列名: {fieldnames}")

    # 显示前几行数据作为验证
    print("\n处理后数据的前5行:")
    for i, record in enumerate(final_data[:5]):
        print(f"行 {i+1}: {record}")

    return final_data

if __name__ == "__main__":
    process_train_data()

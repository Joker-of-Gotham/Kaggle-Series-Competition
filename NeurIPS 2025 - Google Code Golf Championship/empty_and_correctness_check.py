import os
import json
import numpy as np
import sys
import importlib.util
import copy
import re
import traceback
import time

# ==================================================================
# --- 路径配置 ---
# 请根据您的文件夹结构修改这些路径

# 解决方案 .py 文件所在的文件夹
SOLUTIONS_DIR = r"./solutions"

# 任务 .json 数据文件所在的文件夹
DATA_DIR = r"./data"
# ------------------------------------------------------------------


# --- 提供的工具函数 (来自 code_golf_utils.py) ---
# Copyright 2025 Google LLC
# (Licensing information as provided)
colors = [
    (0, 0, 0), (30, 147, 255), (250, 61, 49), (78, 204, 48), (255, 221, 0),
    (153, 153, 153), (229, 59, 163), (255, 133, 28), (136, 216, 241), (147, 17, 49),
]

def load_examples(task_num):
  """从 JSON 文件加载任务示例。"""
  file_path = os.path.join(DATA_DIR, f"task{task_num:03d}.json")
  with open(file_path, 'r', encoding='utf-8') as f:
    examples = json.load(f)
  return examples
  
# --- 批量验证的核心函数 ---
def batch_verify_solutions(start_task=1, end_task=400):
    """
    遍历指定范围内的所有任务，检查解决方案文件是否为空，
    如果不为空，则验证其正确性，并汇总报告结果。
    """
    empty_files = []
    failed_tasks = []
    passed_tasks = []
    
    # 确保用于动态导入的临时文件夹存在
    working_dir = "./working"
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        
    print(f"开始批量验证任务 {start_task} 到 {end_task}...")
    print("="*60)
    start_time = time.time()

    for task_num in range(start_task, end_task + 1):
        task_py_path = os.path.join(SOLUTIONS_DIR, f"task{task_num:03d}.py")
        task_json_path = os.path.join(DATA_DIR, f"task{task_num:03d}.json")
        
        # 检查解决方案文件是否存在且不为空
        if not os.path.exists(task_py_path) or os.path.getsize(task_py_path) == 0:
            empty_files.append(f"task{task_num:03d}.py")
            continue
            
        # 检查数据文件是否存在
        if not os.path.exists(task_json_path):
            print(f"警告：找不到任务数据文件 {task_json_path}，跳过任务 {task_num:03d}")
            continue

        # 运行单个任务的验证
        print(f"--- 正在测试 task{task_num:03d}.py ---")
        try:
            with open(task_py_path, 'r', encoding='utf-8') as f:
                solution_code = f.read()
            
            # 将代码写入临时文件以供导入
            temp_task_path = os.path.join(working_dir, "task.py")
            with open(temp_task_path, "w", encoding='utf-8') as f:
                f.write(solution_code)

            # 加载任务示例
            examples = load_examples(task_num)
            
            # --- 验证逻辑 ---
            task_name = f"task_verifier_{task_num}"
            spec = importlib.util.spec_from_file_location(task_name, temp_task_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, "p") or not callable(getattr(module, "p")):
                failed_tasks.append(f"task{task_num:03d}.py (错误: 找不到可调用的 p() 函数)")
                continue

            program = getattr(module, "p")
            
            all_examples = examples.get('train', []) + examples.get('test', []) + examples.get('arc-gen', [])
            wrong_count = 0
            for example in all_examples:
                example_copy = copy.deepcopy(example)
                try:
                    result = program(example_copy["input"])
                    user_output = np.array(result)
                    label_output = np.array(example_copy["output"])
                    if not np.array_equal(user_output, label_output):
                        wrong_count += 1
                        break
                except Exception:
                    wrong_count += 1
                    break
            
            if wrong_count == 0:
                length = os.path.getsize(task_py_path)
                print(f"✅ PASSED (大小: {length} 字节)")
                passed_tasks.append(f"task{task_num:03d}.py")
            else:
                print(f"❌ FAILED")
                failed_tasks.append(f"task{task_num:03d}.py")

        except Exception as e:
            print(f"❌ FAILED (发生严重错误: {e})")
            failed_tasks.append(f"task{task_num:03d}.py (严重错误)")

    end_time = time.time()
    
    # --- 最终报告 ---
    print("\n" + "="*60)
    print("           批量验证最终报告")
    print("="*60)
    print(f"总耗时: {end_time - start_time:.2f} 秒\n")

    print(f"✅ 通过测试的任务 ({len(passed_tasks)}个):")
    if passed_tasks:
        print(", ".join(passed_tasks))
    else:
        print("无")
        
    print(f"\n❌ 未通过测试的任务 ({len(failed_tasks)}个):")
    if failed_tasks:
        print(", ".join(failed_tasks))
    else:
        print("无")

    print(f"\n텅 빈 空文件或不存在的文件 ({len(empty_files)}个):")
    if empty_files:
        print(", ".join(empty_files))
    else:
        print("无")
        
    print("\n" + "="*60)

# --- 运行批量验证 ---
# 您可以修改这里的范围，例如 batch_verify_solutions(1, 10) 只测试前10个任务
batch_verify_solutions(1, 400)
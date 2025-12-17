import os

# --- 配置区 ---
# 目标文件夹路径。请确保这个路径是正确的。
# 使用 'r' 前缀可以确保 Windows 的反斜杠 '\' 不会被错误地转义。
solutions_dir = r"D:\Kaggle_Sereies\NeurIPS 2025 - Google Code Golf Championship\solutions"

# --- 脚本执行区 ---

# 1. 检查目标文件夹是否存在，如果不存在则创建它
if not os.path.exists(solutions_dir):
    print(f"文件夹不存在，正在创建: {solutions_dir}")
    os.makedirs(solutions_dir)
else:
    print(f"文件夹已存在: {solutions_dir}")

# 2. 循环创建文件
files_created = 0
for i in range(10, 401):
    # 格式化文件名，确保数字是三位数且有前导零 (例如, 10 -> 010, 123 -> 123)
    filename = f"task{i:03d}.py"
    
    # 组合成完整的文件路径
    file_path = os.path.join(solutions_dir, filename)
    
    # 创建一个空的 .py 文件
    # 'w' 模式会创建文件（如果不存在）并立即关闭它
    with open(file_path, 'w') as f:
        # 你可以在这里写入一个默认的函数模板，如果需要的话
        # f.write("def p(g):\n")
        # f.write("    return g\n")
        pass # pass 会创建一个完全空的文件

    files_created += 1

print(f"\n操作完成！")
print(f"总共创建了 {files_created} 个空文件。")
print(f"文件已保存至: {os.path.abspath(solutions_dir)}") # 显示绝对路径以供确认
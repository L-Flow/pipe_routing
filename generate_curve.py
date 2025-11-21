#!/usr/bin/env python3
# generate_curve.py
#
# 这个脚本读取 PPO Agent 生成的控制点和权重，
# 1. 构造最终的 NURBS 曲线。
# 2. 将曲线导出为 JSON (用于存档)。
# 3. 将曲线的 200 个采样点导出为 .txt (用于 SolidWorks)。

import numpy as np
from geomdl import NURBS, utilities
# [!! 修正 !!] 导入正确的 export_json 函数
from geomdl.exchange import export_json

# --- 1. 定义文件路径 ---
POINTS_FILE = "/home/ljh/PycharmProjects/pipe_env/final_control_points7.txt"
WEIGHTS_FILE = "/home/ljh/PycharmProjects/pipe_env/final_weights7.txt"

# 定义两个输出文件
OUTPUT_CURVE_JSON = "final_nurbs_curve7.json"  # 曲线的数学定义
OUTPUT_CURVE_POINTS_TXT = "final_curve_XYZ_points7.txt"  # SolidWorks 导入文件

# --- 2. 加载数据 ---
print(f"--- 正在从文件加载数据 ---")
try:
    control_points = np.loadtxt(POINTS_FILE).tolist()
    weights = np.loadtxt(WEIGHTS_FILE).tolist()
    print(f"成功加载 {len(control_points)} 个控制点和 {len(weights)} 个权重。")
except Exception as e:
    print(f"错误: 无法加载文件。 {e}")
    print(f"请确保您已成功运行 'test_model.py' 并生成了 '{POINTS_FILE}' 和 '{WEIGHTS_FILE}'。")
    exit()

if len(control_points) != len(weights):
    print("错误: 控制点和权重的数量不匹配！")
    exit()

# --- 3. 构建 NURBS 曲线 ---
curve_degree = 3
num_ctrlpts = len(control_points)

if num_ctrlpts < curve_degree + 1:
    print(f"错误: 三次曲线至少需要 4 个控制点, 但只找到了 {num_ctrlpts} 个。")
    exit()

print(f"--- 正在构建 {curve_degree} 阶 NURBS 曲线 ---")
curve = NURBS.Curve()
curve.degree = curve_degree
curve.ctrlpts = control_points
curve.weights = weights
curve.knotvector = utilities.generate_knot_vector(curve.degree, curve.ctrlpts_size)

# --- 4. 验证并导出 ---
print("--- 曲线构建成功 ---")

# 1) 导出 NURBS 定义 (JSON)
try:
    export_json(curve, OUTPUT_CURVE_JSON)
    print(f"\n[成功] 曲线的 JSON 定义已保存到: {OUTPUT_CURVE_JSON}")
except Exception as e:
    print(f"\n[警告] 导出 JSON 失败: {e}")

# 2) 导出 SolidWorks 的 XYZ 采样点
print(f"--- 正在为 SolidWorks 生成 {200} 个采样点 ---")
try:
    # 沿着曲线 [0, 1] 区间均匀采样 200 个点
    num_sample_points = 200
    eval_points = curve.evaluate_list(np.linspace(0, 1, num_sample_points))

    # 将点保存为 XYZ 文本文件
    np.savetxt(OUTPUT_CURVE_POINTS_TXT, np.array(eval_points), fmt="%.6f", delimiter="\t")

    print(f"\n[!! 成功 !!]")
    print(f"用于 SolidWorks 的 XYZ 点已保存到: {OUTPUT_CURVE_POINTS_TXT}")
    print("请导入 SolidWorks 生成三维曲线模型。")

except Exception as e:
    print(f"\n错误: 导出 XYZ 点失败。 {e}")
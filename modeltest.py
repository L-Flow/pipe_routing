#!/usr/bin/env python3
# test_model.py
#
# 加载一个训练好的 PPO 模型并运行它一次，
# 以生成并保存最终的管线控制点。

import torch
import numpy as np
import os
import json

#  关键  导入您的自定义环境
from pipe_env import PipeRoutingEnv

# --- 1. 定义环境参数 (必须与 train.py 完全一致!) ---

# a) 势能图文件路径
PE_MAP_FILE = "/home/ljh/PycharmProjects/octree_test1025/out_octree_pe/cylindrical_pe_leaflevel.npy"
META_FILE = "/home/ljh/PycharmProjects/octree_test1025/out_octree_pe/cylindrical_pe_meta.json"

# b) 起点和终点标定
START_PT = [-302.43, 360.42, -893.05]
START_N = [-1, 1.19, 0]
TARGET_PT = [460.44, 212.48, -612.75]
TARGET_N = [1, -0.9, 0]
PIPE_DIAMETER = 10.0

# c) 定义 env_fn (与 train.py 相同)
env_fn = lambda: PipeRoutingEnv(
    pe_map_path=PE_MAP_FILE,
    meta_path=META_FILE,
    start_point=START_PT,
    start_normal=START_N,
    target_point=TARGET_PT,
    target_normal=TARGET_N,
    pipe_diameter=PIPE_DIAMETER
)

# --- 2. 定义模型加载路径 ---
# (!! 确保这指向您训练好的模型 !!)
MODEL_PATH = "/home/ljh/PycharmProjects/pipe_env/ppo_training_results/run7/pyt_save/model.pt"

if not os.path.exists(MODEL_PATH):
    print(f"错误：找不到模型文件 {MODEL_PATH}")
    print("请先运行 train.py 进行训练。")
    exit()


# --- 3. 加载并运行模型 ---

def run_trained_agent():
    print(f"--- 正在加载模型: {MODEL_PATH} ---")

    # 1. 加载 Spinning Up 保存的 PyTorch 模型 [cite: 3036]
    # ac (Actor-Critic) 是包含策略网络和价值网络的对象
    ac = torch.load(MODEL_PATH)

    # 2. 创建一个环境实例
    env = env_fn()
    obs = env.reset()
    done = False

    print("--- 模型加载完毕, 开始运行评估... ---")

    # 运行一个回合
    while not done:
        # 3. 将观测值转换为 PyTorch 张量
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        # 4. (关键) 让模型给出“确定性”的动作
        #    我们不能使用 ac.act()，因为它总是随机采样的。
        #    我们必须直接访问策略网络 ac.pi 来获取分布的“均值”。
        with torch.no_grad():
            # ac.pi._distribution(obs) 会返回一个 Torch 分布对象
            distribution = ac.pi._distribution(obs_tensor)

        # .mean 属性就是确定性的（最佳）动作
        action_tensor = distribution.mean

        # 5. 将动作转回 numpy 格式以传入 Gym
        action = action_tensor.cpu().numpy()

        # 6. 在环境中执行一步
        obs, reward, done, info = env.step(action)

        # 可选：查看实时步骤
        # env.render()

    print("--- 回合结束 ---")

    # 7. (关键) 从环境中提取最终的控制点
    #    在 env.step() 检测到 done=True 后，它会自动添加 P_t_minus_2...P_t 等夹持点
    final_control_points = np.array(env.control_points)
    final_weights = np.array(env.control_weights)

    print(f"成功生成管线！总共 {len(final_control_points)} 个控制点。")

    # 8. 保存控制点和权重到文件
    output_file_pts = "final_control_points8.txt"
    output_file_wts = "final_weights8.txt"

    np.savetxt(output_file_pts, final_control_points, fmt="%.6f")
    np.savetxt(output_file_wts, final_weights, fmt="%.6f")

    print(f"最终的控制点已保存到: {output_file_pts}")
    print(f"最终的权重已保存到: {output_file_wts}")


if __name__ == "__main__":
    run_trained_agent()
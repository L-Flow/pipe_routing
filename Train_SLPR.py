#!/usr/bin/env python3
# train.py
#
# 使用 OpenAI Spinning Up 的 PPO 算法来训练 pipe_env.py 中定义的自定义环境。

import os
# 导入 Spinning Up 的 PPO (PyTorch 版)
from spinup import ppo_pytorch as ppo

# 导入自定义布管环境
from pipe_env import PipeRoutingEnv

# --- 1. 定义环境参数 ---
# (根据您的设置修改路径和坐标)

# a) 势能图文件路径
PE_MAP_FILE = "/home/ljh/PycharmProjects/octree_test1025/out_octree_pe/assembly7_pe_leaflevel.npy"
META_FILE = "/home/ljh/PycharmProjects/octree_test1025/out_octree_pe/assembly7_pe_meta.json"

# b) 检查文件是否存在
if not (os.path.exists(PE_MAP_FILE) and os.path.exists(META_FILE)):
    print(f"错误：找不到环境文件。")
    print(f"请先运行 'octree_oedt_pipeline.py' 生成 {PE_MAP_FILE} 和 {META_FILE}")
    exit()

# c) 起点和终点标定
START_PT = [-302.43, 360.42, -893.05]
START_N = [-1, 1.19, 0]
TARGET_PT = [302.43, 360.42, -893.05]
TARGET_N = [1, 1.19, 0]
PIPE_DIAMETER = 10.0

# --- 2. 创建 env_fn (环境函数) ---
# Spinning Up 的 PPO 算法需要一个“环境函数” (env_fn)。
# 这是一个可调用的函数 (lambda)，它在被调用时返回您的环境的一个新实例。
env_fn = lambda: PipeRoutingEnv(
    pe_map_path=PE_MAP_FILE,
    meta_path=META_FILE,
    start_point=START_PT,
    start_normal=START_N,
    target_point=TARGET_PT,
    target_normal=TARGET_N,
    pipe_diameter=PIPE_DIAMETER
)

# --- 3. 设置 PPO 超参数 ---
# 基于论文中的表1

# a) 神经网络架构
#  2 个隐藏层, 每层 256 个神经元
ac_kwargs = dict(
    hidden_sizes=[256, 256]
)

# b) PPO 和 GAE (折扣因子等)
gamma = 0.9          #
lam = 0.98           #
clip_ratio = 0.2     #


# c) 学习率
pi_lr = 3e-5         # Actor (Policy) 学习率TODO：已经尝试修改，与论文不同
vf_lr = 5e-4         # Critic (Value F) 学习率


# d) 训练量
# 论文: 5000 训练回合 (episodes)
# 论文: 每回合最多 20 步 (steps)
total_episodes = 4000
max_ep_len = 40   #TODO：调试中

# Spinning Up 按“步数”来组织
# 计算出 PPO 运行的总交互步数：
total_steps = total_episodes * max_ep_len  # 5000 * 20 = 100,000

# 告诉 Spinning Up 每 4000 步收集一次数据并执行一次更新
#
steps_per_epoch = 4000

# 计算出总共需要多少个“Epochs”（更新周期）
epochs = total_steps // steps_per_epoch
if epochs == 0:  # 确保至少运行 1 次
    epochs = 1
    steps_per_epoch = total_steps

# --- 4. 设置 Logger (日志) ---
# 定义 PPO 训练结果的保存位置
output_dir = "ppo_training_results/run7"
exp_name = "pipe_routing_ppo_run7"

logger_kwargs = dict(
    output_dir=output_dir,
    exp_name=exp_name
)

# --- 5. 开始训练 ---
if __name__ == "__main__":
    print("==================================================")
    print("--- 开始 PPO 训练 (复现 SLPR 论文) ---")
    print(f"环境: PipeRoutingEnv")
    print(f"PPO 算法库: OpenAI Spinning Up (PyTorch)")
    print(f"总交互步数: {total_steps} ({epochs} 个 Epochs, 每 Epoch {steps_per_epoch} 步)")
    print(f"每回合最大步数: {max_ep_len}")
    print(f"保存到: {output_dir}")
    print("==================================================")

    # 调用 Spinning Up PPO 函数
    ppo(
        env_fn=env_fn,  # 自定义环境
        ac_kwargs=ac_kwargs,  #  网络架构
        steps_per_epoch=steps_per_epoch,  # 每次更新收集的步数
        epochs=epochs,  # 总共的更新次数
        gamma=gamma,  # 折扣因子
        clip_ratio=clip_ratio,  # PPO 裁剪率
        pi_lr=pi_lr,  # Policy 学习率
        vf_lr=vf_lr,  # Value 学习率
        lam=lam,  # GAE Lambda
        max_ep_len=max_ep_len,  # 设为 20
        logger_kwargs=logger_kwargs  # 日志保存位置
    )

    print("--- 训练完成 ---")
#!/usr/bin/env python3
# pipe_env.py

import gym
from gym import spaces
import numpy as np
import json
import os

# --- 关键的 NURBS 库 ---
from geomdl import NURBS, utilities
# 只导入 operations 模块用于计算长度 (这个模块通常比较稳定)
from geomdl import operations


# --- 手动实现曲率和挠率计算函数 ---

def _calculate_curvature(derivs: list) -> float:
    """
    根据导数计算曲率 k(u)。
    输入 derivs 是 geomdl 返回的列表: [C(u), C'(u), C''(u), ...]
    公式: k(u) = |C'(u) x C''(u)| / |C'(u)|^3
    """
    # C'(u) 一阶导数
    c_prime = np.array(derivs[1])
    # C''(u) 二阶导数
    c_double_prime = np.array(derivs[2])

    # 检查导数模长，防止除以零 (例如直线段)
    c_prime_norm = np.linalg.norm(c_prime)
    if c_prime_norm < 1e-6:
        return 0.0

    # 分子: |C'(u) x C''(u)|
    cross_prod = np.cross(c_prime, c_double_prime)
    numerator = np.linalg.norm(cross_prod)

    # 分母: |C'(u)|^3
    denominator = c_prime_norm ** 3

    if denominator < 1e-6:
        return 0.0

    # 截断保护：防止数值不稳定导致曲率爆炸
    kappa = numerator / denominator
    return min(kappa, 10.0)


def _calculate_torsion(derivs: list) -> float:
    """
    根据导数计算挠率 t(u)。
    输入 derivs 是 geomdl 返回的列表: [C(u), C'(u), C''(u), C'''(u)]
    公式: t(u) = ( (C'(u) x C''(u)) . C'''(u) ) / |C'(u) x C''(u)|^2
    """
    # 获取各阶导数
    c_prime = np.array(derivs[1])
    c_double_prime = np.array(derivs[2])
    c_triple_prime = np.array(derivs[3])

    # C'(u) x C''(u)
    cross_prod = np.cross(c_prime, c_double_prime)
    cross_prod_norm_sq = np.linalg.norm(cross_prod) ** 2

    # 如果 |C' x C''|^2 接近于 0 (例如平面曲线或直线)，挠率为 0
    if cross_prod_norm_sq < 1e-6:
        return 0.0

    # 分子: (C'(u) x C''(u)) . C'''(u)  (混合积)
    numerator = np.dot(cross_prod, c_triple_prime)

    # 截断保护
    tau = numerator / cross_prod_norm_sq
    return min(abs(tau), 10.0)


# --- NURBS 曲线的封装 ---
class Geomdl_NURBS:
    """使用 geomdl 库来封装 NURBS 曲线。"""

    def __init__(self, ctrlpts: list, weights: list, degree: int = 3):
        self.degree = degree
        self.num_ctrlpts = len(ctrlpts)

        if self.num_ctrlpts < self.degree + 1:
            raise ValueError(
                f"NURBS 曲线至少需要 {self.degree + 1} 个控制点，但只收到了 {self.num_ctrlpts} 个。")

        self.curve = NURBS.Curve()
        self.curve.degree = self.degree
        self.curve.ctrlpts = ctrlpts
        self.curve.weights = weights
        self.curve.knotvector = utilities.generate_knot_vector(self.degree, self.num_ctrlpts)

    def sample_list(self, u_values: list) -> list:
        return self.curve.evaluate_list(u_values)

    def get_derivatives(self, u: float, order: int) -> list:
        """
        获取曲线在 u 处的导数。
        返回列表 [point, 1st_deriv, 2nd_deriv, 3rd_deriv...]
        """
        return self.curve.derivatives(u, order=order)

    @property
    def length(self) -> float:
        # 使用 operations.length_curve 计算长度
        return operations.length_curve(self.curve)

    def get_new_segment_parameter_range(self) -> tuple:
        n = self.num_ctrlpts - 1
        p = self.degree
        u_start = (n - p) / (n - p + 1)
        return u_start, 1.0


class PipeRoutingEnv(gym.Env):
    """
    符合 OpenAI Gym API 的自定义管线布局环境。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 pe_map_path: str,
                 meta_path: str,
                 start_point: list,
                 start_normal: list,
                 target_point: list,
                 target_normal: list,
                 pipe_diameter: float = 10.0):
        super(PipeRoutingEnv, self).__init__()

        # 1. 加载环境数据
        print(f"Loading PE map from {pe_map_path}...")
        self.pe_map = np.load(pe_map_path)
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.origin = np.array(self.meta['leaf_grid_origin'])
        self.leaf_size = float(self.meta['leaf_size'])
        self.grid_shape = self.meta['leaf_grid_shape']
        self.max_potential = np.max(self.pe_map)
        self.obstacle_penalty = -1.0

        # 2. 定义任务参数
        self.P_s = np.array(start_point)
        self.N_s = np.array(start_normal) / np.linalg.norm(start_normal)
        self.P_t = np.array(target_point)
        self.N_t = np.array(target_normal) / np.linalg.norm(target_normal)

        # 3. 预先计算端点
        self.P_0 = self.P_s + self.N_s * pipe_diameter
        self.P_1 = self.P_0 + self.N_s * pipe_diameter * 2
        self.P_t_minus_1 = self.P_t + self.N_t * pipe_diameter
        self.P_t_minus_2 = self.P_t_minus_1 + self.N_t * pipe_diameter * 2

        # 4. 定义超参数
        self.max_steps_per_episode = 70
        self.d_max = 100.0
        self.sampling_interval_dl = 5.0

        # [新增] 对齐引导的参数
        self.guidance_dist = 300.0  # 当距离目标小于这个值时，开始给对齐奖励(建议设为 2-3 倍的 d_max)

        # 奖励权重 (包含流阻项)
        self.reward_weights = {
            'R1_dist': 0.1,
            'R2_len': 0.002,
            'R3_obs': 0.05,
            'R_kappa': 0.05,  # 曲率惩罚权重
            'R_tau': 0.01,  # 挠率惩罚权重
            'R4_pe': 1,
            'R5_success': 20.0,  # [!!] 这里定义了 R5
            'R_align': 2.0
        }

        # 5. 定义传感器
        self.sensor_range = 200.0
        self.ray_step_size = self.leaf_size

        dir_list = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
        ]
        self.sensor_directions = [np.array(d) / np.linalg.norm(d) for d in dir_list]
        self.num_sensors = len(self.sensor_directions)

        # 6. 定义 Action Space
        self.action_step_size = 75.0
        action_low = np.array([-1.0, -1.0, -1.0, -1.0])
        action_high = np.array([1.0, 1.0, 1.0, 1.0])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # 7. 定义 Observation Space
        # 原来是: 3 (当前点) + 3 (目标向量) + num_sensors
        # 现在增加: 3 (当前末端切向量)
        obs_dim = 3 + 3 + 3 + self.num_sensors  # <--- 这里增加了 3
        obs_low = np.full(obs_dim, -np.inf)
        obs_high = np.full(obs_dim, np.inf)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 8. 初始化内部状态
        self.current_step = 0
        self.control_points = []
        self.control_weights = []
        self.current_point = None
        self.previous_point = None
        self.current_curve_length = 0.0

    def _query_potential(self, point_xyz) -> float:
        point_xyz = np.array(point_xyz)
        if point_xyz.ndim != 1:
            raise ValueError(f"Point must be a 1D array (x,y,z), but got {point_xyz.ndim} dimensions")
        idx = np.floor((point_xyz - self.origin) / self.leaf_size).astype(int)
        if (idx[0] < 0 or idx[0] >= self.grid_shape[0] or
                idx[1] < 0 or idx[1] >= self.grid_shape[1] or
                idx[2] < 0 or idx[2] >= self.grid_shape[2]):
            return self.obstacle_penalty
        return self.pe_map[idx[0], idx[1], idx[2]]

    def _get_surrounding_info(self) -> np.ndarray:
        p_current = self.current_point
        distances = []
        for direction in self.sensor_directions:
            distance = 0.0
            while distance <= self.sensor_range:
                check_point = p_current + direction * distance
                pe = self._query_potential(check_point)
                if pe < 0:
                    break
                distance += self.ray_step_size
            distances.append(distance)
        return np.array(distances)

    def _get_observation(self) -> np.ndarray:
        p_current = self.current_point

        # [新增] 计算当前管路末端的切向量 (Unit Tangent Vector)
        # Agent 需要知道自己当前的"朝向"，才能决定下一步怎么拐
        tangent_vec = p_current - self.previous_point
        norm = np.linalg.norm(tangent_vec)
        if norm < 1e-6:
            # 如果是初始状态或重合，默认使用起点法向，或者上一次的有效切向
            # 在 reset() 中，previous_point 初始化为 P_0, current 为 P_1，所以初始切向就是 N_s
            current_tangent = self.N_s
        else:
            current_tangent = tangent_vec / norm

        vec_to_target = self.P_t_minus_2 - p_current
        surrounding_info = self._get_surrounding_info()

        # [修改] 拼接状态向量
        return np.concatenate([
            p_current,
            vec_to_target,
            current_tangent,  # <--- 加入切向量
            surrounding_info
        ])

    def _calculate_reward(self, p_new: np.ndarray, w_new: float) -> float:
        """
        计算每一步的过程奖励 (R1, R2, R3, R4, R_kappa, R_tau)
        注意：R5 不在这里，因为它只在到达终点时触发。
        """
        p_prev = self.previous_point
        all_points = self.control_points + [p_new.tolist()]
        all_weights = self.control_weights + [w_new]

        if len(all_points) < 4:
            dist_old = np.linalg.norm(self.P_t_minus_2 - p_prev)
            dist_new = np.linalg.norm(self.P_t_minus_2 - p_new)
            R1 = dist_old - dist_new
            return self.reward_weights['R1_dist'] * R1

        try:
            curve = Geomdl_NURBS(all_points, all_weights, degree=3)
        except Exception as e:
            print(f"NURBS 曲线创建失败: {e}")
            return -100.0

        # R1
        dist_old = np.linalg.norm(self.P_t_minus_2 - p_prev)
        dist_new = np.linalg.norm(self.P_t_minus_2 - p_new)
        R1 = dist_old - dist_new

        # R2
        new_total_length = curve.length
        segment_length = new_total_length - self.current_curve_length
        self.current_curve_length = new_total_length
        R2 = -segment_length

        # R3, R4, R_kappa, R_tau
        R3 = 0.0
        R4_potentials = []
        all_curvatures = []
        all_torsions = []

        if segment_length <= 1e-3:
            num_samples = 0
        else:
            num_samples = int(np.ceil(segment_length / self.sampling_interval_dl))

        if num_samples > 1:
            u_start, u_end = curve.get_new_segment_parameter_range()
            u_values = np.linspace(u_start, u_end, num_samples)

            # 1. 采样点坐标 (用于 R3, R4)
            sample_points = curve.sample_list(u_values)

            for i in range(len(sample_points)):
                pt = sample_points[i]
                u = u_values[i]

                # --- R3 和 R4 ---
                pe = self._query_potential(pt)
                if pe < 0:
                    R3 += pe
                else:
                    R4_potentials.append(pe)

                # --- R_kappa 和 R_tau (手动计算) ---
                if 0.0 < u < 1.0:
                    try:
                        # 获取直到3阶的导数 [C, C', C'', C''']
                        derivs = curve.get_derivatives(u, order=3)

                        k = _calculate_curvature(derivs)
                        t = _calculate_torsion(derivs)

                        all_curvatures.append(k)
                        all_torsions.append(np.abs(t))
                    except Exception:
                        pass

        if not R4_potentials:
            R4 = 0.0
        else:
            R4 = np.mean(R4_potentials) - self.max_potential

        R_kappa = 0.0
        R_tau = 0.0

        if all_curvatures:
            R_kappa = -np.mean(all_curvatures)
        if all_torsions:
            R_tau = -np.mean(all_torsions)


        R_align = 0.0
        dist_to_target_zone = np.linalg.norm(self.P_t_minus_2 - p_new)

        if dist_to_target_zone < self.guidance_dist:
            # 1. 计算这一步的行进方向 (Agent 当前生成的切向)
            step_vec = p_new - self.current_point
            step_norm = np.linalg.norm(step_vec)

            if step_norm > 1e-6:
                current_heading = step_vec / step_norm

                # 2. 目标的理想进入方向
                # 注意：self.N_t 是目标处的法向量。
                # 管路是连接到 P_t_minus_2 -> P_t_minus_1 -> P_t
                # 这一段的方向是 (P_t - P_t_minus_2)，方向约为 -N_t
                ideal_heading = -self.N_t

                # 3. 计算余弦相似度 (Dot Product)
                # 值域 [-1, 1]。1 表示完美平行，-1 表示反向，0 表示垂直
                alignment_score = np.dot(current_heading, ideal_heading)

                # 4. 只有当方向大致正确(>0)时才给奖励，或者直接给线性奖励
                # 建议使用 max(0, score) 避免惩罚它的探索，只奖励正确的行为
                R_align = max(0.0, alignment_score)

                # [修改] 总奖励公式
        total_reward = (
                        self.reward_weights['R1_dist'] * R1 +
                        self.reward_weights['R2_len'] * R2 +
                        self.reward_weights['R3_obs'] * R3 +
                        self.reward_weights['R4_pe'] * R4 +
                        self.reward_weights['R_kappa'] * R_kappa +
                        self.reward_weights['R_tau'] * R_tau +
                        self.reward_weights['R_align'] * R_align  # <--- 加入这一项
                )

        return total_reward

    def reset(self):
        self.current_step = 0
        self.control_points = [self.P_s.tolist(), self.P_0.tolist(), self.P_1.tolist()]
        self.control_weights = [1.0, 1.0, 1.0]
        self.current_curve_length = 0.0
        self.current_point = self.P_1
        self.previous_point = self.P_0
        return self._get_observation()

    def step(self, action: np.ndarray):
        self.current_step += 1
        delta_xyz = action[0:3] * self.action_step_size
        weight_new = 1.0 + action[3] * 0.5
        weight_new = max(0.1, weight_new)

        self.previous_point = self.current_point
        point_new = self.current_point + delta_xyz

        reward = self._calculate_reward(point_new, weight_new)

        self.current_point = point_new
        self.control_points.append(point_new.tolist())
        self.control_weights.append(weight_new)

        done = False
        dist_to_target = np.linalg.norm(self.P_t_minus_2 - self.current_point)

        # 检查是否到达目标
        if dist_to_target <= self.d_max:
            done = True
            reward += self.reward_weights['R5_success']

            # 添加夹持点 (这是为了生成最终完整曲线)
            self.control_points.extend([self.P_t_minus_2.tolist(), self.P_t_minus_1.tolist(), self.P_t.tolist()])
            self.control_weights.extend([1.0, 1.0, 1.0])

            # --- [!! 新增 !!] 终端曲率/挠率检查 ---
            # 既然任务已完成，我们现在拥有了完整的控制点列表。
            # 我们可以构建整条曲线，并专门检查最后这一段（连接到终点的部分）是否平滑。
            try:
                # 1. 构建包含夹持点的完整曲线
                full_curve = Geomdl_NURBS(self.control_points, self.control_weights, degree=3)

                # 2. 采样最后一段区域 (例如 u 从 0.9 到 1.0)
                # 我们只关心最后这部分是否为了"硬凑"终点而产生了剧烈弯曲
                u_values = np.linspace(0.9, 0.99, 20)  # 避开 u=1.0 防止导数不稳定

                terminal_kappas = []
                terminal_taus = []

                for u in u_values:
                    # 获取导数
                    derivs = full_curve.get_derivatives(u, order=3)
                    # 计算曲率和挠率
                    k = _calculate_curvature(derivs)
                    t = _calculate_torsion(derivs)

                    terminal_kappas.append(k)
                    terminal_taus.append(np.abs(t))

                # 3. 计算终端平均值
                avg_terminal_kappa = np.mean(terminal_kappas)
                avg_terminal_tau = np.mean(terminal_taus)

                # 4. 施加额外的惩罚
                # 如果终端曲率过大，说明是一个"急转弯"进站
                # 我们可以给一个比平时更大的权重，甚至抵消掉一部分 R5 奖励

                # 定义终端惩罚权重 (建议比过程权重 R_kappa 更大，例如 10 倍)
                w_terminal_kappa = self.reward_weights.get('R_kappa', 2.0) * 10.0
                w_terminal_tau = self.reward_weights.get('R_tau', 1.0) * 10.0

                terminal_penalty = - (w_terminal_kappa * avg_terminal_kappa + w_terminal_tau * avg_terminal_tau)

                # 将惩罚加到当前的 reward 中
                reward += terminal_penalty

                # (可选) 打印调试信息，看看惩罚有多大
                # print(f"终端检查: Kappa={avg_terminal_kappa:.4f}, Penalty={terminal_penalty:.2f}")

            except Exception as e:
                # 如果构建失败（极少见），忽略
                pass
            # --- [!! 新增结束 !!] ---

        if self.current_step >= self.max_steps_per_episode:
            done = True

        obs = self._get_observation()
        info = {}

        return obs, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"  Current Point: {self.current_point.tolist()}")
            print(f"  Control Points: {len(self.control_points)}")

    def close(self):
        pass


# --- (可选) 环境测试 ---
if __name__ == "__main__":
    pe_map_file = "/home/ljh/PycharmProjects/octree_test1025/out_octree_pe/assembly7_pe_leaflevel.npy"
    meta_file = "/home/ljh/PycharmProjects/octree_test1025/out_octree_pe/assembly7_pe_meta.json"

    if not (os.path.exists(pe_map_file) and os.path.exists(meta_file)):
        print(f"错误：找不到环境文件。")
    else:
        print("--- 环境测试 ---")
        # (这里的坐标只是测试用的，训练时会用 train.py 里的坐标)
        start_pt = [-302.43, 360.42, -893.05]
        start_n = [-1, 1.19, 0]
        target_pt = [302.43, 360.42, -893.05]
        target_n = [1, 1.19, 0]

        env = PipeRoutingEnv(
            pe_map_path=pe_map_file,
            meta_path=meta_file,
            start_point=start_pt,
            start_normal=start_n,
            target_point=target_pt,
            target_normal=target_n,
            pipe_diameter=10.0
        )

        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"启动测试... 最大步数: {env.max_steps_per_episode}")

        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            print(f"Step {step_count}: Reward = {reward:.2f}")

        print("--- 回合结束 ---")
        print(f"总步数: {step_count}")
        print(f"总奖励: {total_reward:.2f}")
        print(f"最终控制点数量: {len(env.control_points)}")
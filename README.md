# Mujoco Playground - Trico 灵巧手强化学习项目

## 📋 项目概述

本项目使用 MuJoCo 物理引擎和 JAX 框架，对 Trico 灵巧手进行强化学习训练。目标是通过 PPO 算法训练一个策略，使灵巧手能够执行物体操纵任务（推动物体到指定方向）。

**项目状态**：🔄 持续改进中（最后更新：2026年2月2日）

---

## 🎯 核心任务

### 当前任务：物体推动（+X/+Y 方向）

灵巧手通过两个指尖接触并推动物体，使其在世界坐标系中沿 +X 和 +Y 方向移动，目标是最大化物体位移 (X + Y)。

**物理约束**：
- 物体初始位置：x, y ∈ [-0.01, 0.01] m，z = 0.029 m（地表）
- 估计推动极限：~70 mm（0.07 m）
- 灵巧手控制频率：50 Hz（ctrl_dt = 0.02 s）
- 物理仿真步长：1 ms（sim_dt = 0.001 s）

---

## 🔧 环境设置

### 依赖项

```bash
conda create -n mjx_env python=3.10
conda activate mjx_env
pip install jax jaxlib mujoco brax ml-collections tensorboard wandb
```

### 核心文件详解

#### 1. **trico.py** - 环境定义（主要修改文件）
📍 路径：`mujoco_playground/_src/manipulation/trico/trico.py`

| 功能 | 代码位置 | 说明 |
|------|---------|------|
| 初始化参数 | L1-45 | 加载 XML 模型，设置求解器参数（CG 迭代50次）|
| 观测设计 | L90-130 | 生成 27 维观测（关节8 + 力6 + 物体位姿7 + 相对位置6）|
| **奖励函数** | **L145-175** | ⭐ **核心修改点** - 4 项奖励 + 2 项惩罚 |
| 重置函数 | L180-210 | 初始化环境，设置 11 个指标跟踪 |

**奖励函数详细（L145-175）：**

```python
# move_reward: 仅奖励正向位移 (L155-156)
move_reward = (jnp.maximum(0.0, obj_pos[0]) + jnp.maximum(0.0, obj_pos[1])) * 10.0

# vel_reward: 仅奖励正向速度 (L157-158)
vel_reward = (jnp.maximum(0.0, obj_vel_linear[0]) + jnp.maximum(0.0, obj_vel_linear[1])) * 1.0

# neg_penalty: 惩罚负向位移 (L159-160) ⭐ 新增
neg_penalty = 5.0 * (jnp.minimum(0.0, obj_pos[0]) + jnp.minimum(0.0, obj_pos[1]))

# 最终奖励 (L161)
reward = reach_reward + move_reward + vel_reward + survival_reward + neg_penalty - action_cost
```

#### 2. **train_jax_ppo.py** - 训练脚本
📍 路径：`learning/train_jax_ppo.py`

| 功能 | 代码位置 | 说明 |
|------|---------|------|
| 环境加载 | L100-150 | 注册并加载 `Trico` 环境 |
| 训练参数 | L150-200 | PPO 超参数（学习率、熵成本、步数） |
| TensorBoard 日志 | L300-350 | 记录训练曲线和评估指标 |
| 模型保存 | L400-450 | 定期保存检查点 |

**关键命令行参数：**
```bash
--env_name=Trico              # 使用 Trico 环境
--num_timesteps=500000000     # 总训练步数
--num_envs=2048               # 并行环境数
--learning_rate=0.0005        # PPO 学习率
--entropy_cost=0.05           # 熵正则化系数
--episode_length=500          # 单个 episode 长度
--use_tb=True                 # 启用 TensorBoard
```

#### 3. **trico_hand_for_rl.xml** - 物理模型定义
📍 路径：`mujoco_playground/_src/manipulation/trico/xmls/trico_hand_for_rl.xml`

| 部分 | 说明 |
|------|------|
| `<body>` | 灵巧手的 8 个关节和物体定义 |
| `<actuator>` | 8 个电机控制（主动关节） |
| `<sensor>` | 6 个力传感器（左右指各 3 轴） |
| `<site>` | 指尖位置标记，用于计算相对位置 |

#### 4. **registry.py** - 环境注册表
📍 路径：`mujoco_playground/_src/registry.py`

需要在此文件中注册 `Trico` 环境，使得训练脚本能找到它：

```python
# 在 registry.py 中应该有类似代码：
register(
    id='Trico-v0',
    entry_point='mujoco_playground._src.manipulation.trico:TricoEnv',
    max_episode_steps=500,
)
```

### 项目结构概览

```
mujoco_playground/
├── _src/
│   ├── manipulation/
│   │   └── trico/
│   │       ├── trico.py                    # ⭐ 主要修改：观测 + 奖励
│   │       ├── trico_推动XY尽可能远.py    # 旧版本参考（已验证的奖励）
│   │       ├── trico_老的中心转.py         # 历史版本（抬升任务）
│   │       └── xmls/
│   │           └── trico_hand_for_rl.xml  # 物理模型定义
│   ├── inference/                          # 🌉 Sim2Real 转换核心
│   │   ├── sim2real_bridge.py              # ⭐ RealRobotBridge 类（转换入口）
│   │   ├── hrz_test.py                     # 往返精度测试
│   │   ├── hrz_test_chazhi.py              # 插值鲁棒性测试
│   │   └── README.md                       # 模块说明
│   ├── mjx_env.py                          # 基础环境类（不需要改）
│   ├── wrapper.py                          # Brax 包装器（不需要改）
│   └── registry.py                         # 环境注册（需要注册 Trico）
├── test_sim2real/                          # 🧪 单个CSPM模块测试
│   ├── debug_cspm_ik.py                    # IK 求解调试
│   ├── debug_real2sim_pipeline.py          # 观测管道追踪
│   ├── oa_interactive.py                   # 交互式调试
│   ├── oa_right_simple.xml                 # 简化场景
│   └── README.md                           # 测试说明
├── doc/                                    # 📖 中文文档
│   ├── SIM2REAL转换使用指南.md             # API + 集成示例
│   ├── Sim2Real工程记录.md                 # 技术背景 + 问题记录
│   └── info.py                             # 其他信息
├── learning/
│   ├── train_jax_ppo.py                    # PPO 训练脚本（命令行参数在此配置）
│   ├── train_jax_ppo_origin.py             # 原始版本备份
│   └── notebooks/                          # Jupyter 示例
├── logs/                                   # 训练日志输出目录
│   └── {experiment_id}/
│       ├── events.out.tfevents.*           # TensorBoard 数据
│       └── checkpoints/                    # 模型检查点
├── scripts/                                # 辅助脚本
├── README.md                               # 本文件
├── BRIDGE_USAGE.md                         # (已弃用，参看 doc/ 中的新文档)
└── LOOK_CONTEXT_MEMORY.md                  # (已弃用，参看 doc/ 中的新文档)
```

**关键目录说明**：
- `_src/inference/` - RL 与真机通信的**唯一通道** （往返精度 0.001°）
- `test_sim2real/` - 低层 CSPM 模块调试，用于验证转换正确性
- `doc/` - 中文技术文档（**优先阅读**）

---

## 🚀 快速开始

### 1. 训练模型

```bash
cd /home/rune/proj_rune/mujoco_playground && \
CUDA_VISIBLE_DEVICES=0,1 python learning/train_jax_ppo.py \
  --env_name=Trico \
  --num_timesteps=500000000 \
  --num_envs=2048 \
  --num_evals=5 \
  --reward_scaling=1.0 \
  --episode_length=500 \
  --learning_rate=0.0005 \
  --entropy_cost=0.05 \
  --use_tb=True
```

**训练参数说明**：
- `num_timesteps`: 总训练步数（5 亿步约 100 小时）
- `num_envs`: 并行环境数（2048 个）
- `episode_length`: 单个 episode 长度（500 步 = 10 秒）
- `use_tb`: 启用 TensorBoard 日志

### 2. 监控训练

```bash
tensorboard --logdir /home/hurz/code/Mujoco-playground-Trico/logs --port 6006
```

访问 `http://localhost:6006` 查看实时曲线。

### 3. 生成视频

```bash
cd /home/hurz/code/Mujoco-playground-Trico && \
CUDA_VISIBLE_DEVICES=0,1 python learning/train_jax_ppo.py \
  --env_name=Trico \
  --play_only=True \
  --load_checkpoint_path=logs/YOUR_EXPERIMENT/checkpoints \
  --num_videos=1 \
  --episode_length=500 \
  --use_tb=False
```

生成的视频保存为 `rollout0.mp4`。

---

## 📊 观测空间与奖励函数

### 观测空间（27 维）

| 维度 | 内容 | 说明 |
|------|------|------|
| 8 | 主动关节角度 | 8 个电机控制的关节位置 |
| 6 | 力传感器 | 左右指尖的 3D 接触力（EMA 滤波） |
| 7 | 物体位姿 | 位置 (x, y, z) + 四元数 (w, x, y, z) |
| 6 | 相对位置 | 指尖-物体距离（右指 3D + 左指 3D） |

### 奖励函数（当前版本 v2.0）

```python
总奖励 = reach_reward + move_reward + vel_reward + survival_reward + neg_penalty - ctrl_cost
```

| 项 | 公式 | 说明 |
|------|------|------|
| **reach_reward** | $1.0 - \tanh(5.0 \times d_{avg})$ | 双手接近物体 |
| **move_reward** | $(\max(0, x) + \max(0, y)) \times 10.0$ | ⭐ 核心：最大化正向位移 |
| **vel_reward** | $(\max(0, v_x) + \max(0, v_y)) \times 1.0$ | 仅奖励正向速度 |
| **survival_reward** | $1.0 \times [0.01 < z < 0.2]$ | 物体保持在桌面 |
| **neg_penalty** | $5.0 \times (\min(0, x) + \min(0, y))$ | 惩罚负向位移 |
| **ctrl_cost** | $-0.001 \times \sum action^2$ | 动作能耗 |

---

## 📈 训练历史与改进

### 版本 v1.0（2026/01/31）- 抬升任务
**失败原因**：
- 观测不含物体/指尖位姿 → 隐变量奖励
- 里程碑奖励容易陷入"接触但不推"的局部最优
- **最终结果**：奖励停留在 150 左右，无明显进展

**代码位置**：`trico_老的中心转.py`

### 版本 v1.5（2026/02/01 375M步）- +X 单轴推动（一进制方向）
**改进**：
- 加入观测中的物体位姿 + 指尖相对位置（27维）
- 改为单轴推动 +X 方向，里程碑：1cm → 3cm → 5cm
- **最终结果**：奖励 ~789，**但方向错误**（推向 -X！）

**根本原因**：
- 奖励函数对称性：`move_reward = (obj_x) * 10.0` 对正负方向等价
- 物理可能性：灵巧手向 -X 方向推可能更容易（手指布局、关节可达性）
- 生存奖励掩盖：即使往错方向推，接近 + 生存奖励仍给出正总奖励 (~2.0/步)

**结论**：策略学到了"往最小阻力方向"而非"往指定方向"

### 版本 v2.0（2026/02/01 进行中）- +X+Y 双轴推动（带方向约束）
**关键改进**：
- 恢复旧版本的**连续奖励设计**（而非离散里程碑）
- 观测保持 27 维（含物体位姿+相对位置+力传感）
- **加入正向约束**：
  - `move_reward = (max(0, x) + max(0, y)) * 10.0`（仅奖励正向）
  - `neg_penalty = 5.0 * (min(0, x) + min(0, y))`（惩罚负向）
- 速度也仅奖励正向：`vel_reward = (max(0, vx) + max(0, vy)) * 1.0`

**预期结果**：
- eval/episode_obj_xy_sum: -0.28 → **+0.05 ~ +0.15**（正向）
- eval/episode_reward: 789 → **预计 800-900**（略微上升）
- **训练进度**：正在进行（目标 5 亿步，~100 小时）

---

## ✏️ 如何修改各部分功能

### 1. 修改奖励函数

**文件**：`trico.py` → L145-175（`step()` 方法中）

**当前代码示例**：
```python
# 计算各项奖励
reach_reward = 1.0 - jnp.tanh(5.0 * avg_distance_to_obj)
move_reward = (jnp.maximum(0.0, obj_pos[0]) + jnp.maximum(0.0, obj_pos[1])) * 10.0
vel_reward = (jnp.maximum(0.0, obj_vel_linear[0]) + jnp.maximum(0.0, obj_vel_linear[1])) * 1.0
neg_penalty = 5.0 * (jnp.minimum(0.0, obj_pos[0]) + jnp.minimum(0.0, obj_pos[1]))
survival_reward = 1.0 * ((obj_z > 0.01) & (obj_z < 0.2)).astype(jnp.float32)
action_cost = 0.001 * jnp.sum(action**2)

# 组合奖励
reward = reach_reward + move_reward + vel_reward + survival_reward + neg_penalty - action_cost
```

**如果要调整系数**：
- 增加 `move_reward` 的权重：改 `* 10.0` 为 `* 15.0`
- 加强负向惩罚：改 `* 5.0` 为 `* 8.0`
- 减弱存活奖励：改 `1.0 *` 为 `0.5 *`

**如果要改奖励逻辑**（例如加入接触力约束）：
```python
# 原始版本（当前）
move_reward = (jnp.maximum(0.0, obj_pos[0]) + jnp.maximum(0.0, obj_pos[1])) * 10.0

# 新版本：仅当有接触力时才奖励推动
force_magnitude = jnp.sqrt(jnp.sum(force_right**2 + force_left**2))
has_contact = force_magnitude > 0.1  # 接触力阈值
move_reward = has_contact * (jnp.maximum(0.0, obj_pos[0]) + jnp.maximum(0.0, obj_pos[1])) * 10.0
```

### 2. 修改观测空间

**文件**：`trico.py` → L90-130（`_get_obs()` 方法）

**当前观测（27维）**：
```python
# L95-105
obs_list = [
    state.qpos,                      # [0:8]   关节角度
    filtered_force_right,            # [8:11]  右指力
    filtered_force_left,             # [11:14] 左指力
    obj_pos,                         # [14:17] 物体位置
    obj_rot_quat,                    # [17:21] 物体旋转（四元数）
    relative_pos_right,              # [21:24] 右指-物体相对位置
    relative_pos_left,               # [24:27] 左指-物体相对位置
]
obs = jnp.concatenate(obs_list)
```

**如果要添加物体速度（变成 33 维）**：
```python
obs_list = [
    state.qpos,                      # [0:8]   关节角度
    filtered_force_right,            # [8:11]  右指力
    filtered_force_left,             # [11:14] 左指力
    obj_pos,                         # [14:17] 物体位置
    obj_rot_quat,                    # [17:21] 物体旋转
    obj_vel_linear,                  # [21:24] ✨ NEW：物体线性速度
    obj_vel_angular,                 # [24:27] ✨ NEW：物体角速度
    relative_pos_right,              # [27:30] 右指-物体相对位置
    relative_pos_left,               # [30:33] 左指-物体相对位置
]
obs = jnp.concatenate(obs_list)
```

### 3. 修改控制频率和物理参数

**文件**：`trico.py` → L25-35（`__init__` 方法）

```python
# 当前设置
config.ctrl_dt = 0.02   # 50Hz 控制频率（20ms）
config.sim_dt = 0.001   # 1ms 物理仿真步长
config.episode_length = 500  # 500 * 20ms = 10 秒

# 改为 100Hz 控制（更高精度）
config.ctrl_dt = 0.01   # 100Hz（10ms）
config.sim_dt = 0.001   # 保持 1ms（保持物理准确性）
config.episode_length = 1000  # 1000 * 10ms = 10 秒

# 或者改为 25Hz 控制（更低延迟）
config.ctrl_dt = 0.04   # 25Hz（40ms）
```

### 4. 修改训练超参数

**文件**：`train_jax_ppo.py` → 命令行参数

```bash
# 当前设置
python learning/train_jax_ppo.py \
  --env_name=Trico \
  --num_timesteps=500000000 \     # 500M 步
  --num_envs=2048 \               # 2048 并行环境
  --learning_rate=0.0005 \        # PPO 学习率
  --entropy_cost=0.05 \           # 熵正则系数
  --episode_length=500

# 改为：更快收敛（较少并行，更多训练步数）
python learning/train_jax_ppo.py \
  --env_name=Trico \
  --num_timesteps=1000000000 \    # 10 亿步（2 倍）
  --num_envs=4096 \               # 4096 并行环境（2 倍）
  --learning_rate=0.001 \         # 提高学习率（2 倍）
  --entropy_cost=0.02 \           # 降低熵成本（更确定的策略）
  --episode_length=500
```

### 5. 修改指标跟踪

**文件**：`trico.py` → L180-210（`reset()` 和 `step()` 方法的 metrics）

**当前跟踪的 11 个指标**：
```python
metrics = {
    'obj_height': obj_z,                    # 物体高度
    'obj_x': obj_pos[0],                    # X 位移
    'obj_y': obj_pos[1],                    # Y 位移
    'obj_xy_sum': obj_pos[0] + obj_pos[1],  # X+Y 总位移
    'reward_reach': reach_reward,           # 各项奖励
    'reward_move': move_reward,
    'reward_vel': vel_reward,
    'reward_survival': survival_reward,
    'reward_neg_penalty': neg_penalty,
    'total_reward': reward,
    'reward': reward,                       # 副本（Brax 要求）
}
```

**如果要增加新指标**（如接触力大小）：
```python
force_magnitude = jnp.sqrt(jnp.sum(force_right**2 + force_left**2))
metrics['force_magnitude'] = force_magnitude  # ✨ 新指标
metrics['obj_distance_to_origin'] = jnp.sqrt(obj_pos[0]**2 + obj_pos[1]**2)  # ✨ 新指标
```

---

## 📊 训练监控关键指标解释

### TensorBoard 中应该看到的曲线

**训练进行中** → 监控这些指标确认训练朝向正确方向：

| 指标 | 预期趋势 | 说明 |
|------|---------|------|
| `eval/episode_reward` | ↗️ 上升至 800-900 | 总奖励，反映策略整体质量 |
| `eval/episode_obj_xy_sum` | ↗️ 从 -0.28 → +0.05~+0.15 | **最关键**：物体应向 +X/+Y 推 |
| `eval/episode_obj_x` | ↗️ 正数且上升 | X 轴位移应为正 |
| `eval/episode_obj_y` | ↗️ 正数且上升 | Y 轴位移应为正 |
| `eval/episode_reward_neg_penalty` | ↘️ 下降（趋向 0） | 负向惩罚，应逐渐消失 |
| `loss/policy_loss` | ↘️ 下降（收敛） | PPO 策略梯度损失 |
| `loss/value_loss` | ↘️ 下降（收敛） | PPO 价值函数损失 |

**异常情况**：
- ⚠️ `eval/episode_obj_xy_sum` 保持负数 → 奖励函数有问题，检查 L155-160
- ⚠️ `eval/episode_reward` 不上升 → 学习率太低或观测设计问题
- ⚠️ `loss/*` 不收敛 → 超参数需要调整



### 当前关键点

1. **观测设计** ✅
   - 27 维：关节(8) + 力(6) + 物体位姿(7) + 相对位置(6)
   - 考虑加入物体速度(6维)使用更复杂的策略网络

2. **奖励函数** 🔄 **正在优化**
   - v2.0 加入了正向约束
   - 可监控 TensorBoard 中的 `eval/reward_neg_penalty` 观察是否生效

3. **物理参数** ⚠️
   - 求解器：CG，迭代数 50，力求稳定但不过度
   - 控制频率 50Hz，仿真 1ms，action_repeat=1（即时响应）

### 下一步任务清单

- [ ] 验证 v2.0 训练是否推向 +X/+Y
- [ ] 如果仍出现方向问题，考虑加入"视觉中心定位"（object在屏幕中心时额外奖励）
- [ ] 评估最终策略在真实硬件上的可部署性（位姿估计延迟、力传感器噪声）
- [ ] 建立 sim2real 域随机化管道

### 常见问题

**Q: 为什么观测不包含物体速度？**  
A: 为了降低维度和部署复杂度。策略网络可通过 RNN 从位置序列自动推断速度。如果需要，可改成 33 维。

**Q: 奖励的 10.0 和 5.0 系数从哪来？**  
A: 经验值。基于旧版本（推XY任意方向）的结果调整。可在实验中扫参。

**Q: 为什么没有接触力的下界检查？**  
A: 因为生存奖励已经隐含地鼓励"保持物体在桌面"。如果需要硬约束，可加 `contact_penalty`。

---

## � Sim-to-Real 部署

### 快速开始

Sim2Real 转换的**唯一入口**：

```python
from mujoco_playground._src.inference.sim2real_bridge import RealRobotBridge

# 初始化
bridge = RealRobotBridge()

# 观测转换：真机舵机角 → 仿真Hinge角
servo_pos = np.array([260, 170, 170, 260, 260, 170, 170, 260])  # 度数
hinge_obs = bridge.servo_angles_to_hinge_angles(servo_pos)  # 弧度

# 控制转换：仿真Hinge角 → 真机舵机角
hinge_action = np.array([0.1, 0.05, 0.1, 0.05, 0, 0, 0, 0])  # 弧度
servo_cmd = bridge.hinge_angles_to_servo_angles(hinge_action, apply_bank_ab=False)  # 度数
```

### 文档导航

| 文档 | 用途 | 对象 |
|------|------|------|
| 📖 [转换使用指南](doc/SIM2REAL转换使用指南.md) | API 详解、集成示例、故障排查 | RL 工程师 |
| 📖 [工程记录](doc/Sim2Real工程记录.md) | 技术背景、问题记录、设计原理 | 系统维护者 |
| 📁 [inference/](mujoco_playground/_src/inference/) | 核心转换模块（`sim2real_bridge.py`）| 开发者 |
| 📁 [test_sim2real/](test_sim2real/) | 单个CSPM测试脚本 | 调试人员 |

### 核心特性

- ✅ **往返精度** < 0.001°（真机舵机 → Hinge → 舵机）
- ✅ **双向转换** Real2Sim (观测) + Sim2Real (控制)
- ✅ **坐标系统一** VirtualHingeMapper 自动处理 OA/OB 偏置
- ✅ **IK 连续性** 避免启动时的大跳变
- ✅ **8D 支持** 双手同时转换
- ✅ **Bank A/B 解耦** 转换层禁用，硬件层启用

### 验证状态

- [x] 单个CSPM的往返转换 (0.000° 误差)
- [x] 插值策略鲁棒性 (5/10/20/50步均0.00°)
- [x] 双手独立转换
- [ ] 8D 完整转换验证脚本（待完成）
- [ ] 边界角度与安全约束测试（待完成）

---

## 📚 参考资源

- **Sim-to-Real部署指南**: [REAL_ROBOT_DEPLOYMENT_GUIDE.md](REAL_ROBOT_DEPLOYMENT_GUIDE.md) ⭐
- **MuJoCo 官方文档**：https://mujoco.readthedocs.io/
- **Brax PPO 实现**：https://github.com/google/brax
- **JAX 教程**：https://jax.readthedocs.io/

---

## 📝 最后更新日志

- **2026-03-06**：合并所有部署文档为单一综合指南 (REAL_ROBOT_DEPLOYMENT_GUIDE.md)
- **2026-02-02**：新增 v2.0 奖励函数（正向约束）
- **2026-02-01**：发现 v1.5 方向错误，分析并修复
- **2026-01-31**：初始版本 v1.0，尝试抬升任务（失败）

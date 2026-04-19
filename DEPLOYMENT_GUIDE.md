# Sim2Real 部署完整指南 - 检查点 000128450560

## 📋 概览

本指南详细说明如何使用训练好的 PPO 策略在真实 Trico 机器人上部署。检查点 `000128450560` 已在 128M+ 步的训练中收敛。

**关键参数摘要：**
- **观测维数：** 668 维（不可变）
- **动作维数：** 4 维，范围 [-1, 1]
- **控制频率：** 50 Hz（ctrl_dt=0.02s）
- **归一化：** 已启用（必须使用）
- **网络架构：** 3×256→4-dim tanh_normal

---

## 🔧 第 1 步：加载检查点和归一化参数

### 1.1 检查点文件结构

```
checkpoints/000128450560/
├── ppo_network_config.json      # 网络配置，含normalize_observations=true
├── config.json                  # 环境参数（ctrl_dt, obs_history_len等）
├── _METADATA                    # Orbax 元数据，包含 mean[668] 和 std[668]
├── d/                           # Orbax 数据目录
├── ocdbt.process_0/             # OCDBT 格式检查点分片
└── manifest.ocdbt
```

### 1.2 加载 JAX 检查点（Python/JAX 环境）

```python
import jax
import jax.numpy as jp
from pathlib import Path
from brax.training.agents.ppo import networks as ppo_networks
import orbax.checkpoint
import pickle

# 检查点路径
ckpt_path = Path(
    "logs/TricoDriverSingle-contine_3_10M_near_dr-20260409-201011-trico_driver_v3"
    "/checkpoints/000128450560"
)

# 加载 Orbax 检查点
checkpointer = orbax.checkpoint.OrbaxCheckpointer()
abstract_state = {
    "params": {
        "0": {
            "mean": jax.ShapeDtypeStruct(shape=(668,), dtype=jp.float32),
            "std": jax.ShapeDtypeStruct(shape=(668,), dtype=jp.float32),
            # ... 其他参数 ...
        }
    }
}

params = checkpointer.restore(str(ckpt_path), abstract_state)

# 提取归一化统计数据
norm_mean = np.array(params["0"]["mean"])       # shape=(668,)
norm_std = np.array(params["0"]["std"])         # shape=(668,)
norm_std_eps = 1e-8  # 标准的 epsilon 值，防止除零
```

### 1.3 提取策略网络

```python
# 从 ppo_network_config.json 读取网络配置
import json

config_path = ckpt_path / "ppo_network_config.json"
with open(config_path, 'r') as f:
    net_config = json.load(f)

# 关键配置值
action_size = net_config["action_size"]  # = 4
obs_size = 668  # net_config["observation_size"]
normalize_obs = net_config["normalize_observations"]  # = true

# 网络参数
kwargs = net_config["network_factory_kwargs"]
policy_hidden_sizes = kwargs["policy_hidden_layer_sizes"]  # [256, 256, 256]
activation = kwargs["activation"]  # "silu"
distribution_type = kwargs["distribution_type"]  # "tanh_normal"

print(f"✓ 网络配置加载完毕: {obs_size}D → {policy_hidden_sizes} → {action_size}D")
```

### 1.4 加载环境参数

```python
env_config_path = ckpt_path / "config.json"
with open(env_config_path, 'r') as f:
    env_config = json.load(f)

# 关键参数
ctrl_dt = env_config["ctrl_dt"]  # = 0.02 秒
obs_history_len = env_config["obs_history_len"]  # = 30
contact_threshold = env_config["contact_force_threshold"]  # = 0.1
episode_length = env_config["episode_length"]  # = 500（仿真中的步数）
sim_dt = env_config["sim_dt"]  # = 0.001 秒

print(f"✓ 环境配置: 控制频率={1/ctrl_dt}Hz, 历史长度={obs_history_len}")
```

---

## 📊 第 2 步：观测构建管道（Observation Pipeline）

### 2.1 单帧观测结构（22 维）

```
单帧观测 (这个会被重复30次以形成660维历史)

索引范围  维数  内容                     来源
0-7      8    关节角度 qpos[0-7]       8 个致动器位置
8-9      2    接触标志                 left_contact, right_contact (bool)
10-12    3    右手尖位置 xyz            site_xpos[tip_right_id]
13-15    3    左手尖位置 xyz            site_xpos[tip_left_id]
16-18    3    相对向量 right (目标→尖)  handle_center - tip_right
19-21    3    相对向量 left (目标→尖)   handle_center - tip_left

× 30 帧历史 = 660 维
+ 8 维 离散速度 = 668 维总计
```

### 2.2 单帧观测数据收集

```python
def collect_single_frame_observation(
    real_robot_interface,
    sim_kinematics,
    ctrl_dt=0.02
) -> np.ndarray:
    """
    从真实机器人采集单帧 22 维观测。
    
    Args:
        real_robot_interface: 真机接口，提供关节反馈和传感器数据
        sim_kinematics: 仿真运动学模块（用于 IK 和位置计算）
        ctrl_dt: 控制周期（0.02秒）
    
    Returns:
        obs_frame: 22维单帧观测向量
    """
    obs_frame = np.zeros(22, dtype=np.float32)
    
    # [0-7] 关节角度：从真机读取 8 个致动器的当前位置
    # 舵机顺序（必须与仿真训练一致）:
    #   OA_x_right(索引0), OA_y_right(1), OB_x_right(2), OB_y_right(3),
    #   OA_x_left(4), OA_y_left(5), OB_x_left(6), OB_y_left(7)
    qpos = real_robot_interface.get_joint_angles()  # shape=(8,), 弧度
    obs_frame[0:8] = qpos
    
    # [8-9] 接触标志：计算力传感器是否超过阈值
    force_left = real_robot_interface.get_left_contact_force()  # 3D 向量
    force_right = real_robot_interface.get_right_contact_force()  # 3D 向量
    contact_threshold = 0.1  # N
    
    obs_frame[8] = 1.0 if np.linalg.norm(force_left) > contact_threshold else 0.0
    obs_frame[9] = 1.0 if np.linalg.norm(force_right) > contact_threshold else 0.0
    
    # [10-12] 右手尖端位置：使用正向运动学
    tip_right_pos = sim_kinematics.forward_right(qpos[0:4])  # xyz, shape=(3,)
    obs_frame[10:13] = tip_right_pos
    
    # [13-15] 左手尖端位置：使用正向运动学
    tip_left_pos = sim_kinematics.forward_left(qpos[4:8])  # xyz, shape=(3,)
    obs_frame[13:16] = tip_left_pos
    
    # [16-18] 右相对向量：从物体中心指向右尖端
    # (以仿真中的 object_center 为参考，可从真机相机或已知位置获取)
    object_center = real_robot_interface.get_object_center()  # xyz
    obs_frame[16:19] = object_center - tip_right_pos
    
    # [19-21] 左相对向量：从物体中心指向左尖端
    obs_frame[19:22] = object_center - tip_left_pos
    
    return obs_frame  # 22维
```

### 2.3 历史窗口维护（30 帧滑动窗口）

```python
class ObservationBuffer:
    """
    维护 30 帧的观测历史，形成 660 维的历史堆栈。
    
    设计：
    - 当前帧总是被堆在末尾（最新）
    - 新帧进入时，最老的帧被丢弃
    - 历史顺序：[帧0(最老), 帧1, ..., 帧29(最新)]
    """
    
    def __init__(self, history_len=30, frame_dim=22):
        """
        Args:
            history_len: 历史帧数（30）
            frame_dim: 每帧维数（22）
        """
        self.history_len = history_len
        self.frame_dim = frame_dim
        self.history = deque(maxlen=history_len)  # 自动维护长度
        self.qpos_buffer = deque(maxlen=2)  # 保存最后 2 个 qpos 用于速度计算
    
    def push(self, frame: np.ndarray) -> None:
        """
        添加新帧到历史缓冲区。自动丢弃最老帧。
        """
        if len(self.history) == 0:
            # 初始化：首次添加，用当前帧重复填充 30 个位置
            for _ in range(self.history_len):
                self.history.append(frame.copy())
        else:
            # 正常：新帧进入，old frame 自动被 deque 丢弃
            self.history.append(frame.copy())
    
    def push_qpos(self, qpos: np.ndarray) -> None:
        """用于速度计算的 qpos 缓冲。"""
        self.qpos_buffer.append(qpos.copy())
    
    def get_history_stack(self) -> np.ndarray:
        """
        获取 660 维的历史堆栈（30 帧 × 22 维）。
        顺序：[帧0维0-21, 帧1维0-21, ..., 帧29维0-21]
        """
        return np.concatenate(list(self.history), axis=0)  # shape=(660,)
    
    def get_discrete_velocity(self, ctrl_dt=0.02) -> np.ndarray:
        """
        计算离散速度：(qpos_curr - qpos_prev) / ctrl_dt
        
        关键点：
        - 速度是从 qpos 历史计算的，不使用直接的 qvel
        - ctrl_dt = 0.02 秒（50 Hz 控制）
        - 如果缓冲区中少于 2 个 qpos，返回零向量
        """
        if len(self.qpos_buffer) < 2:
            return np.zeros(8, dtype=np.float32)
        
        qpos_prev = self.qpos_buffer[0]
        qpos_curr = self.qpos_buffer[1]
        velocity = (qpos_curr - qpos_prev) / ctrl_dt
        return velocity  # shape=(8,)
    
    def reset(self) -> None:
        """重置缓冲区（在剧集或初始化时调用）。"""
        self.history.clear()
        self.qpos_buffer.clear()


# 使用示例
obs_buffer = ObservationBuffer(history_len=30, frame_dim=22)

# 每个控制循环：
def control_loop():
    while running:
        # 1. 从真机采集新帧
        frame = collect_single_frame_observation(...)
        qpos = real_robot_interface.get_joint_angles()
        
        # 2. 更新历史缓冲区
        obs_buffer.push(frame)
        obs_buffer.push_qpos(qpos)
        
        # 3. 构建完整的 668 维观测
        obs_history = obs_buffer.get_history_stack()  # 660维
        obs_velocity = obs_buffer.get_discrete_velocity(ctrl_dt=0.02)  # 8维
        obs_raw = np.concatenate([obs_history, obs_velocity], axis=0)  # 668维
        
        # 4. 进行策略推断（见第 3 步）
        ...
```

### 2.4 初始化流程

```python
def initialize_observation_buffer(
    real_robot_interface,
    sim_kinematics,
    history_len=30
) -> ObservationBuffer:
    """
    初始化观测缓冲区。在启动机器人或开始新剧集时调用。
    
    设计原则：
    - 首次读取的帧被重复填充 30 个位置（所有历史卡在当前状态）
    - 速度初始化为零（因为没有前一帧）
    - 经过 1-2 秒的运行后，缓冲区自然填充真实历史
    """
    obs_buffer = ObservationBuffer(history_len=history_len)
    
    # 读取初始状态
    initial_frame = collect_single_frame_observation(
        real_robot_interface, sim_kinematics
    )
    initial_qpos = real_robot_interface.get_joint_angles()
    
    # 填充缓冲区
    obs_buffer.push(initial_frame)  # deque 自动重复填充
    obs_buffer.push_qpos(initial_qpos)
    obs_buffer.push_qpos(initial_qpos)  # 两个相同的 qpos → velocity = 0
    
    return obs_buffer
```

---

## 🔐 第 3 步：观测归一化

### 3.1 归一化公式

```
归一化观测 = (原始观测 - 均值) / (标准差 + epsilon)

其中：
- 均值：shape=(668,)，来自检查点的 mean
- 标准差：shape=(668,)，来自检查点的 std
- epsilon：1e-8（数值稳定性）
- 所有操作都是逐元素的
```

### 3.2 应用归一化

```python
def normalize_observation(
    obs_raw: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    应用检查点中存储的运行统计数据进行观测归一化。
    
    Args:
        obs_raw: 原始 668 维观测
        norm_mean: 平均值 [668]，从检查点加载
        norm_std: 标准差 [668]，从检查点加载
        eps: 数值稳定性常数
    
    Returns:
        obs_normalized: 归一化后的 668 维观测
    """
    assert obs_raw.shape == (668,), f"期望形状 (668,)，得到 {obs_raw.shape}"
    assert norm_mean.shape == (668,), f"均值形状错误: {norm_mean.shape}"
    assert norm_std.shape == (668,), f"标差形状错误: {norm_std.shape}"
    
    obs_normalized = (obs_raw - norm_mean) / (norm_std + eps)
    return obs_normalized.astype(np.float32)


# 在主控制循环中使用：
def control_loop_with_normalization():
    # 初始化归一化参数（从检查点加载，见第 1 步）
    norm_mean = np.array(...)  # [668]
    norm_std = np.array(...)   # [668]
    
    obs_buffer = initialize_observation_buffer(...)
    
    while running:
        # 1. 采集新帧并更新缓冲区
        frame = collect_single_frame_observation(...)
        qpos = real_robot_interface.get_joint_angles()
        obs_buffer.push(frame)
        obs_buffer.push_qpos(qpos)
        
        # 2. 构建原始观测 [668]
        obs_history = obs_buffer.get_history_stack()  # [660]
        obs_velocity = obs_buffer.get_discrete_velocity()  # [8]
        obs_raw = np.concatenate([obs_history, obs_velocity])  # [668]
        
        # 3. **关键步骤**：应用归一化
        obs_normalized = normalize_observation(
            obs_raw,
            norm_mean,
            norm_std,
            eps=1e-8
        )
        
        # 4. 进行策略推断（见第 4 步）
        action = policy_inference(obs_normalized)
        
        # 5. 执行动作
        ...
```

---

## 🧠 第 4 步：策略推断

### 4.1 策略网络架构

```
输入：668 维（归一化观测）
      ↓
隐藏层 1: 256 个神经元, SiLU 激活
      ↓
隐藏层 2: 256 个神经元, SiLU 激活
      ↓
隐藏层 3: 256 个神经元, SiLU 激活
      ↓
输出层-策略: 8 个神经元 (mu, sigma for 4-dim action)
输出层-价值: 1 个神经元 (state value)

分布：Tanh Normal
- μ 和 σ → 参数化高斯分布
- tanh 限制输出到 [-1, 1]
```

### 4.2 策略推断代码（JAX）

```python
import jax
import jax.numpy as jp
from brax.training.agents.ppo import networks as ppo_networks

class PolicyInference:
    """使用 JAX 加载和运行 Brax PPO 策略。"""
    
    def __init__(self, checkpoint_path: Path, params: dict):
        """
        初始化策略推断器。
        
        Args:
            checkpoint_path: 检查点目录路径
            params: 从 Orbax 加载的参数字典
        """
        self.params = params
        self.checkpoint_path = checkpoint_path
        
        # 加载网络配置
        config_file = checkpoint_path / "ppo_network_config.json"
        with open(config_file, 'r') as f:
            self.net_config = json.load(f)
        
        # 创建网络工厂
        self.network_factory = ppo_networks.FeedForwardNetwork(
            output_size=4,  # action_size
            **self.net_config["network_factory_kwargs"]
        )
        
        # JIT 编译推断函数以加快速度
        self.policy_fn = jax.jit(self._policy_forward)
    
    def _policy_forward(self, obs_normalized: jp.ndarray) -> tuple:
        """
        前向传播：计算策略输出。
        
        Args:
            obs_normalized: [668] 归一化观测
        
        Returns:
            (action_mean, action_std): 策略分布参数
            state_value: 价值函数输出
        """
        # 调用网络
        policy_out, value_out = self.network_factory(
            obs_normalized[jp.newaxis, :],  # 添加批处理维度 [1, 668]
            self.params
        )
        
        # policy_out 是 TanhTransformedDistribution
        # 提取均值和标准差
        action_mean = policy_out.mean()  # [4]
        action_std = policy_out.stddev()  # [4]
        state_value = value_out[0]  # 标量
        
        return action_mean, action_std, state_value
    
    def sample_action(self, obs_normalized: np.ndarray) -> np.ndarray:
        """
        从策略分布中采样动作。
        
        Args:
            obs_normalized: [668] 归一化观测
        
        Returns:
            action: [4] 动作向量，范围 [-1, 1]
        """
        obs_jax = jp.array(obs_normalized, dtype=jp.float32)
        
        action_mean, action_std, _ = self.policy_fn(obs_jax)
        
        # 采样：a ~ N(μ, σ²)，然后应用 tanh
        key = jax.random.PRNGKey(0)  # 或使用真实的 RNG 状态
        noise = jax.random.normal(key, shape=(4,))
        action_raw = action_mean + action_std * noise
        action = jp.tanh(action_raw)  # 限制到 [-1, 1]
        
        return np.array(action)
    
    def get_action_mean(self, obs_normalized: np.ndarray) -> np.ndarray:
        """
        获取策略均值（无随机性，用于评估）。
        
        Args:
            obs_normalized: [668] 归一化观测
        
        Returns:
            action: [4] 动作均值，范围 [-1, 1]
        """
        obs_jax = jp.array(obs_normalized, dtype=jp.float32)
        action_mean, _, _ = self.policy_fn(obs_jax)
        return np.array(action_mean)


# 使用示例
ckpt_path = Path("logs/.../checkpoints/000128450560")
# params 从 Orbax 加载（见第 1 步）

policy = PolicyInference(ckpt_path, params)

# 在控制循环中：
obs_normalized = normalize_observation(obs_raw, mean, std)
action = policy.get_action_mean(obs_normalized)  # [4]，范围 [-1, 1]
```

### 4.3 策略推断代码（纯 NumPy，不依赖 JAX）

如果您的机器人环境不支持 JAX，可以使用 ONNX 导出框架：

```python
# 首先，从 JAX 模型导出为 ONNX
# (这需要在拥有 JAX 的编译环境中完成一次)

import onnx
import onnxruntime

class PolicyInferenceONNX:
    """使用 ONNX 运行时进行策略推断（无 JAX 依赖）。"""
    
    def __init__(self, onnx_model_path: str):
        """
        Args:
            onnx_model_path: 导出的 ONNX 模型路径
        """
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [o.name for o in self.ort_session.get_outputs()]
    
    def get_action(self, obs_normalized: np.ndarray) -> np.ndarray:
        """
        Args:
            obs_normalized: [668] 归一化观测
        
        Returns:
            action: [4] 动作均值
        """
        obs_input = obs_normalized.astype(np.float32).reshape(1, 668)
        
        ort_inputs = {self.input_name: obs_input}
        ort_outs = self.ort_session.run(self.output_names, ort_inputs)
        
        # 提取动作输出（取决于 ONNX 模型的输出顺序）
        action = ort_outs[0][0]  # [4]
        return action
```

---

## 🎮 第 5 步：动作转换为电机命令

### 5.1 动作空间映射

```
策略输出：[4]，范围 [-1, 1]
  ↓
对应的仿真关节：
  a[0] → 右手 OA_x（俯仰）
  a[1] → 右手 OA_y（转向）
  a[2] → 左手 OB_x（俯仰）
  a[3] → 左手 OB_y（转向）

（注意：真机有 8 个舵机，但仅控制 4 个 Hinge 自由度）
```

### 5.2 动作转换流程

```python
def transform_action_to_motor_commands(
    action_policy: np.ndarray,  # [4], range [-1, 1]
    qpos_curr: np.ndarray,       # [8], 当前关节角度
    sim_kinematics,
    ctrlrange_min: np.ndarray,   # [4] 或 [8]，最小值
    ctrlrange_max: np.ndarray,   # [4] 或 [8]，最大值
) -> np.ndarray:
    """
    将 [-1,1] 的策略动作转换为真机舵机的目标位置。
    
    转换步骤：
    1. 仿真动作 [4] → 仿真关节目标 (Hinge 角度)
    2. 从当前 Hinge 角度推导目标 qpos [8]
    3. 通过 Sim2Real 桥接将仿真 qpos 转换为真机舵机目标
    
    Args:
        action_policy: [4] 策略输出，来自网络
        qpos_curr: [8] 当前关节角度（从真机读取）
        sim_kinematics: 仿真运动学模块
        ctrlrange_min: [4] 最小控制范围（从 XML 加载）
        ctrlrange_max: [4] 最大控制范围（从 XML 加载）
    
    Returns:
        motor_targets: [8] 真机舵机的目标位置（弧度）
    """
    
    # 步骤 1: 将 [-1,1] 动作映射到控制范围
    #   a ∈ [-1, 1]  →  target ∈ [ctrlrange_min, ctrlrange_max]
    #   target = (ctrlrange_min + ctrlrange_max) / 2 + a * (ctrlrange_max - ctrlrange_min) / 2
    
    control_center = (ctrlrange_min + ctrlrange_max) / 2  # [4]
    control_range = (ctrlrange_max - ctrlrange_min) / 2  # [4]
    hinge_target = control_center + action_policy * control_range  # [4]
    
    # 步骤 2: 使用运动学约束派生 qpos 目标
    # （这是仿真机制的一部分，真机通过 IK 求解器完成）
    # 这里简化为假设映射关系（实际需要使用完整 Trico-Control 库）
    qpos_target = sim_kinematics.hinge_to_qpos(
        hinge_target,  # [4] Hinge 目标
        qpos_curr      # [8] 当前状态，用于连续性
    )  # → [8]
    
    # 步骤 3: Sim2Real 桥接
    # 将仿真的 qpos [8] 转换为真机舵机目标
    from sim2real_bridge import RealRobotBridge
    bridge = RealRobotBridge()
    
    servo_targets = bridge.hinge_angles_to_servo_angles(
        qpos_target,           # [8] 仿真 Hinge 角度
        apply_bank_ab=False    # (**关键**) 真机推断模式下不应用 Bank A/B
    )  # → [8] 真机舵机目标
    
    return servo_targets  # [8]
```

### 5.3 负载 ctrlrange 信息

```python
def load_ctrlrange(xml_path: str) -> tuple:
    """
    从 MuJoCo XML 文件加载致动器控制范围。
    
    XML 示例:
    <actuator>
      <position name="A_OA_x_right" joint="A_OA_x_right" 
                ctrlrange="-0.5 1.0" ... />
      ...
    </actuator>
    
    Returns:
        (ctrlrange_min, ctrlrange_max): 每个为 [n_actuators]
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    actuators = root.findall(".//actuator/position")
    ctrlrange_min = []
    ctrlrange_max = []
    
    for act in actuators:
        ctrlrange_str = act.get("ctrlrange")
        min_val, max_val = map(float, ctrlrange_str.split())
        ctrlrange_min.append(min_val)
        ctrlrange_max.append(max_val)
    
    return np.array(ctrlrange_min), np.array(ctrlrange_max)


# 在初始化时加载一次
xml_path = "path/to/trico_driver_v3.xml"
ctrlrange_min, ctrlrange_max = load_ctrlrange(xml_path)
```

### 5.4 真机舵机命令执行

```python
def execute_control_step(
    real_robot_interface,
    servo_targets: np.ndarray,  # [8] 舵机目标位置
    ctrl_dt: float = 0.02
) -> None:
    """
    向真机发送舵机目标位置命令。
    
    Args:
        real_robot_interface: 真机通信接口
        servo_targets: [8] 舵机目标（弧度）
        ctrl_dt: 控制周期（0.02s）
    
    舵机 ID 对应：
      0: OA_x_right
      1: OA_y_right
      2: OB_x_right
      3: OB_y_right
      4: OA_x_left
      5: OA_y_left
      6: OB_x_left
      7: OB_y_left
    """
    # 舵机 ID 到数组索引的映射
    servo_ids = [51, 52, 53, 54, 61, 62, 63, 64]
    
    for i, servo_id in enumerate(servo_ids):
        target_pos = servo_targets[i]
        real_robot_interface.set_servo_target(servo_id, target_pos)
    
    # 等待执行
    time.sleep(ctrl_dt)
```

---

## 🔄 第 6 步：完整控制循环

```python
def main_deployment_loop():
    """
    主部署控制循环：集成观测→归一化→推断→转换→执行。
    """
    import time
    from pathlib import Path
    
    # ===== 初始化阶段 =====
    print("初始化...")
    
    # 1. 加载检查点和参数
    ckpt_path = Path(
        "logs/TricoDriverSingle-contine_3_10M_near_dr-20260409-201011-trico_driver_v3"
        "/checkpoints/000128450560"
    )
    params = load_checkpoint_orbax(ckpt_path)
    norm_mean = np.array(params["0"]["mean"])
    norm_std = np.array(params["0"]["std"])
    
    # 2. 初始化组件
    real_robot = RealRobotInterface(port="/dev/ttyUSB0")
    sim_kinematics = SimulationKinematics(xml_path="trico_driver_v3.xml")
    obs_buffer = initialize_observation_buffer(real_robot, sim_kinematics)
    policy = PolicyInference(ckpt_path, params)
    ctrlrange_min, ctrlrange_max = load_ctrlrange(sim_kinematics.xml_path)
    
    # 3. 环境参数
    ctrl_dt = 0.02  # 控制频率 50 Hz
    episode_steps = 500  # 仿真中的步数 (= 10 秒)
    
    print(f"✓ 准备就绪。控制循环以 {1/ctrl_dt:.0f} Hz 运行。")
    
    # ===== 主控制循环 =====
    step = 0
    episode_start_time = time.time()
    
    try:
        while step < episode_steps:
            step_start = time.time()
            
            # 步骤 1: 采集观测
            frame = collect_single_frame_observation(
                real_robot, sim_kinematics, ctrl_dt
            )
            qpos = real_robot.get_joint_angles()
            
            # 步骤 2: 更新观测缓冲区
            obs_buffer.push(frame)
            obs_buffer.push_qpos(qpos)
            
            # 步骤 3: 构建 668 维原始观测
            obs_history = obs_buffer.get_history_stack()  # [660]
            obs_velocity = obs_buffer.get_discrete_velocity(ctrl_dt)  # [8]
            obs_raw = np.concatenate([obs_history, obs_velocity])  # [668]
            
            # **步骤 4: 归一化** (这是关键!)
            obs_normalized = normalize_observation(
                obs_raw, norm_mean, norm_std, eps=1e-8
            )
            
            # 步骤 5: 策略推断
            action = policy.get_action_mean(obs_normalized)  # [4], [-1, 1]
            
            # 步骤 6: 动作转换
            servo_targets = transform_action_to_motor_commands(
                action, qpos, sim_kinematics,
                ctrlrange_min, ctrlrange_max
            )  # [8]
            
            # 步骤 7: 执行动作
            execute_control_step(real_robot, servo_targets, ctrl_dt)
            
            # 步骤 8: 日志记录（可选）
            if step % 50 == 0:
                print(f"步 {step:3d} | "
                      f"动作: {action} | "
                      f"舵机: {servo_targets}")
            
            step += 1
            
            # 同步控制频率
            elapsed = time.time() - step_start
            sleep_time = ctrl_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n中止...")
    finally:
        # 清理
        real_robot.close()
        elapsed_total = time.time() - episode_start_time
        print(f"✓ 剧集完成。总时间: {elapsed_total:.1f}s, 步数: {step}")


if __name__ == "__main__":
    main_deployment_loop()
```

---

## 📋 检查清单

在部署前，确保：

- [ ] 检查点路径正确：`000128450560`
- [ ] 归一化参数已加载：`mean` 和 `std` 形状均为 [668]
- [ ] 观测缓冲区正确初始化（首帧重复 30 次）
- [ ] 观测归一化应用在策略推断前
- [ ] 策略输出是 [4] 维，范围 [-1, 1]
- [ ] 动作转换使用正确的 ctrlrange_min/max
- [ ] 真机舵机顺序与仿真一致
- [ ] Sim2Real 桥接使用 `apply_bank_ab=False`
- [ ] 控制循环以 50 Hz（0.02s）同步执行

---

## 🚨 常见陷阱

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| 策略输出都是 NaN | 应用了错误的归一化或加载了错误的 mean/std | 验证 mean/std 来自正确的检查点 |
| 机器人移动不同步 | 观测历史不正确或速度计算错误 | 确保 30 帧缓冲区正确维护，速度 = (qpos_curr - qpos_prev) / 0.02 |
| 动作幅度太大或太小 | ctrlrange_min/max 加载不正确 | 从 XML 直接检查致动器范围 |
| 机器人跳变 | Sim2Real 桥接中应用了 Bank A/B 转换 | 使用 `apply_bank_ab=False` 在推断模式下 |
| 持续崩溃/保护触发 | 初始状态与仿真相差太大 | 在首次运行前执行域随机化初始化过程 |

---

## 📚 参考

- **Brax PPO 文档**：[github.com/google-deepmind/brax](https://github.com/google-deepmind/brax)
- **Sim2Real 桥接**：[Trico-Control/trico_code/policy_adapter/convert/sim2real_bridge.py](../Trico-Control/trico_code/policy_adapter/convert/sim2real_bridge.py)
- **检查点配置**：[logs/.../checkpoints/000128450560/](logs/TricoDriverSingle-contine_3_10M_near_dr-20260409-201011-trico_driver_v3/checkpoints/000128450560/)
- **运动学参考**：[Trico-Control/trico_code/mechanics/](../Trico-Control/trico_code/mechanics/)

---

**最后更新**：2026-04-10  
**检查点版本**：000128450560 (128M+ 步)  
**框架**：Brax 0.12.5+, JAX, MuJoCo 3.3.6+

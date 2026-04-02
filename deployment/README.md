# Checkpoint直接部署指南

## 📋 概览

本模块为部署脚本包。**核心推理逻辑已迁移到 `mujoco_playground._src.inference` 模块**（v1.1+），本模块为向后兼容性而保留。

### 关键特性
- ✓ 直接从JAX checkpoint加载网络参数
- ✓ 处理sim-real观测差异（转换函数、丢帧处理等）
- ✓ 模块化设计，易于扩展
- ✓ Sim验证→真机部署的平滑过渡

---

## 📦 结构重组 (v1.1)

自v1.1开始，项目采用新的目录结构以提高可维护性：

```
mujoco_playground/
├── _src/
│   └── inference/              ← 核心推理模块（新位置）
│       ├── __init__.py
│       ├── checkpoint_loader.py
│       ├── observation_adapter.py
│       ├── policy.py
│       └── sim_real_mapping.py
│
└── deployment/                 ← 部署脚本包（应用层）
    └── ...                    （从 _src.inference 重导出）
```

### 导入方式

**推荐（新）:**
```python
from mujoco_playground._src.inference import TricoPolicy, PolicyDeployer
```

**兼容（旧）:**
```python
from deployment import TricoPolicy, PolicyDeployer
```

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────┐
│    TricoPolicy + PolicyDeployer         │
│  (推理入口，统一sim和real接口)         │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌─────────────┐  ┌──────────────────┐
│ Checkpoint  │  │ Observation      │
│ Loader      │  │ Adapter          │
└─────────────┘  ├──────────────────┤
                 │ SimAdapter       │ ─→ 仿真验证
                 │ RealAdapter      │ ─→ 真机部署
                 └──────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    Joint Angles   Force Sensors  Object Pose
    (8 dims)       (6 dims)        (7 dims)
    + Relative Position (6 dims)
    ────────────────────────────────────
         → 27维标准化观测
```

---

## 📁 文件结构

```
deployment/
├── __init__.py                    # 模块入口
├── checkpoint_loader.py           # checkpoint加载
├── observation_adapter.py         # 观测适配层
└── policy.py                      # 策略推理和部署

scripts/
└── verify_checkpoint.py           # 验证脚本
```

---

## 🚀 快速开始

### 1. 验证Checkpoint

```bash
python3 scripts/verify_checkpoint.py \
    --checkpoint logs/555-20260202-134501/checkpoints/000128450560
```

**输出示例:**
```
✓ Checkpoint structure verified
✓ Configuration looks correct
✓ Observation dimension: 27
```

### 2. 在仿真中验证

```python
from deployment import TricoPolicy, SimObservationAdapter, PolicyDeployer
from mujoco_playground._src.manipulation.trico import TricoEnv

# 加载策略
policy = TricoPolicy(
    checkpoint_dir='logs/555.../checkpoints/000128450560',
    obs_adapter=None,  # 稍后创建
)

# 创建环境和适配器
env = TricoEnv()
adapter = SimObservationAdapter(env.reset())
policy.obs_adapter = adapter

# 运行推理循环
deployer = PolicyDeployer(policy)
deployer.inference_loop_with_adapter(env, num_steps=100)
```

### 3. 准备真机部署

```python
from deployment import TricoPolicy, RealObservationAdapter

# 初始化硬件接口
servo_reader = HardwareServoReader()        # 你的舵机读取接口
force_reader = HardwareForceSensor()        # 力传感器接口
vision = AirTagVisionModule()               # 你现有的AirTag模块

# 创建观测适配器
adapter = RealObservationAdapter(
    servo_reader=servo_reader,
    force_sensor_reader=force_reader,
    vision_module=vision,
    servo_to_sim_fn=servo_position_conversion_fn,  # 转换函数
)

# 创建策略
policy = TricoPolicy(
    checkpoint_dir='...',
    obs_adapter=adapter,
)

# 推理循环
while robot_running:
    action = policy.forward_from_adapter()
    cmd_bytes = policy.process_action_for_real(action)
    robot.send_command(cmd_bytes)
```

---

## 🔑 关键点说明

### A. 观测维度 (27D)

| 索引 | 维度 | 名称 | 来源 |
|------|------|------|------|
| 0-7 | 8 | 关节角度 | 舵机编码器 / sim qpos |
| 8-13 | 6 | 力传感器 | 力传感器 / sim sensordata |
| 14-16 | 3 | 物体位置 | 摄像头 / sim xpos |
| 17-20 | 4 | 物体四元数 | 摄像头 / sim quat |
| 21-26 | 6 | 相对位置 | 计算: [tip_right - obj_pos, tip_left - obj_pos] |

### B. Sim-Real差异处理

| 差异 | Sim | Real | 处理方案 |
|------|-----|------|---------|
| 舵机pos | 完全对应 | 需要转换 | `servo_to_sim_fn()` |
| 力传感器 | 精确 | 有噪声 | 使用training时的norm统计 |
| 物体pose | 稳定 | 偶尔丢帧 | 使用上一帧或预测 |
| 相对位置 | 直接可得 | 需要FK计算 | 从关节角度计算 |

### C. 观测归一化

Checkpoint中保存了training时的obs归一化统计：
```
normalizer_params = params[0]  # RunningStatisticsState
  ├─ mean: (27,)
  ├─ std: (27,)
  └─ count, summed_variance
```

推理时：
```
normalized_obs = (obs - mean) / std
```

---

## ⚠️ 当前限制和TODO

1. **Checkpoint参数加载**
   - 问题: Orbax checkpoint保存在GPU上，CPU加载需要特殊处理
   - 状态: ⏳ 需要实现
   - 影响: forward pass暂时不可用，但架构完整

2. **真机动作处理**
   - 问题: 不知道真机协议细节
   - 状态: ⏳ 需要你的硬件接口
   - 接口: `policy.process_action_for_real(action)` - 预留

3. **FK计算**
   - 问题: 真机计算指尖位置需要URDF或DH参数
   - 状态: ⏳ 需要你的机器人model
   - 接口: `adapter.get_tip_positions()` - 预留

4. **Mean/std提取**
   - 问题: 需要从GPU checkpoint或训练日志中提取
   - 状态: ⏳ 需要完整的checkpoint加载器
   - 建议: 从Tensorboard事件文件或直接运行训练脚本提取

---

## 🔧 集成清单

### 仿真验证（本周）
- [ ] 完成checkpoint参数加载
- [ ] 实现forward pass
- [ ] 在sim中验证performance
- [ ] 对比with original training

### 真机准备（下周）
- [ ] 实现舵机pos转换函数 `servo_to_sim_fn()`
- [ ] 实现真机FK计算 `get_tip_positions()`
- [ ] 实现硬件接口集成
- [ ] 测试观测管道

### 真机部署（后续）
- [ ] 实现 `process_action_for_real()`
- [ ] 实现通信协议
- [ ] 安全测试
- [ ] 运行验证

---

## 📞 需要确认的事项

你需要提供：

1. **舵机pos转换函数**
   ```python
   def servo_to_sim(servo_pos: np.ndarray) -> np.ndarray:
       """将真机舵机pos转换到sim坐标系"""
       # servo_pos: (8,) 真机读到的位置
       # return: (8,) 对应的sim坐标系位置
       pass
   ```

2. **硬件接口模板**
   ```python
   class HardwareServoReader:
       def read(self) -> np.ndarray:  # (8,)
           pass
   
   class HardwareForceSensor:
       def read(self) -> np.ndarray:  # (6,)
           pass
   ```

3. **正向运动学(FK)函数**
   ```python
   def compute_tip_positions(joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
       """从关节角度计算指尖位置"""
       # return: (tip_right, tip_left) 两个(3,)向量
       pass
   ```

4. **Mean/std值**
   - 从training日志中提取观测归一化统计
   - 或运行脚本生成

---

## 🎯 下一步行动

**建议顺序：**

1. **现在**: 架构设计完成 ✓
2. **今天**: 
   - [ ] 处理GPU device问题，完成checkpoint参数加载
   - [ ] 实现实际的forward pass
3. **明天**:
   - [ ] 在sim中验证推理
   - [ ] 对比training performance
4. **后天**:
   - [ ] 上真机试pilot
   - [ ] 基于feedback调整

---

## 📝 更新日志

- **2026-03-02**: 初始架构设计和实现
  - Created checkpoint_loader.py
  - Created observation_adapter.py  
  - Created policy.py
  - Created verify_checkpoint.py

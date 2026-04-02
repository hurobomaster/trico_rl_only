## 单个 CSPM 模块调试文档

这个目录包含了用于调试和测试单个 CSPM (Crossed Spoked Parallel Mechanism) 机构的脚本和工具。

### 文件说明

#### `debug_cspm_ik.py`
- **目的**：调试 CSPM 的 IK 求解过程
- **功能**：
  - 单个 CSPM 的 IK 反解验证
  - 方向向量与舵机角的转换测试
  - 多解情况下的解选择测试
- **运行**：`python debug_cspm_ik.py`
- **检查项**：
  - IK 往返误差是否 < 0.01°
  - 两个数学解是否都被正确计算
  - `prev_state` 同步是否生效

#### `debug_real2sim_pipeline.py`
- **目的**：测试完整的 Real2Sim 观测转换管道
- **功能**：
  - 真机舵机角 → FK → 方向向量 → VirtualHingeMapper → 仿真Hinge角
  - 每一步的中间结果检查
  - 坐标系偏置验证
- **运行**：`python debug_real2sim_pipeline.py`
- **检查项**：
  - FK 输出向量的模 ≈ 1.0（单位向量）
  - 偏置应用前后的 Hinge 角差异
  - HOME 位置的零位偏置计算

#### `oa_interactive.py`
- **目的**：交互式的 OA(外层) 链测试
- **功能**：
  - 实时输入舵机角，查看对应的方向向量
  - Hinge 角的可视化
  - 手动验证 FK/IK 的正确性
- **运行**：`python oa_interactive.py`
- **交互流程**：
  1. 输入舵机角（如：260, 170）
  2. 观看 FK 计算的方向向量
  3. 观看 VirtualHingeMapper 转换的 Hinge 角
  4. 通过 IK 验证反向转换

#### `oa_right_simple.xml`
- **目的**：简化的 MuJoCo 场景文件
- **用途**：OA 链的仿真模型配置
- **格式**：MJCF (MuJoCo XML)
- **用于**：与仿真环境联动调试

---

## 使用场景

### 场景1：验证单个 CSPM 的 IK 反解

```bash
python debug_cspm_ik.py
```

**预期输出示例**：
```
=== CSPM IK 测试 ===
输入舵机角: [260.0, 170.0]°

FK 正解:
  OA 方向向量: [-0.123, 0.456, 0.789]
  OB 方向向量: [0.234, -0.567, 0.891]

IK 反解 (候选解):
  解1: θ1=260.0° θ2=170.0° (误差: 0.000°) ← 选中 (离prev最近)
  解2: θ1=100.0° θ2=190.0° (误差: 0.000°)

✓ 往返转换成功
```

**验证清单**：
- [ ] 两个数学解都被正确计算
- [ ] 选中的解离 prev_state 最近
- [ ] 误差 < 0.01°

### 场景2：追踪完整的观测管道

```bash
python debug_real2sim_pipeline.py
```

**预期输出示例**：
```
=== Real2Sim 观测管道追踪 ===

Step 1: FK 正运动学
  输入舵机: [260, 170]°
  OA 向量: [-0.123, 0.456, 0.789]
  向量模: 1.000 ✓

Step 2: VirtualHingeMapper (应用偏置)
  OA_x (原始): 0.123 rad
  OA_y (原始): 0.456 rad
  偏置应用: OA_offset_x = 121.6°
  OA_x (补偿后): -0.234 rad

Step 3: 零位偏置消除
  初始偏置: [0.001, 0.002, 0.003, 0.004]
  相对Hinge角: [-0.235, 0.454, 0.245, 0.889]

✓ 观测转换完成
```

**验证清单**：
- [ ] FK 输出向量的模是否 ≈ 1.0
- [ ] VirtualHingeMapper 的偏置是否正确应用
- [ ] HOME 位置的零位偏置是否被正确计算

### 场景3：交互式调试

```bash
python oa_interactive.py
```

**交互示例**：
```
输入舵机角 (格式: θ1,θ2): 260,170

FK 结果:
  OA = [-0.123, 0.456, 0.789]
  OB = [0.234, -0.567, 0.891]

Hinge 角 (应用偏置):
  OA_x = -0.234 rad, OA_y = 0.345 rad
  OB_x = 0.456 rad, OB_y = -0.123 rad

反解验证 (IK):
  目标向量: OA = [-0.123, 0.456, 0.789]
  IK 结果: θ1 = 260.00°, θ2 = 170.00°
  误差: 0.000°

继续? (y/n): n
```

---

## 关键验证指标

### 1. FK 向量正确性

**检查项**：向量的模
```python
import numpy as np

dir_oa = fk.get_oa_vector(servo_angle)
assert np.abs(np.linalg.norm(dir_oa) - 1.0) < 0.01, "不是单位向量！"
```

**正常范围**：模 ∈ [0.99, 1.01]

### 2. VirtualHingeMapper 偏置

**检查项**：坐标系偏置的应用
```python
# OA 链的偏置应该是 121.6°
# OB 链的偏置应该是 211.6°
```

**验证方法**：
- 将偏置值与 Trico-Control 源码中的值对比
- 检查偏置方向（+还是-）是否一致


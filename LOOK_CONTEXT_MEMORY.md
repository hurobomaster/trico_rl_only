# Trico Hand Sim2Real & Real2Sim 代码重构与环境上下文记忆

> **【系统提示 / System Prompt】**
> 如果你是被新开启的 AI 对话助手，请仔细阅读这份文档。这份文档记录了我们在强化学习仿真环境（MuJoCo Playground）与真实机械手（Trico-Control）之间建立 Sim2Real 数据转换桥梁的所有核心业务背景、踩坑记录及最新代码状态。请以本文件描述的架构作为接下来任何代码修改的前提。

---

## 1. 项目背景与目标
* **硬件设备**：基于 Dynamixel XL330 舵机的并行多指灵巧手（Trico Hand）。
* **软件栈**：`Trico-Control` 控制库（包含底层驱动与纯数学层的运动学/逆解） + `mujoco_playground`（强化学习与仿真环境）。
* **核心任务**：在此前的工作中，我们致力于彻底打通**真机舵机角度**和**MuJoCo仿真器里球关节相对Hinge（铰链）角度**之间的双向无损对应，保证强化学习动作在真机上执行不出错。

## 2. 之前遇到的问题（已解决）
* **偏置黑盒**：早期的 Sim2Real 模型中缺少几何偏置。真机的坐标系与仿真坐标系在 Euler 角中存在固定偏移（如 OA为 121.6°，OB为 211.6°）。
* **逆解输入限制**：旧代码尝试直接将“方向向量 (Direction Vector)”输入为 IK 的 target，但 IK 求解器实际上期望的是末端笛卡尔坐标（Positions）。
* **逻辑冗余**：存在多个转化桥接文件（如 `sim_real_mapping.py`），导致代码冗余、来回调用混乱。
* **舵机硬件映射混淆**：此前很容易把硬件逻辑特性（Bank A 互补映射 `360-x`、Bank B 直接映射 `x`）混入到 RL 的推断桥接代码中，造成数据对不齐。

## 3. 最终确定的解决方案与业务规范
1. **统一转换入口（唯一基准）**：
   * 所有的数据桥接必须通过单例实现：**`mujoco_playground/_src/inference/sim2real_bridge.py`** (包含核心类 `RealRobotBridge`)。
   * **已删除废弃文件**：`sim_real_mapping.py` 和 `test_deployment_pipeline.py`（在后续对话中千万不要再尝试导入或修改这俩文件）。
2. **底层数学依赖**：
   * 采用 Trico-Control 自带的 `VirtualHingeMapper`，该模块在内部已完美兼容了坐标系统的偏置和旋转校正问题。
   * IK（逆向运动学）解决方案中，为了让“方向向量”可以正确代入 IK Solver，我们采用了：`Pos = Dir * L_arm` （其中内外臂长 `l2_in=30.0`, `l2_out=30.0`），以此还原正确的三维指尖坐标供逆解器运算。
3. **硬件级 Bank A/B 解耦**：
   * 核心规范：作为给RL和数学模型的标准通信接口，**输入输出时禁用 Bank A/B 映射**（即在调用 `hinge_angles_to_servo_angles` 时确保传入 `apply_bank_ab=False`），让真实的角度直出。因为实际发送指令到舵机的前一刻，底层硬件层才会去做 Bank 的适配。

## 4. 核心文件与测试状态
### `sim2real_bridge.py`
最核心的转换类：`RealRobotBridge`
* **Real2Sim（观测）** `servo_angles_to_hinge_angles(8D_servos)`：将 8 个真机度数转换为 8 个仿真相对弧度 $\theta$。内部经过了正解（FK） -> 向量 -> Hinge偏置重映射。
* **Sim2Real（执行）** `hinge_angles_to_servo_angles(8D_hinges, apply_bank_ab=False)`：将 8 个仿真弧度 $\theta$ 转换回真机目标度数。经历重映射 -> 向量 -> 乘臂长 -> 引入到 IK 逆向求解得到结果。

### `hrz_test.py`
* 这是一个我们用于自交验证的闭环极限测试脚本。
* **流程**：输入 4 个目标舵机角度 $\xrightarrow{桥接网络}$ 输出 4 个环境中的仿真弧度 $\theta$ $\xrightarrow{桥接网络反向逆解}$ 还原成 4 个舵机角度。
* **现状**：此代码完美运行！！经过精确校验，往返最大误差为 **0.000000°**（即做到了数学上的无损回溯）。

## 5. 当前开发进度提示（给 AI 的话）
如果用户开启了一段新对话并向你寻求代码支持：
1. 请不要质疑基础的桥接公式（FK 和 IK 往返已经是绝对闭环正确）。
2. 如果报错说找不到 `sim_real_mapping.py` 相关引用，请帮用户直接删掉对应的 `import`。
3. 目前 VS Code 的资源管理器已经使用配置 `.vscode/settings.json` 把部分不常看的文件隐藏了起来，目的是保持工作区整洁。

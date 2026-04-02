import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer
import time

# 获取当前脚本所在目录，XML 文件应该在同一目录
current_dir = Path(__file__).parent
xml_path = current_dir / "oa_right_simple.xml"

if not xml_path.exists():
    print(f"❌ 错误: 无法找到 XML 文件: {xml_path}")
    exit(1)

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

viz_frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "viz_base_frame_right")
oa_chain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_OA_Chain_right")

joint_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "OA_active_x_right")
joint_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "OA_active_y_right")

oa_endpoint_local = np.array([0, 0, -0.034])

print("\n" + "="*80)
print("MuJoCo 交互式关节控制")
print("操作说明：按住 CTRL + 鼠标拖动 来控制关节")
print("="*80 + "\n")

frame = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        if frame % 5 == 0:
            angle_x = data.qpos[joint_x_id]
            angle_y = data.qpos[joint_y_id]
            
            viz_pos = data.body(viz_frame_id).xpos.copy()
            viz_rot = data.body(viz_frame_id).xmat.reshape(3, 3)
            
            oa_pos = data.body(oa_chain_id).xpos.copy()
            oa_rot = data.body(oa_chain_id).xmat.reshape(3, 3)
            
            oa_endpoint_global = oa_pos + oa_rot @ oa_endpoint_local
            oa_endpoint_in_viz_frame = viz_rot.T @ (oa_endpoint_global - viz_pos)
            
            x_mm = oa_endpoint_in_viz_frame[0] * 1000
            y_mm = oa_endpoint_in_viz_frame[1] * 1000
            z_mm = oa_endpoint_in_viz_frame[2] * 1000
            
            # 计算单位向量
            vec = np.array([x_mm, y_mm, z_mm])
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                ux, uy, uz = 0, 0, 0
            else:
                ux, uy, uz = vec / norm

            angle_x_deg = np.degrees(angle_x)
            angle_y_deg = np.degrees(angle_y)
            
            # 只显示两个hinge的值和A点xyz坐标及单位向量
            print(f"\rθx={angle_x_deg:6.1f}° θy={angle_y_deg:6.1f}° | Pos(mm):[{x_mm:5.1f}, {y_mm:5.1f}, {z_mm:5.1f}] | Unit:[{ux:.4f}, {uy:.4f}, {uz:.4f}]", end="", flush=True)
        
        frame += 1
        
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

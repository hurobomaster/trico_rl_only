from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


XML_NAME = "trico_driver_v2.xml"
TIP_SITE_NAMES = ("tip_right", "tip_left")
HANDLE_SITE_NAME = "driver_handle_center"
TIP_SITE_GROUP = 2
PRINT_EVERY_N_FRAMES = 20


def _format_vec(vec: np.ndarray, scale: float = 1.0) -> str:
    scaled = vec * scale
    return f"[{scaled[0]: .4f}, {scaled[1]: .4f}, {scaled[2]: .4f}]"


def _format_quat(quat: np.ndarray) -> str:
    return f"[{quat[0]: .4f}, {quat[1]: .4f}, {quat[2]: .4f}, {quat[3]: .4f}]"


def _site_pose(data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = data.site_xpos[site_id].copy()
    mat = data.site_xmat[site_id].reshape(3, 3).copy()
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return pos, mat, quat


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    xml_path = current_dir / XML_NAME

    if not xml_path.exists():
        raise FileNotFoundError(f"Cannot find XML file: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    tip_site_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in TIP_SITE_NAMES
    }
    handle_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, HANDLE_SITE_NAME
    )

    print("=" * 90)
    print("MuJoCo Trico tip pose viewer")
    print(f"XML: {xml_path}")
    print(f"Tracking sites: {', '.join(TIP_SITE_NAMES)}")
    print("The viewer shows only site group 2 and draws site frames.")
    print("=" * 90)

    frame = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            for i in range(len(viewer.opt.sitegroup)):
                viewer.opt.sitegroup[i] = 0
            viewer.opt.sitegroup[TIP_SITE_GROUP] = 1
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)

            right_pos, right_mat, right_quat = _site_pose(
                data, tip_site_ids["tip_right"]
            )
            left_pos, left_mat, left_quat = _site_pose(data, tip_site_ids["tip_left"])
            handle_pos = data.site_xpos[handle_site_id].copy()

            tip_gap = np.linalg.norm(right_pos - left_pos)
            right_handle_dist = np.linalg.norm(right_pos - handle_pos)
            left_handle_dist = np.linalg.norm(left_pos - handle_pos)

            left_text = (
                f"tip_right pos [m]: {_format_vec(right_pos)}\n"
                f"tip_left  pos [m]: {_format_vec(left_pos)}\n"
                f"tip gap [mm]: {tip_gap * 1000: .3f}\n"
                f"dist to handle [mm] R/L: "
                f"{right_handle_dist * 1000: .3f} / {left_handle_dist * 1000: .3f}"
            )
            right_text = (
                f"tip_right quat [wxyz]: {_format_quat(right_quat)}\n"
                f"tip_left  quat [wxyz]: {_format_quat(left_quat)}\n"
                f"tip_right z-axis: {_format_vec(right_mat[:, 2])}\n"
                f"tip_left  z-axis: {_format_vec(left_mat[:, 2])}"
            )
            viewer.set_texts((None, None, left_text, right_text))
            viewer.sync()

            if frame % PRINT_EVERY_N_FRAMES == 0:
                print(
                    " | ".join(
                        [
                            f"tip_right xyz [mm] {_format_vec(right_pos, scale=1000.0)}",
                            f"tip_left xyz [mm] {_format_vec(left_pos, scale=1000.0)}",
                            f"gap [mm] {tip_gap * 1000: .3f}",
                        ]
                    ),
                    flush=True,
                )

            frame += 1
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

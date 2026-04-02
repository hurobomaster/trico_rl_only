
"""Constants for leap hand."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "leap_hand"
CUBE_XML = ROOT_PATH / "xmls" / "scene_mjx_cube.xml"

NQ = 16
NV = 16
NU = 16

JOINT_NAMES = [
    # index
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    # middle
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    # ring
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    # thumb
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
]

ACTUATOR_NAMES = [
    # index
    "if_mcp_act",
    "if_rot_act",
    "if_pip_act",
    "if_dip_act",
    # middle
    "mf_mcp_act",
    "mf_rot_act",
    "mf_pip_act",
    "mf_dip_act",
    # ring
    "rf_mcp_act",
    "rf_rot_act",
    "rf_pip_act",
    "rf_dip_act",
    # thumb
    "th_cmc_act",
    "th_axl_act",
    "th_mcp_act",
    "th_ipl_act",
]

FINGERTIP_NAMES = [
    "th_tip",
    "if_tip",
    "mf_tip",
    "rf_tip",
]

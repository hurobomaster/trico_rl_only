"""Trico screwdriver rotation task."""

import os

import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env


DEBUG_TRICO_DRIVER_OBS = False


class TricoDriverEnv(mjx_env.MjxEnv):
    """Screwdriver rotation task using the Trico hand."""

    def __init__(self, config: config_dict.ConfigDict = None, **kwargs):
        del kwargs
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self._xml_path = os.path.join(curr_dir, "xmls", "trico_driver_v3.xml")

        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 50
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.tolerance = 1e-4
        self._mjx_model = mjx.put_model(self._mj_model)

        if config is None:
            config = config_dict.ConfigDict()
        config.ctrl_dt = 0.02
        config.sim_dt = 0.001
        if "episode_length" not in config:
            config.episode_length = 500
        if "action_repeat" not in config:
            config.action_repeat = 1
        if "rotate_reward_scale" not in config:
            config.rotate_reward_scale = 2.5
        if "symmetry_penalty_scale" not in config:
            config.symmetry_penalty_scale = 0.1
        if "tip_penetration_penalty_scale" not in config:
            config.tip_penetration_penalty_scale = 150.0
        if "allowed_tip_penetration" not in config:
            config.allowed_tip_penetration = 0.005
        if "delta_action_penalty_scale" not in config:
            config.delta_action_penalty_scale = 0.0

        super().__init__(config=config)

        actuator_joint_ids = self._mj_model.actuator_trnid[:, 0]
        actuator_qpos_adr = self._mj_model.jnt_qposadr[actuator_joint_ids]
        self._actuator_qpos_adr = jnp.asarray(actuator_qpos_adr, dtype=jnp.int32)
        self._ctrlrange = jnp.asarray(self._mj_model.actuator_ctrlrange)
        self._init_ctrl_qpos = jnp.asarray(self._mj_model.qpos0[actuator_qpos_adr])

        self._tip_right_id = self._mj_model.site("tip_right").id
        self._tip_left_id = self._mj_model.site("tip_left").id
        self._driver_center_site_id = self._mj_model.site("driver_handle_center").id
        self._tip_right_geom_id = self._mj_model.geom("tip_geom_right").id
        self._driver_handle_collision_geom_id = self._mj_model.geom(
            "driver_handle_collision"
        ).id

        self._driver_joint_id = self._mj_model.joint("driver_handle_joint").id
        self._driver_qpos_adr = int(self._mj_model.jnt_qposadr[self._driver_joint_id])
        self._driver_qvel_adr = int(self._mj_model.jnt_dofadr[self._driver_joint_id])

        self._rotate_reward_scale = float(config.rotate_reward_scale)
        self._reward_clip_min = -4.0
        self._reward_clip_max = 4.0
        self._symmetry_penalty_scale = float(config.symmetry_penalty_scale)
        self._allowed_tip_penetration = float(config.allowed_tip_penetration)
        tip_radius = float(self._mj_model.geom_size[self._tip_right_geom_id][0])
        threshold_site_id = mujoco.mj_name2id(
            self._mj_model,
            mujoco.mjtObj.mjOBJ_SITE,
            "driver_penetration_threshold",
        )
        if threshold_site_id >= 0:
            handle_half_width = float(self._mj_model.site_size[threshold_site_id][0])
        else:
            handle_half_width = float(
                self._mj_model.geom_size[self._driver_handle_collision_geom_id][1]
            )
        self._min_tip_center_dist = (
            handle_half_width + tip_radius - self._allowed_tip_penetration
        )
        self._penetration_penalty_scale = float(config.tip_penetration_penalty_scale)
        self._delta_action_penalty_scale = float(config.delta_action_penalty_scale)
        # Only regularize left/right action symmetry once the fingertips are close
        # enough to the handle for a symmetric pinch to be meaningful.
        self._symmetry_gate_dist = self._min_tip_center_dist + 0.010
        self._symmetry_gate_width = 0.008

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = jnp.asarray(self.mjx_model.qpos0)
        qpos = qpos.at[self._driver_qpos_adr].set(0.0)
        qvel = jnp.zeros(self.mjx_model.nv)

        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._init_ctrl_qpos)
        data = mjx.forward(self.mjx_model, data)

        obs = self._get_obs(data)
        reward = jnp.array(0.0, dtype=jnp.float32)
        done = jnp.array(0.0, dtype=jnp.float32)
        metrics = {
            "reward_rotate": jnp.array(0.0, dtype=jnp.float32),
            "penalty_symmetry": jnp.array(0.0, dtype=jnp.float32),
            "penalty_tip_penetration": jnp.array(0.0, dtype=jnp.float32),
            "penalty_delta_action": jnp.array(0.0, dtype=jnp.float32),
            "driver_joint_pos": jnp.array(0.0, dtype=jnp.float32),
            "driver_joint_vel": jnp.array(0.0, dtype=jnp.float32),
            "reward": jnp.array(0.0, dtype=jnp.float32),
        }
        info = {
            "rng": rng,
            "last_action": jnp.zeros(self.action_size, dtype=jnp.float32),
        }
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        ctrl_center = (self._ctrlrange[:, 0] + self._ctrlrange[:, 1]) * 0.5
        ctrl_half = (self._ctrlrange[:, 1] - self._ctrlrange[:, 0]) * 0.5
        motor_targets = ctrl_center + action * ctrl_half
        motor_targets = jnp.clip(
            motor_targets, self._ctrlrange[:, 0], self._ctrlrange[:, 1]
        )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        driver_joint_pos = data.qpos[self._driver_qpos_adr]
        driver_joint_vel_signed = data.qvel[self._driver_qvel_adr]
        tip_right = data.site_xpos[self._tip_right_id]
        tip_left = data.site_xpos[self._tip_left_id]
        handle_center = data.site_xpos[self._driver_center_site_id]
        rotate_reward = (
            jnp.clip(
                driver_joint_vel_signed,
                self._reward_clip_min,
                self._reward_clip_max,
            )
            * self._rotate_reward_scale
        )
        tip_dists = jnp.array(
            [
                jnp.linalg.norm(tip_right - handle_center),
                jnp.linalg.norm(tip_left - handle_center),
            ]
        )
        mean_tip_dist = jnp.mean(tip_dists)
        symmetry_gate = jnp.clip(
            (self._symmetry_gate_dist - mean_tip_dist) / self._symmetry_gate_width,
            0.0,
            1.0,
        )
        # Penalize action asymmetry, not state asymmetry. This better matches the
        # control objective and does not punish transient mirrored approach poses.
        symmetry_penalty = self._symmetry_penalty_scale * symmetry_gate * jnp.mean(
            jnp.abs(action[:4] - action[4:])
        )
        tip_penetration = jnp.maximum(0.0, self._min_tip_center_dist - tip_dists)
        penetration_penalty = self._penetration_penalty_scale * jnp.sum(
            tip_penetration
        )
        delta_action = action - state.info["last_action"]
        delta_action_penalty = self._delta_action_penalty_scale * jnp.mean(
            jnp.square(delta_action)
        )

        obs = self._get_obs(data)
        is_invalid = ~jnp.all(jnp.isfinite(obs))
        total_reward = (
            rotate_reward
            - symmetry_penalty
            - penetration_penalty
            - delta_action_penalty
        )
        reward = jnp.where(is_invalid, 0.0, total_reward)
        done = is_invalid.astype(jnp.float32)

        metrics = {
            "reward_rotate": rotate_reward,
            "penalty_symmetry": symmetry_penalty,
            "penalty_tip_penetration": penetration_penalty,
            "penalty_delta_action": delta_action_penalty,
            "driver_joint_pos": driver_joint_pos,
            "driver_joint_vel": driver_joint_vel_signed,
            "reward": reward,
        }

        if DEBUG_TRICO_DRIVER_OBS:
            jax.debug.print(
                (
                    "driver pos={pos:.3f} vel={vel:.3f} "
                    "r_rot={rrot:.3f} p_sym={psym:.3f} p_tip={ptip:.3f} "
                    "p_delta={pdelta:.3f} "
                    "reward={rew:.3f}"
                ),
                pos=driver_joint_pos,
                vel=driver_joint_vel_signed,
                rrot=rotate_reward,
                psym=symmetry_penalty,
                ptip=penetration_penalty,
                pdelta=delta_action_penalty,
                rew=reward,
            )

        info = {**state.info, "last_action": action}
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        qpos_act = data.qpos[self._actuator_qpos_adr]
        driver_joint_pos = data.qpos[self._driver_qpos_adr : self._driver_qpos_adr + 1]
        driver_joint_vel = data.qvel[self._driver_qvel_adr : self._driver_qvel_adr + 1]
        handle_center = data.site_xpos[self._driver_center_site_id]
        tip_right = data.site_xpos[self._tip_right_id]
        tip_left = data.site_xpos[self._tip_left_id]
        tip_right_rot = jnp.reshape(data.site_xmat[self._tip_right_id], (9,))
        tip_left_rot = jnp.reshape(data.site_xmat[self._tip_left_id], (9,))
        rel_pos = jnp.concatenate(
            [handle_center - tip_right, handle_center - tip_left]
        )
        return jnp.concatenate(
            [
                qpos_act,
                driver_joint_pos,
                driver_joint_vel,
                tip_right,
                tip_left,
                tip_right_rot,
                tip_left_rot,
                handle_center,
                rel_pos,
            ]
        )

    def render(self, trajectory, height, width, scene_option=None, camera=None):
        return super().render(
            trajectory=trajectory,
            height=height,
            width=width,
            scene_option=scene_option,
            camera=camera or "closeup",
        )


def default_config() -> config_dict.ConfigDict:
    """Default configuration for the TricoDriver environment."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.001,
        episode_length=500,
        action_repeat=1,
        rotate_reward_scale=2.5,
        symmetry_penalty_scale=0.1,
        tip_penetration_penalty_scale=150.0,
        allowed_tip_penetration=0.005,
        delta_action_penalty_scale=0.0,
    )

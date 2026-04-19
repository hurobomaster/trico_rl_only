"""Trico reach-only task with simplified observations and reward."""

import os

import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env


DEBUG_TRICO_DRIVER_SINGLE_REACH_OBS = False


class TricoDriverSingleReachEnv(mjx_env.MjxEnv):
  """Reach-only Trico task with mirrored 4-D actions for both hands."""

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
      config.episode_length = 1000
    if "action_repeat" not in config:
      config.action_repeat = 1
    if "target_radius" not in config:
      config.target_radius = 0.04
    if "reach_reward_scale" not in config:
      config.reach_reward_scale = 1.0
    if "reach_distance_scale" not in config:
      config.reach_distance_scale = 25.0
    if "tip_penetration_penalty_scale" not in config:
      config.tip_penetration_penalty_scale = 150.0
    if "allowed_tip_penetration" not in config:
      config.allowed_tip_penetration = 0.005
    if "delta_action_penalty_scale" not in config:
      config.delta_action_penalty_scale = 0.0
    if "max_joint_delta_deg_per_step" not in config:
      config.max_joint_delta_deg_per_step = 1.0
    if "joint_speed_penalty_scale" not in config:
      config.joint_speed_penalty_scale = 5.0
    if "hand_init_noise_scale" not in config:
      config.hand_init_noise_scale = 0.02
    if "action_min_delay" not in config:
      config.action_min_delay = 0
    if "action_max_delay" not in config:
      config.action_max_delay = 0
    if "action_history_len" not in config:
      config.action_history_len = 6
    if "obs_noise_std" not in config:
      config.obs_noise_std = 0.0
    if "action_noise_std" not in config:
      config.action_noise_std = 0.0

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

    self._target_radius = float(config.target_radius)
    self._reach_reward_scale = float(config.reach_reward_scale)
    self._reach_distance_scale = float(config.reach_distance_scale)
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
    self._max_joint_delta_per_step = jnp.deg2rad(
      float(config.max_joint_delta_deg_per_step)
    )
    self._joint_speed_penalty_scale = float(config.joint_speed_penalty_scale)
    self._hand_init_noise_scale = float(config.hand_init_noise_scale)
    self._action_min_delay = int(config.action_min_delay)
    self._action_max_delay = int(config.action_max_delay)
    self._action_history_len = int(config.action_history_len)
    self._obs_noise_std = float(config.obs_noise_std)
    self._action_noise_std = float(config.action_noise_std)
    if self._action_history_len <= 0:
      raise ValueError("action_history_len must be positive")
    if self._action_min_delay < 0:
      raise ValueError("action_min_delay must be >= 0")
    if self._action_max_delay < self._action_min_delay:
      raise ValueError("action_max_delay must be >= action_min_delay")
    if self._action_max_delay >= self._action_history_len:
      raise ValueError("action_max_delay must be smaller than action_history_len")

    self._single_action_size = 4
    self._obs_size = 14

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._single_action_size

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = jnp.asarray(self.mjx_model.qpos0)
    rng, hand_key = jax.random.split(rng)
    hand_noise_single = jax.random.uniform(
        hand_key,
        (self._single_action_size,),
        minval=-self._hand_init_noise_scale,
        maxval=self._hand_init_noise_scale,
    )
    hand_noise = jnp.concatenate([hand_noise_single, hand_noise_single])
    qpos = qpos.at[self._actuator_qpos_adr].add(hand_noise)
    qpos = qpos.at[self._driver_qpos_adr].set(0.0)
    qvel = jnp.zeros(self.mjx_model.nv)
    init_ctrl = self._init_ctrl_qpos + hand_noise
    init_ctrl = jnp.clip(init_ctrl, self._ctrlrange[:, 0], self._ctrlrange[:, 1])

    data = mjx.make_data(self.mjx_model)
    data = data.replace(qpos=qpos, qvel=qvel, ctrl=init_ctrl)
    data = mjx.forward(self.mjx_model, data)

    obs = self._get_obs(data)
    reward = jnp.array(0.0, dtype=jnp.float32)
    done = jnp.array(0.0, dtype=jnp.float32)
    metrics = {
        "reward_reach": jnp.array(0.0, dtype=jnp.float32),
        "penalty_tip_penetration": jnp.array(0.0, dtype=jnp.float32),
        "penalty_delta_action": jnp.array(0.0, dtype=jnp.float32),
      "penalty_joint_speed": jnp.array(0.0, dtype=jnp.float32),
        "mean_tip_dist": jnp.array(0.0, dtype=jnp.float32),
        "driver_joint_pos": jnp.array(0.0, dtype=jnp.float32),
        "driver_joint_vel": jnp.array(0.0, dtype=jnp.float32),
        "reward": jnp.array(0.0, dtype=jnp.float32),
    }
    info = {
        "rng": rng,
        "last_action": jnp.zeros(self.action_size, dtype=jnp.float32),
      "last_qpos_act": data.qpos[self._actuator_qpos_adr],
        "action_history": jnp.zeros(
            self._action_history_len * self.action_size, dtype=jnp.float32
        ),
    }
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action_history = jnp.roll(state.info["action_history"], self.action_size).at[
        : self.action_size
    ].set(action)
    rng, delay_key = jax.random.split(state.info["rng"])
    delay_idx = jax.random.randint(
        delay_key,
        shape=(),
        minval=self._action_min_delay,
        maxval=self._action_max_delay + 1,
    )
    action_w_delay = action_history.reshape(
        (self._action_history_len, self.action_size)
    )[delay_idx]

    rng, action_noise_key = jax.random.split(rng)
    action_noise = (
        jax.random.normal(action_noise_key, shape=action_w_delay.shape)
        * self._action_noise_std
    )
    action_w_delay = action_w_delay + action_noise

    mirrored_action = jnp.concatenate([action_w_delay, action_w_delay])
    ctrl_center = (self._ctrlrange[:, 0] + self._ctrlrange[:, 1]) * 0.5
    ctrl_half = (self._ctrlrange[:, 1] - self._ctrlrange[:, 0]) * 0.5
    motor_targets = ctrl_center + mirrored_action * ctrl_half
    motor_targets = jnp.clip(
        motor_targets, self._ctrlrange[:, 0], self._ctrlrange[:, 1]
    )

    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

    driver_joint_pos = data.qpos[self._driver_qpos_adr]
    driver_joint_vel_signed = data.qvel[self._driver_qvel_adr]
    handle_center = data.site_xpos[self._driver_center_site_id]
    tip_right = data.site_xpos[self._tip_right_id]
    tip_left = data.site_xpos[self._tip_left_id]

    tip_dists = jnp.array(
        [
            jnp.linalg.norm(tip_right - handle_center),
            jnp.linalg.norm(tip_left - handle_center),
        ]
    )
    dist_outside_target = jnp.maximum(0.0, tip_dists - self._target_radius)
    mean_outside_dist = jnp.mean(dist_outside_target)

    # Reward is saturated once tips are inside a radius-4cm target sphere.
    reach_reward = (
        self._reach_reward_scale
        * jnp.exp(-self._reach_distance_scale * mean_outside_dist)
    )

    tip_penetration = jnp.maximum(0.0, self._min_tip_center_dist - tip_dists)
    penetration_penalty = self._penetration_penalty_scale * jnp.sum(
        tip_penetration
    )

    delta_action = action_w_delay - state.info["last_action"]
    delta_action_penalty = self._delta_action_penalty_scale * jnp.mean(
        jnp.square(delta_action)
    )

    qpos_act_curr = data.qpos[self._actuator_qpos_adr]
    joint_delta = jnp.abs(qpos_act_curr - state.info["last_qpos_act"])
    joint_delta_excess = jnp.maximum(0.0, joint_delta - self._max_joint_delta_per_step)
    joint_speed_penalty = self._joint_speed_penalty_scale * jnp.mean(
      jnp.square(joint_delta_excess)
    )

    obs = self._get_obs(data)
    if self._obs_noise_std > 0.0:
      rng, obs_noise_key = jax.random.split(rng)
      obs_noise = (
          jax.random.normal(obs_noise_key, shape=obs.shape) * self._obs_noise_std
      )
      obs = obs + obs_noise

    is_invalid = ~jnp.all(jnp.isfinite(obs))
    total_reward = (
      reach_reward
      - penetration_penalty
      - delta_action_penalty
      - joint_speed_penalty
    )
    reward = jnp.where(is_invalid, 0.0, total_reward)
    done = is_invalid.astype(jnp.float32)

    metrics = {
        "reward_reach": reach_reward,
        "penalty_tip_penetration": penetration_penalty,
        "penalty_delta_action": delta_action_penalty,
        "penalty_joint_speed": joint_speed_penalty,
        "mean_tip_dist": jnp.mean(tip_dists),
        "driver_joint_pos": driver_joint_pos,
        "driver_joint_vel": driver_joint_vel_signed,
        "reward": reward,
    }

    if DEBUG_TRICO_DRIVER_SINGLE_REACH_OBS:
      jax.debug.print(
          (
              "mean_tip_dist={dist:.4f} outside={outside:.4f} "
              "r_reach={rreach:.3f} p_tip={ptip:.3f} "
                "p_delta={pdelta:.3f} p_joint={pjoint:.3f} reward={rew:.3f}"
          ),
          dist=jnp.mean(tip_dists),
          outside=mean_outside_dist,
          rreach=reach_reward,
          ptip=penetration_penalty,
          pdelta=delta_action_penalty,
              pjoint=joint_speed_penalty,
          rew=reward,
      )

    info = {
        **state.info,
        "rng": rng,
        "last_action": action_w_delay,
        "last_qpos_act": qpos_act_curr,
        "action_history": action_history,
    }

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
    tip_right = data.site_xpos[self._tip_right_id]
    tip_left = data.site_xpos[self._tip_left_id]
    handle_center = data.site_xpos[self._driver_center_site_id]
    rel_right = handle_center - tip_right
    rel_left = handle_center - tip_left
    return jnp.concatenate([qpos_act, rel_right, rel_left])

  def render(self, trajectory, height, width, scene_option=None, camera=None):
    return super().render(
        trajectory=trajectory,
        height=height,
        width=width,
        scene_option=scene_option,
        camera=camera or "closeup",
    )


def default_config() -> config_dict.ConfigDict:
  """Default configuration for the TricoDriverSingleReach environment."""
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.001,
      episode_length=1000,
      action_repeat=1,
      target_radius=0.04,
      reach_reward_scale=1.0,
      reach_distance_scale=25.0,
      tip_penetration_penalty_scale=150.0,
      allowed_tip_penetration=0.005,
      delta_action_penalty_scale=0.0,
      max_joint_delta_deg_per_step=1.0,
      joint_speed_penalty_scale=5.0,
      hand_init_noise_scale=0.02,
      action_min_delay=0,
      action_max_delay=0,
      action_history_len=6,
      obs_noise_std=0.0,
      action_noise_std=0.0,
  )

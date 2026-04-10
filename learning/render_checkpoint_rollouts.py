"""Render rollout videos for saved checkpoints without creating new log dirs."""

import argparse
import functools
import json
import os
from pathlib import Path

# Configure MuJoCo before importing rendering libraries.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
import numpy as np
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params


TRICO_SUCCESS_ROTATE_THRESHOLD = 2000.0
TRICO_SUCCESS_TIP_PENALTY_THRESHOLD = 1.0
TRICO_SUCCESS_TIP_Y_THRESHOLD = 1.0


_ORIG_MJMODEL_FROM_XML_PATH = mujoco.MjModel.from_xml_path


def _get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.manipulation._envs:
    return manipulation_params.brax_ppo_config(env_name, "jax")
  if env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, "jax")
  if env_name in mujoco_playground.dm_control_suite._envs:
    return dm_control_suite_params.brax_ppo_config(env_name, "jax")
  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def _maybe_override_trico_driver_xml(env_name: str, xml_name: str | None) -> None:
  if env_name not in {"TricoDriver", "TricoDriverSingle"} or not xml_name:
    mujoco.MjModel.from_xml_path = _ORIG_MJMODEL_FROM_XML_PATH
    return

  xml_arg = Path(xml_name)
  if not xml_arg.is_absolute():
    repo_root = Path(__file__).resolve().parents[1]
    xml_arg = (
        repo_root
        / "mujoco_playground"
        / "_src"
        / "manipulation"
        / "trico"
        / "xmls"
        / xml_arg
    )
  xml_path = xml_arg.resolve()
  if not xml_path.exists():
    raise FileNotFoundError(f"Requested TricoDriver XML not found: {xml_path}")

  print(f"Using TricoDriver scene XML: {xml_path}")

  def _patched_from_xml_path(path, *args, **kwargs):
    basename = Path(path).name
    if basename == "trico_driver_base.xml" or basename.startswith("trico_driver_v"):
      path = str(xml_path)
    return _ORIG_MJMODEL_FROM_XML_PATH(path, *args, **kwargs)

  mujoco.MjModel.from_xml_path = _patched_from_xml_path


def _resolve_checkpoints(path: Path) -> tuple[Path, list[Path]]:
  path = path.resolve()
  if not path.exists():
    raise FileNotFoundError(f"Checkpoint path not found: {path}")

  if path.is_dir() and path.name.isdigit():
    return path.parent, [path]

  if path.is_dir() and (path / "checkpoints").is_dir():
    path = path / "checkpoints"

  if not path.is_dir():
    raise ValueError(f"Expected a checkpoint dir or experiment dir, got: {path}")

  ckpt_dirs = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
  ckpt_dirs.sort(key=lambda p: int(p.name))
  if not ckpt_dirs:
    raise ValueError(f"No numeric checkpoint subdirectories found under: {path}")
  return path, ckpt_dirs


def _load_env_config(env_name: str, checkpoints_root: Path, episode_length: int | None):
  env_cfg = registry.get_default_config(env_name)
  env_cfg["impl"] = "jax"

  config_path = checkpoints_root / "config.json"
  if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as fp:
      saved_cfg = json.load(fp)
    for key, value in saved_cfg.items():
      env_cfg[key] = value

  if episode_length is not None:
    env_cfg["episode_length"] = episode_length

  return env_cfg


def _load_policy(env_name: str, env_cfg, checkpoint_dir: Path):
  env = registry.load(env_name, config=env_cfg)
  ppo_params = _get_rl_config(env_name)

  network_fn = ppo_networks.make_ppo_networks
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(network_fn, **ppo_params.network_factory)
  else:
    network_factory = network_fn

  train_kwargs = dict(ppo_params)
  train_kwargs.pop("network_factory", None)
  
  # Determine num_envs based on device count (must be divisible)
  import jax
  device_count = jax.device_count()
  num_envs = max(1, device_count)  # Use device_count or 1, whichever is larger
  
  # For inference-only mode, use simple compatible settings
  # The constraint is: batch_size * num_minibatches % num_envs == 0
  batch_size = num_envs  # Make batch_size compatible with num_envs
  num_minibatches = 1
  
  train_kwargs.update(
      num_timesteps=0,
      num_envs=num_envs,
      num_eval_envs=num_envs,
      batch_size=batch_size,
      num_minibatches=num_minibatches,
      num_updates_per_batch=1,
      unroll_length=1,
      log_training_metrics=False,
      run_evals=False,
      restore_checkpoint_path=checkpoint_dir,
      save_checkpoint_path=None,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      seed=1,
  )

  make_inference_fn, params, _ = ppo.train(
      environment=env,
      network_factory=network_factory,
      **train_kwargs,
  )
  return env, make_inference_fn, params


def _sensor_slice(model: mujoco.MjModel, sensor_name: str):
  sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
  if sensor_id < 0:
    return None
  sensor_adr = int(model.sensor_adr[sensor_id])
  sensor_dim = int(model.sensor_dim[sensor_id])
  return sensor_adr, sensor_adr + sensor_dim


def _prepare_trico_plot_handles(env):
  if env.__class__.__name__ not in {"TricoDriverEnv", "TricoDriverSingleEnv"}:
    return None

  model = env.mj_model
  right_force_slice = _sensor_slice(model, "sensor_F_right")
  left_force_slice = _sensor_slice(model, "sensor_F_left")
  return dict(
      actuator_qpos_adr=np.asarray(env._actuator_qpos_adr),
      driver_qpos_adr=int(env._driver_qpos_adr),
      driver_qvel_adr=int(env._driver_qvel_adr),
      tip_right_id=int(env._tip_right_id),
      tip_left_id=int(env._tip_left_id),
      handle_center_id=int(env._driver_center_site_id),
      min_tip_center_dist=float(getattr(env, "_min_tip_center_dist", 0.0)),
      right_force_slice=right_force_slice,
      left_force_slice=left_force_slice,
      has_tip_y_penalty=env.__class__.__name__ == "TricoDriverSingleEnv",
  )


def _save_rollout_visualizations(
    checkpoint_dir: Path,
    stem: str,
    diag: dict,
    fps: float,
) -> None:
  vis_dir = checkpoint_dir / "vis"
  vis_dir.mkdir(exist_ok=True)

  raw_path = vis_dir / f"{stem}_raw.npz"
  np.savez(raw_path, **diag)

  t = np.arange(diag["reward"].shape[0]) / fps

  if "driver_joint_vel" not in diag:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, diag["reward"], label="reward")
    if diag["policy_action"].ndim == 2:
      for idx in range(diag["policy_action"].shape[1]):
        ax.plot(t, diag["policy_action"][:, idx], alpha=0.6, label=f"action_{idx}")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right")
    ax.set_title("Generic Rollout Diagnostics")
    fig.tight_layout()
    summary_path = vis_dir / f"{stem}_summary.png"
    fig.savefig(summary_path, dpi=180)
    plt.close(fig)
    print(f"Saved rollout diagnostics: {summary_path}")
    print(f"Saved rollout raw arrays: {raw_path}")
    return

  summary_fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

  axes[0].plot(t, diag["driver_joint_vel"], label="driver vel [rad/s]")
  axes[0].plot(t, diag["reward_rotate"], label="rotate reward")
  axes[0].plot(t, diag["reward"], label="total reward", alpha=0.8)
  axes[0].set_ylabel("velocity / reward")
  axes[0].legend(loc="upper right")
  axes[0].set_title("Driver Rotation and Reward")

  axes[1].plot(t, diag["tip_right_penetration_mm"], label="right pen [mm]")
  axes[1].plot(t, diag["tip_left_penetration_mm"], label="left pen [mm]")
  axes[1].plot(
      t,
      diag["penalty_tip_penetration"],
      label="tip penetration penalty",
      alpha=0.8,
  )
  axes[1].plot(t, diag["penalty_tip_y"], label="tip y penalty", alpha=0.8)
  axes[1].set_ylabel("penetration / penalty")
  axes[1].legend(loc="upper right")
  axes[1].set_title("Penetration Diagnostics")

  axes[2].plot(t, diag["tip_right_y_mm"], label="tip_right y [mm]")
  axes[2].plot(t, diag["tip_left_y_mm"], label="tip_left y [mm]")
  axes[2].axhline(10.0, linestyle="--", color="tab:red", alpha=0.6)
  axes[2].axhline(-10.0, linestyle="--", color="tab:blue", alpha=0.6)
  axes[2].set_ylabel("tip y [mm]")
  axes[2].legend(loc="upper right")
  axes[2].set_title("Lateral Tip Position")

  axes[3].plot(t, diag["force_right_norm"], label="right force norm")
  axes[3].plot(t, diag["force_left_norm"], label="left force norm")
  axes[3].plot(t, diag["force_right_x"], label="right Fx", alpha=0.6)
  axes[3].plot(t, diag["force_left_x"], label="left Fx", alpha=0.6)
  axes[3].set_ylabel("force")
  axes[3].legend(loc="upper right", ncol=2)
  axes[3].set_title("Tip Force Diagnostics")

  axes[4].plot(t, diag["penalty_delta_action"], label="delta penalty")
  axes[4].plot(t, diag["driver_joint_pos"], label="driver joint pos")
  axes[4].set_xlabel("time [s]")
  axes[4].set_ylabel("penalty / pos")
  axes[4].legend(loc="upper right")
  axes[4].set_title("Delta Action and Driver Joint Position")

  summary_fig.tight_layout()
  summary_path = vis_dir / f"{stem}_summary.png"
  summary_fig.savefig(summary_path, dpi=180)
  plt.close(summary_fig)

  act_fig, act_axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
  control_labels = ["OA_x", "OA_y", "OB_x", "OB_y"]
  for idx, ax in enumerate(act_axes):
    ax.plot(t, diag["policy_action"][:, idx], label=f"policy {control_labels[idx]}")
    ax.plot(t, diag["ctrl"][:, idx], label=f"ctrl right {control_labels[idx]}")
    ax.plot(t, diag["ctrl"][:, idx + 4], label=f"ctrl left {control_labels[idx]}")
    ax.plot(
        t,
        diag["qpos_act"][:, idx],
        label=f"qpos right {control_labels[idx]}",
        alpha=0.8,
    )
    ax.plot(
        t,
        diag["qpos_act"][:, idx + 4],
        label=f"qpos left {control_labels[idx]}",
        alpha=0.8,
    )
    ax.set_ylabel(control_labels[idx])
    ax.legend(loc="upper right", ncol=3, fontsize=8)
  act_axes[-1].set_xlabel("time [s]")
  act_fig.suptitle("Policy Action vs Control Targets vs Actuated Joint State")
  act_fig.tight_layout()
  act_path = vis_dir / f"{stem}_actions_state.png"
  act_fig.savefig(act_path, dpi=180)
  plt.close(act_fig)

  print(f"Saved rollout diagnostics: {summary_path}")
  print(f"Saved rollout diagnostics: {act_path}")
  print(f"Saved rollout raw arrays: {raw_path}")


def _compute_success_summary(diag: dict, episode_length: int) -> dict:
  rotate_reward = float(np.sum(diag["reward_rotate"]))
  tip_penalty = float(np.sum(diag["penalty_tip_penetration"]))
  tip_y_penalty = float(np.sum(diag["penalty_tip_y"]))
  
  # Handle case where contact_any may not be in diag
  if "contact_any" in diag:
    contact_any_ratio = float(np.mean(diag["contact_any"]))
  else:
    contact_any_ratio = 0.0

  success = (
      rotate_reward >= TRICO_SUCCESS_ROTATE_THRESHOLD
      and tip_penalty <= TRICO_SUCCESS_TIP_PENALTY_THRESHOLD
      and tip_y_penalty <= TRICO_SUCCESS_TIP_Y_THRESHOLD
  )

  reasons = []
  if rotate_reward < TRICO_SUCCESS_ROTATE_THRESHOLD:
    reasons.append(
        f"rotate_reward<{TRICO_SUCCESS_ROTATE_THRESHOLD:.1f}"
    )
  if tip_penalty > TRICO_SUCCESS_TIP_PENALTY_THRESHOLD:
    reasons.append(f"tip_penalty>{TRICO_SUCCESS_TIP_PENALTY_THRESHOLD:.1f}")
  if tip_y_penalty > TRICO_SUCCESS_TIP_Y_THRESHOLD:
    reasons.append(f"tip_y_penalty>{TRICO_SUCCESS_TIP_Y_THRESHOLD:.1f}")

  return {
      "success": bool(success),
      "rotate_reward": rotate_reward,
      "tip_penalty": tip_penalty,
      "tip_y_penalty": tip_y_penalty,
      "contact_any_ratio": contact_any_ratio,
      "episode_length": int(episode_length),
      "thresholds": {
          "rotate_reward": TRICO_SUCCESS_ROTATE_THRESHOLD,
          "tip_penalty": TRICO_SUCCESS_TIP_PENALTY_THRESHOLD,
          "tip_y_penalty": TRICO_SUCCESS_TIP_Y_THRESHOLD,
      },
      "reasons": reasons,
  }


def _render_rollout(
    env,
    make_inference_fn,
    params,
    episode_length: int,
    num_videos: int,
    render_every: int,
    width: int,
    height: int,
    seed: int,
):
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)
  plot_handles = _prepare_trico_plot_handles(env)

  def do_rollout(rng, state):
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_key)[0]
      state = env.step(state, act)
      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.time": state.data.time,
          "data.ctrl": state.data.ctrl,
          "data.mocap_pos": state.data.mocap_pos,
          "data.mocap_quat": state.data.mocap_quat,
          "data.xfrc_applied": state.data.xfrc_applied,
      })
      if plot_handles is None:
        diag = {
            "reward": state.reward,
            "policy_action": act,
        }
      else:
        tip_right = state.data.site_xpos[plot_handles["tip_right_id"]]
        tip_left = state.data.site_xpos[plot_handles["tip_left_id"]]
        handle_center = state.data.site_xpos[plot_handles["handle_center_id"]]
        right_dist = jnp.linalg.norm(tip_right - handle_center)
        left_dist = jnp.linalg.norm(tip_left - handle_center)
        right_pen = jnp.maximum(
            0.0, plot_handles["min_tip_center_dist"] - right_dist
        )
        left_pen = jnp.maximum(
            0.0, plot_handles["min_tip_center_dist"] - left_dist
        )

        if plot_handles["right_force_slice"] is not None:
          r0, r1 = plot_handles["right_force_slice"]
          force_right = state.data.sensordata[r0:r1]
        else:
          force_right = jnp.zeros(3)

        if plot_handles["left_force_slice"] is not None:
          l0, l1 = plot_handles["left_force_slice"]
          force_left = state.data.sensordata[l0:l1]
        else:
          force_left = jnp.zeros(3)

        tip_y_penalty = (
            state.metrics["penalty_tip_y"]
            if plot_handles["has_tip_y_penalty"]
            else jnp.array(0.0, dtype=jnp.float32)
        )

        diag = {
            "reward": state.reward,
            "reward_rotate": state.metrics["reward_rotate"],
          "contact_any": state.metrics["contact_any"],
          "contact_right": state.metrics["contact_right"],
          "contact_left": state.metrics["contact_left"],
            "penalty_tip_penetration": state.metrics["penalty_tip_penetration"],
            "penalty_delta_action": state.metrics["penalty_delta_action"],
            "penalty_tip_y": tip_y_penalty,
            "driver_joint_pos": state.metrics["driver_joint_pos"],
            "driver_joint_vel": state.metrics["driver_joint_vel"],
            "policy_action": act,
            "ctrl": state.data.ctrl,
            "qpos_act": state.data.qpos[plot_handles["actuator_qpos_adr"]],
            "tip_right_y_mm": tip_right[1] * 1000.0,
            "tip_left_y_mm": tip_left[1] * 1000.0,
            "tip_right_penetration_mm": right_pen * 1000.0,
            "tip_left_penetration_mm": left_pen * 1000.0,
            "force_right_x": force_right[0],
            "force_left_x": force_left[0],
            "force_right_norm": jnp.linalg.norm(force_right),
            "force_left_norm": jnp.linalg.norm(force_left),
        }
      return (state, rng), (traj_data, diag)

    _, (traj, diag) = jax.lax.scan(step, (state, rng), None, length=episode_length)
    return traj, diag

  rng = jax.random.split(jax.random.PRNGKey(seed), num_videos)
  reset_states = jax.jit(jax.vmap(env.reset))(rng)
  traj_stacked, diag_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)

  trajectories = []
  diagnostics = []
  for i in range(num_videos):
    traj_i = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    diag_i = jax.tree.map(lambda x, i=i: np.asarray(x[i]), diag_stacked)
    trajectories.append(
        [jax.tree.map(lambda x, j=j: x[j], traj_i) for j in range(episode_length)]
    )
    diagnostics.append(diag_i)

  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  fps = 1.0 / env.dt / render_every
  videos = []
  for rollout in trajectories:
    traj = rollout[::render_every]
    frames = env.render(
        traj,
        height=height,
        width=width,
        scene_option=scene_option,
    )
    videos.append((frames, fps))
  return videos, diagnostics


def _save_success_summary(checkpoint_dir: Path, stem: str, summary: dict) -> None:
  vis_dir = checkpoint_dir / "vis"
  vis_dir.mkdir(exist_ok=True)
  summary_path = vis_dir / f"{stem}_success.json"
  with open(summary_path, "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)
  print(f"Saved success summary: {summary_path}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--env_name", required=True)
  parser.add_argument(
      "--checkpoint_path",
      required=True,
      help="Experiment dir, checkpoints dir, or a single numeric checkpoint dir.",
  )
  parser.add_argument("--trico_driver_xml", default=None)
  parser.add_argument("--episode_length", type=int, default=1000)
  parser.add_argument("--num_videos", type=int, default=1)
  parser.add_argument("--render_every", type=int, default=2)
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--output_name", default="rollout.mp4")
  parser.add_argument(
      "--latest_only",
      action="store_true",
      help="Render only the most recent numeric checkpoint.",
  )
  args = parser.parse_args()

  _maybe_override_trico_driver_xml(args.env_name, args.trico_driver_xml)
  checkpoints_root, checkpoint_dirs = _resolve_checkpoints(Path(args.checkpoint_path))
  if args.latest_only:
    checkpoint_dirs = [checkpoint_dirs[-1]]

  env_cfg = _load_env_config(
      args.env_name, checkpoints_root, episode_length=args.episode_length
  )

  print(f"Rendering {len(checkpoint_dirs)} checkpoint(s) from: {checkpoints_root}")
  all_successes = []
  for checkpoint_dir in checkpoint_dirs:
    print(f"Loading checkpoint: {checkpoint_dir}")
    env, make_inference_fn, params = _load_policy(
        args.env_name, env_cfg, checkpoint_dir
    )
    videos, diagnostics = _render_rollout(
        env=env,
        make_inference_fn=make_inference_fn,
        params=params,
        episode_length=args.episode_length,
        num_videos=args.num_videos,
        render_every=args.render_every,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    if args.num_videos == 1:
      output_path = checkpoint_dir / args.output_name
      media.write_video(output_path, videos[0][0], fps=videos[0][1])
      print(f"Saved rollout video: {output_path}")
      _save_rollout_visualizations(
          checkpoint_dir,
          Path(args.output_name).stem,
          diagnostics[0],
          fps=videos[0][1],
      )
      success_summary = _compute_success_summary(
        diagnostics[0], args.episode_length
      )
      _save_success_summary(
        checkpoint_dir, Path(args.output_name).stem, success_summary
      )
      all_successes.append(success_summary["success"])
      print(
        "Success verdict: "
        f"{success_summary['success']} | "
        f"rotate_reward={success_summary['rotate_reward']:.2f} | "
        f"tip_penalty={success_summary['tip_penalty']:.2f} | "
        f"tip_y_penalty={success_summary['tip_y_penalty']:.2f} | "
        f"contact_ratio={success_summary['contact_any_ratio']:.3f}"
      )
      continue

    stem = Path(args.output_name).stem
    suffix = Path(args.output_name).suffix or ".mp4"
    for idx, ((frames, fps), diag) in enumerate(zip(videos, diagnostics)):
      output_path = checkpoint_dir / f"{stem}{idx}{suffix}"
      media.write_video(output_path, frames, fps=fps)
      print(f"Saved rollout video: {output_path}")
      _save_rollout_visualizations(
          checkpoint_dir,
          f"{stem}{idx}",
          diag,
          fps=fps,
      )
      success_summary = _compute_success_summary(diag, args.episode_length)
      _save_success_summary(checkpoint_dir, f"{stem}{idx}", success_summary)
      all_successes.append(success_summary["success"])
      print(
        "Success verdict: "
        f"{success_summary['success']} | "
        f"rotate_reward={success_summary['rotate_reward']:.2f} | "
        f"tip_penalty={success_summary['tip_penalty']:.2f} | "
        f"tip_y_penalty={success_summary['tip_y_penalty']:.2f} | "
        f"contact_ratio={success_summary['contact_any_ratio']:.3f}"
      )

    if all_successes:
      print(
        f"Overall success rate: {sum(all_successes)}/{len(all_successes)} "
        f"({sum(all_successes) / len(all_successes):.1%})"
      )


if __name__ == "__main__":
  main()

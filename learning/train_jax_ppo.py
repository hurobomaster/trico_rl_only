# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
from pathlib import Path
import re
import sys
import time
import warnings

# Configure JAX/MuJoCo before importing libraries that read these env vars.
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import tensorboardX
from tqdm import tqdm


def _import_wandb():
  """Imports the real wandb package even if a local `wandb/` folder exists."""
  import importlib

  repo_root = str(Path(__file__).resolve().parents[1])
  original_sys_path = list(sys.path)
  try:
    wandb_mod = importlib.import_module("wandb")
    if hasattr(wandb_mod, "init"):
      return wandb_mod

    # Local logs folder `wandb/` can shadow the pip package when running from repo root.
    sys.modules.pop("wandb", None)
    sys.path = [p for p in sys.path if p not in ("", repo_root)]
    importlib.invalidate_caches()
    wandb_mod = importlib.import_module("wandb")
    if hasattr(wandb_mod, "init"):
      return wandb_mod
    raise ImportError("Imported module 'wandb' does not provide init().")
  finally:
    sys.path = original_sys_path


wandb = _import_wandb()

_ORIG_MJMODEL_FROM_XML_PATH = mujoco.MjModel.from_xml_path

# 导入 Trico 真机运动学库（确保 Sim2Real 一致性）
# try:
#     from trico_control import FingerKinematics, FingerInverseSolver
#     TRICO_KINEMATICS_AVAILABLE = True
#     print("✓ Trico 真机运动学库已加载")
# except ImportError:
#     TRICO_KINEMATICS_AVAILABLE = False
#     print("✗ 警告: Trico 真机运动学库未找到，训练将使用纯仿真运动学")  

# Ignore the info logs from brax


# logging.set_verbosity(logging.WARNING)   原来的代码

# 1. 屏蔽 Python/ABSL 警告，只显示严重错误
logging.set_verbosity(logging.ERROR)

# 2. 屏蔽底层的 C++ / TensorFlow 警告





# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
  "TricoDriverSingle",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name",
    None,
    "Optional custom experiment label appended after the environment name.",
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
  True,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", True, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_RENDER_ROLLOUTS = flags.DEFINE_boolean(
    "render_rollouts",
    False,
    "Render rollout videos after training. Disable on headless servers.",
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNINGRATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    True,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_DETERMINISTIC_EVAL = flags.DEFINE_boolean(
    "deterministic_eval",
    False,
    "Use deterministic policy during evaluation (use mean action, not stochastic)."
    " Helps prevent high cross-episode std and allows eval to match deterministic renders.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)
_TRICO_DRIVER_XML = flags.DEFINE_string(
    "trico_driver_xml",
  "trico_driver_v3.xml",
    "Optional XML filename or absolute path for TricoDriver scene selection.",
)
_TRICO_ROTATE_REWARD_SCALE = flags.DEFINE_float(
    "trico_rotate_reward_scale",
    None,
    "Override TricoDriver rotation reward scale inside the environment.",
)
_TRICO_SYMMETRY_PENALTY_SCALE = flags.DEFINE_float(
    "trico_symmetry_penalty_scale",
    None,
    "Override TricoDriver left/right symmetry penalty scale inside the environment.",
)
_TRICO_TIP_PENALTY_SCALE = flags.DEFINE_float(
    "trico_tip_penalty_scale",
    None,
    "Override TricoDriver fingertip penetration penalty scale inside the environment.",
)
_TRICO_ALLOWED_TIP_PENETRATION = flags.DEFINE_float(
    "trico_allowed_tip_penetration",
    None,
    "Override TricoDriver allowed fingertip penetration depth in meters.",
)
_TRICO_DELTA_ACTION_PENALTY_SCALE = flags.DEFINE_float(
    "trico_delta_action_penalty_scale",
    None,
    "Override TricoDriver delta-action penalty scale inside the environment.",
)
_TRICO_TIP_Y_PENALTY_SCALE = flags.DEFINE_float(
    "trico_tip_y_penalty_scale",
    None,
    "Override TricoDriver fingertip y-bound penalty scale inside the environment.",
)
_TRICO_OBS_HISTORY_LEN = flags.DEFINE_integer(
  "trico_obs_history_len",
  None,
  "Override TricoDriverSingle stacked observation history length.",
)
_TRICO_CONTACT_FORCE_THRESHOLD = flags.DEFINE_float(
  "trico_contact_force_threshold",
  None,
  "Override TricoDriverSingle force threshold used for binary contacts.",
)
_TRICO_CONTACT_REWARD_SCALE = flags.DEFINE_float(
  "trico_contact_reward_scale",
  None,
  "Override TricoDriverSingle binary-contact bonus scale.",
)
_TRICO_REACH_REWARD_SCALE = flags.DEFINE_float(
  "trico_reach_reward_scale",
  None,
  "Override TricoDriverSingle dense reach reward scale.",
)
_TRICO_REACH_DISTANCE_SCALE = flags.DEFINE_float(
  "trico_reach_distance_scale",
  None,
  "Override TricoDriverSingle reach distance exponential scale.",
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


def _maybe_override_trico_driver_xml() -> None:
  """Optionally redirects TricoDriver XML loading for this process."""
  if (
      _ENV_NAME.value not in {"TricoDriver", "TricoDriverSingle"}
      or not _TRICO_DRIVER_XML.value
  ):
    mujoco.MjModel.from_xml_path = _ORIG_MJMODEL_FROM_XML_PATH
    return

  xml_arg = Path(_TRICO_DRIVER_XML.value)
  if xml_arg.is_absolute():
    xml_path = xml_arg
  else:
    repo_root = Path(__file__).resolve().parents[1]
    xml_path = (
        repo_root
        / "mujoco_playground"
        / "_src"
        / "manipulation"
        / "trico"
        / "xmls"
        / xml_arg
    )

  if not xml_path.exists():
    raise FileNotFoundError(f"Requested TricoDriver XML not found: {xml_path}")

  target_xml = xml_path.resolve()
  print(f"Using TricoDriver scene XML: {target_xml}")

  def _patched_from_xml_path(path, *args, **kwargs):
    path_obj = Path(path)
    basename = path_obj.name
    if basename == "trico_driver_base.xml" or basename.startswith("trico_driver_v"):
      path = str(target_xml)
    return _ORIG_MJMODEL_FROM_XML_PATH(path, *args, **kwargs)

  mujoco.MjModel.from_xml_path = _patched_from_xml_path


def _sanitize_name_component(value: str) -> str:
  """Sanitizes a path component so experiment directories remain predictable."""
  sanitized = re.sub(r'[\\/:*?"<>|\s]+', "_", value.strip())
  sanitized = sanitized.strip("._-")
  return sanitized


def _get_experiment_label() -> str:
  """Gets a user-provided label from flags or an interactive prompt."""
  raw_label = _EXPERIMENT_NAME.value
  if raw_label is None and not _PLAY_ONLY.value and sys.stdin.isatty():
    print("\n" + "!" * 60)
    print(f"当前环境: {_ENV_NAME.value}")
    try:
      raw_label = input(
          "请输入本次实验名称 (直接回车将仅使用环境名): "
      ).strip()
    except EOFError:
      raw_label = ""
    print("!" * 60 + "\n")

  if not raw_label:
    return ""

  sanitized_label = _sanitize_name_component(raw_label)
  if not sanitized_label:
    print("Experiment label only contained unsupported characters; ignoring it.")
    return ""
  if sanitized_label != raw_label:
    print(f"Sanitized experiment label: {sanitized_label}")
  return sanitized_label


def _build_experiment_name(timestamp: str, experiment_label: str) -> str:
  """Builds a stable experiment directory name."""
  name_parts = [_sanitize_name_component(_ENV_NAME.value) or _ENV_NAME.value]
  if experiment_label:
    name_parts.append(experiment_label)
  name_parts.append(timestamp)
  if _SUFFIX.value is not None:
    suffix = _sanitize_name_component(_SUFFIX.value)
    if suffix:
      name_parts.append(suffix)
  if _ENV_NAME.value in {"TricoDriver", "TricoDriverSingle"} and _TRICO_DRIVER_XML.value:
    xml_stem = _sanitize_name_component(Path(_TRICO_DRIVER_XML.value).stem)
    if xml_stem:
      name_parts.append(xml_stem)
  return "-".join(name_parts)


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  _maybe_override_trico_driver_xml()

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg["impl"] = _IMPL.value
  if _ENV_NAME.value in {"TricoDriver", "TricoDriverSingle"}:
    if _TRICO_ROTATE_REWARD_SCALE.value is not None:
      env_cfg.rotate_reward_scale = _TRICO_ROTATE_REWARD_SCALE.value
    if _TRICO_SYMMETRY_PENALTY_SCALE.value is not None:
      env_cfg.symmetry_penalty_scale = _TRICO_SYMMETRY_PENALTY_SCALE.value
    if _TRICO_TIP_PENALTY_SCALE.value is not None:
      env_cfg.tip_penetration_penalty_scale = _TRICO_TIP_PENALTY_SCALE.value
    if _TRICO_ALLOWED_TIP_PENETRATION.value is not None:
      env_cfg.allowed_tip_penetration = _TRICO_ALLOWED_TIP_PENETRATION.value
    if _TRICO_DELTA_ACTION_PENALTY_SCALE.value is not None:
      env_cfg.delta_action_penalty_scale = _TRICO_DELTA_ACTION_PENALTY_SCALE.value
    if _TRICO_TIP_Y_PENALTY_SCALE.value is not None:
      env_cfg.tip_y_penalty_scale = _TRICO_TIP_Y_PENALTY_SCALE.value
    if _TRICO_OBS_HISTORY_LEN.value is not None:
      env_cfg.obs_history_len = _TRICO_OBS_HISTORY_LEN.value
    if _TRICO_CONTACT_FORCE_THRESHOLD.value is not None:
      env_cfg.contact_force_threshold = _TRICO_CONTACT_FORCE_THRESHOLD.value
    if _TRICO_CONTACT_REWARD_SCALE.value is not None:
      env_cfg.contact_reward_scale = _TRICO_CONTACT_REWARD_SCALE.value
    if _TRICO_REACH_REWARD_SCALE.value is not None:
      env_cfg.reach_reward_scale = _TRICO_REACH_REWARD_SCALE.value
    if _TRICO_REACH_DISTANCE_SCALE.value is not None:
      env_cfg.reach_distance_scale = _TRICO_REACH_DISTANCE_SCALE.value

  ppo_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNINGRATE.present:
    ppo_params.learning_rate = _LEARNINGRATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value
  if _VISION.value:
    env_cfg.vision = True
    env_cfg.vision_config.render_batch_size = ppo_params.num_envs
  env = registry.load(_ENV_NAME.value, config=env_cfg)
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _DETERMINISTIC_EVAL.present:
    ppo_params.deterministic_eval = _DETERMINISTIC_EVAL.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  print(f"Environment Config:\n{env_cfg}")
  print(f"PPO Training Parameters:\n{ppo_params}")
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  experiment_label = _get_experiment_label()
  exp_name = _build_experiment_name(timestamp, experiment_label)
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in:\n  {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Initialize TensorBoard if required
  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = tensorboardX.SummaryWriter(logdir)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      # Support both:
      # 1) checkpoint root dir containing numeric step subdirs
      # 2) a single checkpoint step dir (may contain non-numeric metadata dirs)
      child_dirs = [p for p in ckpt_path.glob("*") if p.is_dir()]
      numeric_child_dirs = [p for p in child_dirs if p.name.isdigit()]
      if numeric_child_dirs:
        numeric_child_dirs.sort(key=lambda x: int(x.name))
        restore_checkpoint_path = numeric_child_dirs[-1]
      else:
        restore_checkpoint_path = ckpt_path
      print(f"Restoring from:\n  {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint:\n  {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path:\n  {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  )
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  if _VISION.value:
    env = wrapper.wrap_for_brax_training(
        env,
        vision=True,
        num_vision_envs=env_cfg.vision_config.render_batch_size,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.action_repeat,
        randomization_fn=training_params.get("randomization_fn"),
    )

  num_eval_envs = (
      ppo_params.num_envs
      if _VISION.value
      else ppo_params.get("num_eval_envs", 128)
  )

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=None if _VISION.value else wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )

  times = [time.monotonic()]








# ============================================================================
  # ★★★ 修改开始：引入 TQDM 进度条 ★★★
  # ============================================================================
  
  # 1. 初始化进度条
  # total: 总步数
  # unit_scale: 自动把 1000000 变成 1M，方便阅读
  # smoothing: 平滑速度显示
  pbar = tqdm(total=ppo_params.num_timesteps, unit="step", unit_scale=True, smoothing=0.1)
  
  # 用于记录上一次回调时的步数，用来计算增量
  last_step = [0]
  # 打印计数器：每5次回调打印一次精简监控信息（频率从50改为5，提高10倍输出）
  print_counter = [0]
  print_interval = 5

  # 2. 重写 progress 回调函数
  def progress(num_steps, metrics):
    # 计算从上次到现在跑了多少步
    inc = num_steps - last_step[0]
    last_step[0] = num_steps
    
    # 更新进度条
    pbar.update(inc)
    
    # 进度条只显示少量关键指标，避免终端行过长被截断。
    postfix = {}
    metric_candidates = {
        "R": ("eval/episode_reward", "episode/sum_reward"),
        "Rot": ("eval/episode_reward_rotate", "episode/reward_rotate"),
      "Cnt": ("eval/episode_contact_any", "episode/contact_any"),
      "Pen": ("eval/episode_sum_penalty", "episode/sum_penalty"),
      "Tip": ("eval/episode_penalty_tip_penetration", "episode/penalty_tip_penetration"),
      "Vel": ("eval/episode_driver_joint_vel", "episode/driver_joint_vel"),
    }
    for label, keys in metric_candidates.items():
      for key in keys:
        if key in metrics:
          postfix[label] = f"{metrics[key]:.2f}"
          break
    if postfix:
      pbar.set_postfix(postfix)

    # 精简单行监控：保留关键奖励/惩罚/接触，降低I/O并避免刷屏。
    print_counter[0] += 1
    if print_counter[0] >= print_interval:
      print_counter[0] = 0
      short_items = []

      def _first_metric(keys):
        for k in keys:
          if k in metrics:
            return metrics[k]
        return None

      val_r = _first_metric(("eval/episode_reward", "episode/sum_reward"))
      val_rot = _first_metric(("eval/episode_reward_rotate", "episode/reward_rotate"))
      val_cnt = _first_metric(("eval/episode_contact_any", "episode/contact_any"))
      val_tip = _first_metric(("eval/episode_penalty_tip_penetration", "episode/penalty_tip_penetration"))
      val_y = _first_metric(("eval/episode_penalty_tip_y", "episode/penalty_tip_y"))
      val_vel = _first_metric(("eval/episode_driver_joint_vel", "episode/driver_joint_vel"))

      if val_r is not None:
        short_items.append(f"R={val_r:.2f}")
      if val_rot is not None:
        short_items.append(f"Rot={val_rot:.2f}")
      if val_cnt is not None:
        short_items.append(f"Cnt={val_cnt:.2f}")
      if val_tip is not None:
        short_items.append(f"Tip={val_tip:.3f}")
      if val_y is not None:
        short_items.append(f"Y={val_y:.3f}")
      if val_vel is not None:
        short_items.append(f"Vel={val_vel:.2f}")

      if short_items:
        tqdm.write(f"[m {num_steps:7d}] " + " | ".join(short_items))

    # --- 计算 success ratio (Trico 环境特定逻辑) ---
    metrics_to_log = dict(metrics)
    if _ENV_NAME.value in {"TricoDriver", "TricoDriverSingle"}:
      # 从eval metrics中提取必要的指标，计算success ratio
      def _get_eval_metric(keys):
        """Helper to get the first matching eval metric from the keys tuple."""
        for key in keys:
          if key in metrics:
            return metrics[key]
        return None

      rotate_reward = _get_eval_metric(("eval/episode_reward_rotate", "episode/reward_rotate"))
      tip_penalty = _get_eval_metric(("eval/episode_penalty_tip_penetration", "episode/penalty_tip_penetration"))
      tip_y_penalty = _get_eval_metric(("eval/episode_penalty_tip_y", "episode/penalty_tip_y"))

      # 如果有完整的metrics数据，计算success判决
      if rotate_reward is not None and tip_penalty is not None and tip_y_penalty is not None:
        # 使用与render_checkpoint_rollouts.py相同的阈值
        TRICO_SUCCESS_ROTATE_THRESHOLD = 2000.0
        TRICO_SUCCESS_TIP_PENALTY_THRESHOLD = 1.0
        TRICO_SUCCESS_TIP_Y_THRESHOLD = 1.0

        # 判决是否本次evaluation成功
        is_success = (
            float(rotate_reward) >= TRICO_SUCCESS_ROTATE_THRESHOLD
            and float(tip_penalty) <= TRICO_SUCCESS_TIP_PENALTY_THRESHOLD
            and float(tip_y_penalty) <= TRICO_SUCCESS_TIP_Y_THRESHOLD
        )
        # 将成功状态添加到metrics中（供wandb记录）
        # 如果想要success_ratio，需要在这里手动追踪成功数
        # 这里我们记录当前evaluation的成功与否
        metrics_to_log["eval/success_rate"] = float(is_success)

    # --- 原有的日志逻辑 (保持不变) ---
    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics_to_log, step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
      
    # 原有的 print 我们就不需要了，因为 tqdm 会接管屏幕输出
    # 如果保留 print，会把进度条打断，很难看

  # 3. 加载评估环境 (保持原样)
  eval_env = None
  if not _VISION.value:
    eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
  num_envs = 1
  if _VISION.value:
    num_envs = env_cfg.vision_config.render_batch_size

  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    from rscope import brax as rscope_utils

    if not _VISION.value:
      rscope_env = registry.load(_ENV_NAME.value, config=env_cfg)
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        ppo_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )

    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
      rscope_handle.set_make_policy(make_policy)
      rscope_handle.dump_rollout(params)

  # 4. 开始训练 (Train)
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )
  
  # 训练结束，关闭进度条
  pbar.close()
  
  # ============================================================================
  # ★★★ 修改结束 ★★★
  # ============================================================================


















  # # Progress function for logging
  # def progress(num_steps, metrics):
  #   times.append(time.monotonic())

  #   # Log to Weights & Biases
  #   if _USE_WANDB.value and not _PLAY_ONLY.value:
  #     wandb.log(metrics, step=num_steps)

  #   # Log to TensorBoard
  #   if _USE_TB.value and not _PLAY_ONLY.value:
  #     for key, value in metrics.items():
  #       writer.add_scalar(key, value, num_steps)
  #     writer.flush()
  #   if _RUN_EVALS.value:
  #     print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
  #   if _LOG_TRAINING_METRICS.value:
  #     if "episode/sum_reward" in metrics:
  #       print(
  #           f"{num_steps}: mean episode"
  #           f" reward={metrics['episode/sum_reward']:.3f}"
  #       )




  # # Load evaluation environment.
  # eval_env = None
  # if not _VISION.value:
  #   eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
  # num_envs = 1
  # if _VISION.value:
  #   num_envs = env_cfg.vision_config.render_batch_size

  # policy_params_fn = lambda *args: None
  # if _RSCOPE_ENVS.value:
  #   # Interactive visualisation of policy checkpoints
  #   from rscope import brax as rscope_utils

  #   if not _VISION.value:
  #     rscope_env = registry.load(_ENV_NAME.value, config=env_cfg)
  #     rscope_env = wrapper.wrap_for_brax_training(
  #         rscope_env,
  #         episode_length=ppo_params.episode_length,
  #         action_repeat=ppo_params.action_repeat,
  #         randomization_fn=training_params.get("randomization_fn"),
  #     )
  #   else:
  #     rscope_env = env

  #   rscope_handle = rscope_utils.BraxRolloutSaver(
  #       rscope_env,
  #       ppo_params,
  #       _VISION.value,
  #       _RSCOPE_ENVS.value,
  #       _DETERMINISTIC_RSCOPE.value,
  #       jax.random.PRNGKey(_SEED.value),
  #       rscope_fn,
  #   )

  #   def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
  #     rscope_handle.set_make_policy(make_policy)
  #     rscope_handle.dump_rollout(params)

  # # Train or load the model
  # make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
  #     environment=env,
  #     progress_fn=progress,
  #     policy_params_fn=policy_params_fn,
  #     eval_env=eval_env,
  # )
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  if not _RENDER_ROLLOUTS.value or _NUM_VIDEOS.value <= 0:
    print(
        "Skipping rollout rendering. "
        "Use --render_rollouts=True --num_videos=N to generate videos."
    )
    return

  print("Starting inference...")

  # Save rendered videos near the selected checkpoint by default.
  if restore_checkpoint_path is not None:
    render_output_dir = Path(str(restore_checkpoint_path)) / "rollouts"
  else:
    render_output_dir = Path(str(logdir)) / "rollouts"
  render_output_dir.mkdir(parents=True, exist_ok=True)

  # Create inference function.
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # Run evaluation rollouts.
  def do_rollout(rng, state):
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})  # pytype: disable=attribute-error
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_key)[0]
      state = eval_env.step(state, act)
      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.time": state.data.time,
          "data.ctrl": state.data.ctrl,
          "data.mocap_pos": state.data.mocap_pos,
          "data.mocap_quat": state.data.mocap_quat,
          "data.xfrc_applied": state.data.xfrc_applied,
      })
      if _VISION.value:
        traj_data = jax.tree_util.tree_map(lambda x: x[0], traj_data)
      return (state, rng), traj_data

    _, traj = jax.lax.scan(
        step, (state, rng), None, length=_EPISODE_LENGTH.value
    )
    return traj

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)
  if _VISION.value:
    reset_states = jax.tree_util.tree_map(lambda x: x[0], reset_states)
  traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(_EPISODE_LENGTH.value)
    ]

  # Render and save the rollout.
  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  try:
    for i, rollout in enumerate(trajectories):
      traj = rollout[::render_every]
      frames = eval_env.render(
          traj, height=480, width=640, scene_option=scene_option
      )
      video_path = render_output_dir / f"rollout{i}.mp4"
      media.write_video(str(video_path), frames, fps=fps)
      print(f"Rollout video saved as '{video_path}'.")
  except Exception as exc:  # pylint: disable=broad-except
    print(
        "Rollout rendering failed. On headless servers, prefer "
        "--render_rollouts=False or ensure EGL/OSMesa is available."
    )
    print(f"Render error: {exc}")


if __name__ == "__main__":
  app.run(main)

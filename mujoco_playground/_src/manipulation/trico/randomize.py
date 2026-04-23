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
"""Domain randomization for Trico screwdriver rotation task."""

from typing import Any, NamedTuple, Tuple, cast

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.trico import trico_driver_single as trico_env


class DrPhaseConfig(NamedTuple):
  """Domain randomization ranges for a training phase."""

  actuator_min: float
  actuator_max: float
  damping_min: float
  damping_max: float
  armature_min: float
  armature_max: float
  driver_mass_min: float
  driver_mass_max: float
  driver_friction_min: float
  driver_friction_max: float
  qpos0_noise_abs: float
  fingertip_friction_min: float
  fingertip_friction_max: float
  obs_noise_std: float
  action_noise_std: float


PHASE1_CONFIG = DrPhaseConfig(
    actuator_min=0.95,
    actuator_max=1.05,
    damping_min=0.95,
    damping_max=1.05,
    armature_min=0.95,
    armature_max=1.05,
    driver_mass_min=0.95,
    driver_mass_max=1.05,
    driver_friction_min=0.9,
    driver_friction_max=1.1,
    qpos0_noise_abs=0.08,
    # qpos0_noise_abs=0.02,
    fingertip_friction_min=0.9,
    fingertip_friction_max=1.1,
    obs_noise_std=0.0,
    action_noise_std=0.0,
)

PHASE2_CONFIG = DrPhaseConfig(
    actuator_min=0.9,
    actuator_max=1.1,
    damping_min=0.9,
    damping_max=1.1,
    armature_min=0.9,
    armature_max=1.1,
    driver_mass_min=0.9,
    driver_mass_max=1.1,
    driver_friction_min=0.8,
    driver_friction_max=1.2,
    qpos0_noise_abs=0.14,
    # qpos0_noise_abs=0.03,
    fingertip_friction_min=0.85,
    fingertip_friction_max=1.15,
    obs_noise_std=0.0,
    action_noise_std=0.0,
)

PHASE3_CONFIG = DrPhaseConfig(
    actuator_min=0.8,
    actuator_max=1.2,
    damping_min=0.8,
    damping_max=1.2,
    armature_min=0.8,
    armature_max=1.2,
    driver_mass_min=0.8,
    driver_mass_max=1.2,
    driver_friction_min=0.6,
    driver_friction_max=1.4,
    qpos0_noise_abs=0.05,
    fingertip_friction_min=0.75,
    fingertip_friction_max=1.25,
    obs_noise_std=0.02,
    action_noise_std=0.02,
)


def _get_all_hand_body_ids(mj_model) -> np.ndarray:
  """Get all body IDs belonging to the Trico hand.
  
  Automatically discovers hand bodies from the model by looking for bodies
  whose names contain the fingers' prefixes (right, left, thumb).
  
  Returns:
    Array of body IDs belonging to the hand structure (fingers).
  """
  # Patterns to identify bodies belonging to the three fingers
  finger_patterns = ("_right", "_left", "figer_right", "figer_left")
  
  hand_body_ids = []
  # Iterate over all bodies in the model
  for body_id in range(mj_model.nbody):
    body_name = mj_model.body(body_id).name
    # Check if this body belongs to a finger structure
    if any(pattern in body_name for pattern in finger_patterns):
      hand_body_ids.append(body_id)
  
  return np.array(hand_body_ids, dtype=np.int32)


def _domain_randomize_with_config(
    model: mjx.Model,
    rng: jax.Array,
    cfg: DrPhaseConfig,
) -> Tuple[mjx.Model, Any]:
  """Domain randomization for Trico environments.

  Applies physics randomization with ranges provided in cfg.

  Args:
    model: MJX model to randomize
    rng: Random number generator key

  Returns:
    Tuple of (randomized_model, in_axes) for vmap
  """
  # Load the reference Trico model to get body/joint names
  mj_model = trico_env.TricoDriverSingleEnv().mj_model

  # Get hand body IDs by automatic discovery from model structure
  hand_body_ids = _get_all_hand_body_ids(mj_model)

  # Get hand joint IDs by discovering all actuated joints in finger structures
  # These are joints whose names contain "_right" or "_left" (finger identifiers)
  finger_patterns = ("_right", "_left")
  hand_qids = []
  for joint_id in range(mj_model.njnt):
    joint_name = mj_model.joint(joint_id).name
    # Include joints that belong to finger structures
    if any(pattern in joint_name for pattern in finger_patterns):
      qpos_adr = mj_model.jnt_qposadr[joint_id]
      # Only include joints that have qpos (not fixed joints)
      if 0 <= qpos_adr < len(mj_model.qpos0):
        hand_qids.append(qpos_adr)
  hand_qids = np.array(hand_qids, dtype=np.int32)

  # Get driver (screwdriver) body ID
  try:
    driver_body_id = mj_model.body("driver_handle").id
  except KeyError:
    # Fallback: try to find it by name prefix
    driver_body_id = mj_model.body("driver").id

  @jax.vmap
  def rand(rng):
    """Randomize a single environment."""
    rng, key = jax.random.split(rng)

    # ========================================================================
    # Domain Randomization
    # ========================================================================

    # 1. Actuator gain (joint stiffness)
    #    This captures manufacturing variation in servo strengths
    rng, key = jax.random.split(rng)
    kp_scale = jax.random.uniform(
      key,
      (model.nu,),
      minval=cfg.actuator_min,
      maxval=cfg.actuator_max,
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
        model.actuator_gainprm[:, 0] * kp_scale
    )
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-actuator_gainprm[:, 0])

    # 2. Dof damping (joint friction)
    #    This captures servo damping variation
    rng, key = jax.random.split(rng)
    damping_scale = jax.random.uniform(
      key,
      shape=(len(hand_qids),),
      minval=cfg.damping_min,
      maxval=cfg.damping_max,
    )
    dof_damping = model.dof_damping.at[hand_qids].set(
        model.dof_damping[hand_qids] * damping_scale
    )

    # 3. Dof armature (joint inertia)
    #    This captures motor inertia and gear ratio variation
    rng, key = jax.random.split(rng)
    armature_scale = jax.random.uniform(
      key,
      shape=(len(hand_qids),),
      minval=cfg.armature_min,
      maxval=cfg.armature_max,
    )
    dof_armature = model.dof_armature.at[hand_qids].set(
        model.dof_armature[hand_qids] * armature_scale
    )

    # 4. Driver (screwdriver) mass
    #    Captures tool weight variation
    rng, key = jax.random.split(rng)
    driver_mass_scale = jax.random.uniform(
      key,
      minval=cfg.driver_mass_min,
      maxval=cfg.driver_mass_max,
    )
    body_mass = model.body_mass.at[driver_body_id].set(
        model.body_mass[driver_body_id] * driver_mass_scale
    )

    # 5. Driver friction
    #    This is the contact friction between tool and hand
    #    Most important for grasp stability
    rng, key = jax.random.split(rng)
    # Find the driver interaction geoms (e.g., driver_handle surface)
    # For now, we'll randomize some geom friction that likely contacts the driver
    try:
      driver_geom_id = mj_model.geom("driver_handle_collision").id
      driver_friction = jax.random.uniform(
          key,
          minval=cfg.driver_friction_min,
          maxval=cfg.driver_friction_max,
      )
      geom_friction = model.geom_friction.at[driver_geom_id, 0].set(
          model.geom_friction[driver_geom_id, 0] * driver_friction
      )
    except (KeyError, IndexError):
      # If driver geom doesn't exist, just don't randomize
      geom_friction = model.geom_friction

    # 6. Hand joint qpos0 (home position)
    #    Captures calibration / zero-point offset variation
    rng, key = jax.random.split(rng)
    qpos0_noise = jax.random.uniform(
        key,
        shape=(len(hand_qids),),
        minval=-cfg.qpos0_noise_abs,
        maxval=cfg.qpos0_noise_abs,
    )
    qpos0 = model.qpos0.at[hand_qids].add(qpos0_noise)

    # 7. Hand finger surface friction
    #    This affects grip quality on the tool
    rng, key = jax.random.split(rng)
    try:
      # Try to find and randomize fingertip friction
      fingertip_geom_names = [
          "tip_geom_right", "tip_geom_left"
      ]
      fingertip_geom_ids = []
      for name in fingertip_geom_names:
        try:
          geom_id = mj_model.geom(name).id
          fingertip_geom_ids.append(geom_id)
        except KeyError:
          pass

      if fingertip_geom_ids:
        fingertip_friction_scale = jax.random.uniform(
            key,
            shape=(len(fingertip_geom_ids),),
            minval=cfg.fingertip_friction_min,
            maxval=cfg.fingertip_friction_max,
        )
        for i, geom_id in enumerate(fingertip_geom_ids):
          geom_friction = geom_friction.at[geom_id, 0].set(
              geom_friction[geom_id, 0] * fingertip_friction_scale[i]
          )
    except Exception:
      # Silently skip if fingertip geoms don't exist
      pass

    # Compile all randomized parameters
    return (
        geom_friction,
        body_mass,
        dof_damping,
        dof_armature,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
    )

  # Apply vmap to generate batch of randomizations
  (
      geom_friction,
      body_mass,
      dof_damping,
      dof_armature,
      qpos0,
      actuator_gainprm,
      actuator_biasprm,
  ) = rand(rng)

  # Define vmap axes for each randomized parameter
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_mass": 0,
      "dof_damping": 0,
      "dof_armature": 0,
      "qpos0": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })

  # Apply all randomizations to the model
  model = cast(mjx.Model, model.tree_replace({
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "dof_damping": dof_damping,
      "dof_armature": dof_armature,
      "qpos0": qpos0,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  }))

  return model, in_axes


def domain_randomize(
  model: mjx.Model, rng: jax.Array
) -> Tuple[mjx.Model, Any]:
  """Phase 1 (near-DR) randomization entrypoint."""
  return _domain_randomize_with_config(model, rng, PHASE1_CONFIG)


def domain_randomize_phase2(
  model: mjx.Model, rng: jax.Array
) -> Tuple[mjx.Model, Any]:
  """Phase 2 (stronger DR) randomization entrypoint."""
  return _domain_randomize_with_config(model, rng, PHASE2_CONFIG)


def domain_randomize_phase3(
  model: mjx.Model, rng: jax.Array
) -> Tuple[mjx.Model, Any]:
  """Phase 3 (strong-dr-v2, aggressive DR) randomization entrypoint."""
  return _domain_randomize_with_config(model, rng, PHASE3_CONFIG)

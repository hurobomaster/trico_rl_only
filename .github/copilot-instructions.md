---
name: trico-rl-workspace
description: |
  Instructions for trico_rl_only workspace: JAX-based RL framework for robotic hand manipulation.
  Use when: working on MuJoCo simulation, training PPO agents, Sim2Real bridging, or deployment.
---

# Trico RL Workspace Instructions

This is a professional robotics RL framework combining **MuJoCo physics simulation**, **JAX acceleration**, **Brax PPO training**, and **real hardware integration** for the Trico dexterous hand.

## 🚀 Quick Start

### Environment Setup
```bash
# Activate conda environment
conda activate /home/hurz/code/trico_rl_only/.conda

# First-time setup (optional—dependencies usually pre-installed)
pip install -e ".[dev,learning]"

# Verify no lint issues
pre-commit run --all-files
```

### Run Training
```bash
# Quick test (10K timesteps, 64 parallel environments)
python learning/train_jax_ppo.py --env_name=TricoDriverSingle --num_timesteps=10000 --num_envs=64 --seed=42

# Full training (150M steps, GPU required)
python learning/train_jax_ppo.py --env_name=TricoDriverSingle --num_timesteps=150000000 --num_envs=4096 --use_tb=True --use_wandb=True
```

### Code Quality
```bash
# Pre-commit checks (required before commit)
pre-commit run --all-files

# Pytest (only for `mujoco_playground/_src/`)
pytest -n auto mujoco_playground/_src/
```

---

## 📁 Project Structure

| Path | Purpose | Key Files |
|------|---------|-----------|
| `mujoco_playground/_src/manipulation/trico/` | Trico hand environments | `trico.py`, `trico_driver_single.py`, `ik_utils.py`, `xmls/` |
| `mujoco_playground/config/` | Hyperparameter templates | `manipulation_params.py` (Trico configs: LR 3e-4, entropy 0.01) |
| `learning/` | Training & inference scripts | `train_jax_ppo.py`, `train_rsl_rl.py`, `render_checkpoint_rollouts.py` |
| `deployment/` | Inference & hardware integration | Hardware command interface (Sim2Real bridge logic) |
| `logs/` | Training outputs | Checkpoints (orbax format), TensorBoard events |
| `mujoco_playground/_src/inference/` | Sim2Real utilities | *(Sim2Real bridge to be moved here)* |

---

## 🏗️ Architecture Concepts

### Environment Registration & Naming
All environments auto-register via `registry.py`. Variants of the same hardware:
- **`Trico`** — Basic hand (7 DOF per finger)
- **`TricoDriver`** — Multi-finger object pushing (harder task)
- **`TricoDriverSingle`** — Single-finger variant (easier, faster training)
- **`LeapCubeReorient`** — Different hand, different task

Load any variant: `env_registry("TricoDriverSingle")`

### Observation & Action Spaces (TricoDriverSingle)
```python
Observation (27-D):
  Joint angles (8) + force sensors (6) + object pose (7) + relative position (6)

Action (8-D):
  Motor commands per joint [normalized -1 to 1]

Episode length: 1000 steps (20 seconds at 50 Hz)
**Physics**: MuJoCo MJX (JAX JIT-compiled), vectorized across 4096 parallel envs
```

### Training Loop (JAX PPO)
```
1. Environment step (vectorized)
2. GAE advantage computation
3. PPO update (multiple epochs)
4. Logging (TensorBoard, W&B, checkpoints every 2.6M steps)
```

See `train_jax_ppo.py` for full pipeline.

---

## ⚠️ Critical Gotchas & Conventions

### 🚩 DELETED FILES — Do NOT use these
- ~~`sim_real_mapping.py`~~ — Old Sim2Real converter
- ~~`test_deployment_pipeline.py`~~ — Deprecated test

**If you see imports from these**: Delete them immediately. The canonical bridge is in **`sim2real_bridge.py`** (or will be in `_src/inference/`).

### Sim2Real Bridge Rules
From [`LOOK_CONTEXT_MEMORY.md`](LOOK_CONTEXT_MEMORY.md):

1. **Canonical source**: `mujoco_playground._src.inference.sim2real_bridge.py` (contains `RealRobotBridge` class)
2. **Two-way mapping**:
   - **Real2Sim (obs)**: `servo_angles_to_hinge_angles(8D_servos)` → 8D hinge angles
   - **Sim2Real (action)**: `hinge_angles_to_servo_angles(8D_hinges, apply_bank_ab=False)` → 8D servo targets
3. **IK input fix**: For direction vectors, convert to Cartesian: `Pos = Dir * L_arm` (arm length: 30mm internal + 30mm external)
4. **Hardware Bank A/B mapping**:
   - **Never** call `apply_bank_ab=True` in RL inference
   - Hardware handles Bank mapping at final stage (do not double-convert)
   - Bank A (left): complementary mapping `360 - x`
   - Bank B (right): direct mapping `x`

### Coordinate Systems
- **Robot base**: Fixed at origin
- **World frame**: +X, +Y for object pushing task
- **Object start**: `[0, 0, 0.029]` (z = table height)
- **Reward function**: Maximizes `max(0, obj_pos[0]) + max(0, obj_pos[1])` only positive displacement

### Code Style & Pre-Commit Checks
- **Formatter**: Pyink (80-char line limit, not 88!)
- **Imports**: isort (force single-line imports, lexicographical order)
- **Linting**: Ruff (rules: E, F, PLC, PLE, PLR, PLW, I)
- **Type checking**: Mypy (targets Python 3.12)
- **Line limit violation**: Blocks commit

Run `pre-commit run --all-files` before every commit.

---

## 🔧 Training Flags & Hyperparameters

### Key Command-Line Arguments
```bash
# Environment & task
--env_name=TricoDriverSingle        # Env variant (must exist in registry)

# Training scale
--num_timesteps=150000000           # Total steps (150M for Trico)
--num_envs=4096                     # Parallel environments (requires GPU)
--episode_length=1000               # Max steps per episode

# PPO hyperparameters
--learning_rate=0.0003              # LR (tuned per task in config)
--entropy_cost=0.01                 # Entropy regularization (Trico-specific)
--num_epochs=32                     # PPO epochs per update

# Features
--use_tb=True                        # TensorBoard logging
--use_wandb=True                     # Weights & Biases integration
--domain_randomization=True          # Enable randomization
--vision=False                       # Pixel observations (if True, needs vision net)

# Resume & inference
--load_checkpoint_path=/path/...     # Resume from checkpoint
--play_only=True                     # Inference-only (no training)
--seed=42                            # Random seed for reproducibility
```

### Config Templates
For Trico (`manipulation_params.py`):
- **Learning rate**: 3e-4 (sensitive parameter)
- **Entropy cost**: 0.01 (prevents premature convergence)
- **Actor/Critic layers**: [512, 512]
- **Discount factor γ**: 0.99
- **GAE λ**: 0.95

---

## 📊 Logging & Checkpoints

### TensorBoard
```bash
tensorboard --logdir=logs/
```
Open browser to `http://localhost:6006`. Logs training curves, returns, action stats.

### Weights & Biases
Logs to `wandb/` (local cache) and `wandb.ai` (if logged in). Set `--use_wandb=True` to enable.

### Checkpoint Format (Orbax)
```
logs/{run_name}/checkpoints/{step}/
├── config.json             # Environment & training config
└── checkpoint file         # JAX pytree (weights + optimizer state)
```
Checkpoints saved every ~2.6M steps. Resume: `--load_checkpoint_path=logs/.../000010485760`

---

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Out of memory (OOM)** | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`; reduce `--num_envs` to 2048 or 1024 |
| **Slow checkpoint save** | Orbax is async; 150M step run = 50+ GB; may take 10+ min per save |
| **Menagerie submodule fails** | Run `git submodule update --init` to clone Trico-Control kinematics repo |
| **ImportError: `wandb`** | Local `wandb/` folder shadows pip package; `train_jax_ppo.py` has workaround |
| **Test discovery fails** | Pytest only runs tests in `mujoco_playground/_src/`; other tests ignored |
| **JAX recompilation lag** | Normal during first few steps; JIT compilation caches after |

---

## 🤖 JAX & Development Tips

### Environment Variables (Set before training)
```bash
# GPU optimization
export JAX_PLATFORM_NAME=gpu
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Headless rendering (Linux)
export MUJOCO_GL=egl

# Training script activates these automatically; shown for reference
```

### Python Dependencies
- **Core**: JAX, MuJoCo (3.3.6+), Brax (>=0.12.5), Flax
- **RL**: Brax PPO agent, Orbax checkpoint framework
- **Logging**: TensorBoard, Weights & Biases
- **Utilities**: ml_collections (ConfigDict), lxml, tqdm

See `pyproject.toml` for full dependency tree.

### Inspection & Debugging
```python
# Print environment observation/action shapes
from mujoco_playground._src.registry import env_registry
env = env_registry("TricoDriverSingle")
print(f"obs shape: {env.observation_size}")
print(f"action shape: {env.action_size}")

# Render rollout from checkpoint
python learning/render_checkpoint_rollouts.py --checkpoint_path=... --num_episodes=5
```

---

## 📚 Key Files to Know

| File | Role | When to Edit |
|------|------|--------------|
| `mujoco_playground/_src/manipulation/trico/trico.py` | Hand kinematics, observation design, reward function | Tuning task, changing observation/reward structure |
| `mujoco_playground/_src/manipulation/trico/trico_driver_single.py` | Environment wrapper, domain randomization | Adding randomization, modifying task |
| `mujoco_playground/_src/manipulation/trico/ik_utils.py` | Inverse kinematics utilities | Extending IK solver, adding finger constraints |
| `learning/train_jax_ppo.py` | PPO training loop | Tuning PPO hyperparams, adding custom logging |
| `mujoco_playground/config/manipulation_params.py` | Hyperparameter templates | Changing default LR, entropy cost, network sizes |
| `LOOK_CONTEXT_MEMORY.md` | Architecture & Sim2Real decisions | Reference for historical context and conventions |

---

## 🔗 Related Documentation

- **Trico Hardware Kinematics** → [`LOOK_CONTEXT_MEMORY.md`](LOOK_CONTEXT_MEMORY.md) (Sim2Real conventions)
- **Project Overview** → [`README.md`](README.md) (Task description, physics constraints)
- **MuJoCo Docs** → [mujoco.org](https://mujoco.org/) (Physics engine reference)
- **Brax Docs** → [github.com/google-deepmind/brax](https://github.com/google-deepmind/brax) (PPO & physics simulation)
- **JAX Docs** → [jax.readthedocs.io](https://jax.readthedocs.io/) (Numerical computing)

---

## ✅ Development Workflow

### Before Committing
```bash
# 1. Run pre-commit checks
pre-commit run --all-files

# 2. Run quick tests
pytest mujoco_playground/_src/ -v

# 3. If training modified: test on small run
python learning/train_jax_ppo.py --num_timesteps=10000 --num_envs=64
```

### Pushing to Production
```bash
# 1. Final validation
pre-commit run --all-files
pytest -n auto mujoco_playground/_src/

# 2. Train on full config
python learning/train_jax_ppo.py --env_name=TricoDriverSingle --use_tb=True --use_wandb=True

# 3. Monitor TensorBoard & W&B for 5+ hours before merging
```

---

## 💡 When to Contact Maintainers

- **Sim2Real mapping fails**: Check `sim2real_bridge.py` and `LOOK_CONTEXT_MEMORY.md` first (likely `apply_bank_ab` flag)
- **Menagerie submodule issues**: `git submodule update --init` usually fixes
- **Hardware integration**: Reference `deployment/` and Trico-Control external repo
- **Training instability**: Likely hyperparameter tuning (entropy cost, learning rate) — see `manipulation_params.py`

---

**Last Updated**: 2026-04-10  
**Framework Version**: Brax 0.12.5+, JAX latest, MuJoCo 3.3.6+

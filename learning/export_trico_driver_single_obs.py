"""Export first N observation frames from a restored TricoDriverSingle policy.

This script restores PPO params from a checkpoint and runs deterministic rollout.
It saves raw observations for strict sim2real alignment checks.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
import numpy as np
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params


def _load_env_config(checkpoint_path: Path):
    env_cfg = registry.get_default_config("TricoDriverSingle")
    env_cfg["impl"] = "jax"

    config_json = checkpoint_path.parent / "config.json"
    if config_json.exists():
        with open(config_json, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for k, v in saved.items():
            env_cfg[k] = v

    return env_cfg


def _restore_policy(checkpoint_path: Path, env_cfg, num_envs: int):
    env = registry.load("TricoDriverSingle", config=env_cfg)
    ppo_params = manipulation_params.brax_ppo_config("TricoDriverSingle", "jax")

    network_factory = ppo_networks.make_ppo_networks
    if hasattr(ppo_params, "network_factory"):
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory,
        )

    train_kwargs = dict(ppo_params)
    train_kwargs.pop("network_factory", None)
    train_kwargs.update(
        num_timesteps=0,
        num_envs=num_envs,
        num_eval_envs=num_envs,
        batch_size=num_envs,
        num_minibatches=1,
        num_updates_per_batch=1,
        unroll_length=1,
        log_training_metrics=False,
        run_evals=False,
        restore_checkpoint_path=checkpoint_path,
        save_checkpoint_path=None,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        seed=42,
    )

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        network_factory=network_factory,
        **train_kwargs,
    )
    return env, make_inference_fn, params


def _rollout_obs(
    env,
    make_inference_fn,
    params,
    steps: int,
    seed: int,
    disable_jit: bool,
):
    inference_fn = make_inference_fn(params, deterministic=True)
    if not disable_jit:
        inference_fn = jax.jit(inference_fn)
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    obs_frames = []
    act_frames = []
    rew_frames = []

    for _ in range(steps):
        rng, act_key = jax.random.split(rng)
        action = inference_fn(state.obs, act_key)[0]
        obs_frames.append(np.asarray(state.obs, dtype=np.float32))
        act_frames.append(np.asarray(action, dtype=np.float32))
        rew_frames.append(float(state.reward))
        state = env.step(state, action)

    return (
        np.asarray(obs_frames, dtype=np.float32),
        np.asarray(act_frames, dtype=np.float32),
        np.asarray(rew_frames, dtype=np.float32),
    )


def _save_csv(obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    obs_dim = obs.shape[1]
    act_dim = actions.shape[1]

    header = ["frame"]
    header += [f"obs_{i:03d}" for i in range(obs_dim)]
    header += [f"act_{i:02d}" for i in range(act_dim)]
    header += ["reward"]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(obs.shape[0]):
            row = [i]
            row += obs[i].tolist()
            row += actions[i].tolist()
            row += [rewards[i]]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--disable_jit",
        action="store_true",
        help="Run without JAX JIT to avoid long compile time.",
    )
    parser.add_argument(
        "--out_csv",
        default="Trico-Control/doc/trico_driver_single_obs_50frames.csv",
    )
    parser.add_argument(
        "--out_npz",
        default="Trico-Control/doc/trico_driver_single_obs_50frames.npz",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_path).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)

    env_cfg = _load_env_config(ckpt)
    env, make_inference_fn, params = _restore_policy(ckpt, env_cfg, args.num_envs)
    obs, actions, rewards = _rollout_obs(
        env,
        make_inference_fn,
        params,
        steps=args.steps,
        seed=args.seed,
        disable_jit=args.disable_jit,
    )

    out_csv = Path(args.out_csv).resolve()
    out_npz = Path(args.out_npz).resolve()

    _save_csv(obs, actions, rewards, out_csv)
    np.savez(
        out_npz,
        obs=obs,
        actions=actions,
        rewards=rewards,
        obs_shape=np.asarray(obs.shape, dtype=np.int32),
        obs_dim=np.asarray([obs.shape[1]], dtype=np.int32),
    )

    print(f"saved csv: {out_csv}")
    print(f"saved npz: {out_npz}")
    print(f"obs shape: {obs.shape}")


if __name__ == "__main__":
    main()

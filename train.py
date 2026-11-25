"""Train a DQN agent to drive on the highway.

    python train.py                  # default: 20,000 steps on highway-fast-v0
    python train.py --steps 50000    # longer training = better driving
    python train.py --steps 50000 --gamma 0.95 --collision-reward -3   # safer driving
    python train.py --obs camera --steps 50000 --gamma 0.95 --collision-reward -3 --out results_camera

Every --eval-freq steps (default 10,000) the agent is tested and the best version is kept
as <out>/best_model.zip, next to the final model <out>/dqn_highway.zip.
"""
import argparse
import json
import os

import src  # noqa: F401

import matplotlib
matplotlib.use("Agg")  # save plots to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.env import DEFAULT_ENV, OBSERVATION_TYPES, make_env


def plot_training(monitor_csv, out_png, env_id):
    """Plot the reward of every training episode, plus a moving average."""
    df = pd.read_csv(monitor_csv, skiprows=1)  # first line of a Monitor file is a comment
    rewards = df["r"]
    smooth = rewards.rolling(20, min_periods=1).mean()

    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.3, label="Episode reward")
    plt.plot(smooth, label="Moving average (20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"DQN training on {env_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print("Saved plot to", out_png)


def plot_evaluations(npz_path, out_png, env_id):
    """Plot the test reward measured every --eval-freq steps during training."""
    data = np.load(npz_path)
    steps, results = data["timesteps"], data["results"]  # results: (n_evals, n_episodes)
    mean, std = results.mean(axis=1), results.std(axis=1)
    best = int(np.argmax(mean))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, mean, marker="o", label="Mean test reward")
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
    plt.scatter([steps[best]], [mean[best]], s=120, facecolors="none", edgecolors="red",
                label=f"Best model (step {steps[best]})")
    plt.xlabel("Training steps")
    plt.ylabel("Reward (deterministic test episodes)")
    plt.title(f"Test performance during training on {env_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"Saved plot to {out_png} (best mean reward {mean[best]:.2f} at step {steps[best]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=DEFAULT_ENV)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.8,
                   help="discount factor: higher = cares more about the future (staying alive)")
    p.add_argument("--collision-reward", type=float, default=None,
                   help="crash penalty, e.g. -3 (highway-env default is -1)")
    p.add_argument("--obs", choices=OBSERVATION_TYPES, default="kinematics",
                   help="what the agent sees: table of positions, or camera images")
    p.add_argument("--eval-freq", type=int, default=10000,
                   help="test the agent every N steps and keep the best model (0 = off)")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Optional changes to the environment's reward; saved so evaluate.py uses the same setup
    config = {}
    if args.collision_reward is not None:
        config["collision_reward"] = args.collision_reward
    with open(os.path.join(args.out, "env_config.json"), "w") as f:
        json.dump({"env": args.env, "obs": args.obs, "config": config, "gamma": args.gamma},
                  f, indent=2)

    # Monitor records each episode's reward and length to results/train.monitor.csv
    env = Monitor(make_env(args.env, config=config or None, obs=args.obs),
                  os.path.join(args.out, "train"))

    if args.obs == "camera":
        # Camera images -> convolutional neural network (CNN), like real self-driving cars
        policy, policy_kwargs = "CnnPolicy", None
    else:
        # Table of numbers -> plain fully-connected network with two layers of 256 neurons
        policy, policy_kwargs = "MlpPolicy", dict(net_arch=[256, 256])

    model = DQN(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-4,
        buffer_size=15000,              # replay memory size
        learning_starts=200,            # random steps collected before learning
        batch_size=32,
        gamma=args.gamma,               # how much future rewards count (0..1)
        train_freq=1,                   # learn after every step
        gradient_steps=1,
        target_update_interval=50,      # copy the network to the target network every 50 steps
        exploration_fraction=0.7,       # explore randomly for the first 70% of training
        verbose=1,
        seed=args.seed,
    )

    print(f"Training on {args.env} with {args.obs} observations for {args.steps} steps "
          f"(gamma={args.gamma}, env config={config or 'default'}, device={model.device})...")
    # DQN's performance can rise and then collapse late in training. So every --eval-freq
    # steps we test the agent (no random exploration) on a separate copy of the road and
    # keep the best version as best_model.zip, in addition to the final model.
    callback, eval_env = None, None
    if args.eval_freq > 0:
        eval_env = Monitor(make_env(args.env, config=config or None, obs=args.obs))
        callback = EvalCallback(
            eval_env,
            best_model_save_path=args.out,   # -> best_model.zip
            log_path=args.out,               # -> evaluations.npz
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1,
        )

    model.learn(total_timesteps=args.steps, callback=callback)

    model_path = os.path.join(args.out, "dqn_highway")
    model.save(model_path)  # creates results/dqn_highway.zip (the FINAL model)
    print("Saved final model to", model_path + ".zip")

    env.close()
    if eval_env is not None:
        eval_env.close()
    plot_training(os.path.join(args.out, "train.monitor.csv"),
                  os.path.join(args.out, "training_rewards.png"), args.env)
    npz = os.path.join(args.out, "evaluations.npz")
    if os.path.exists(npz):
        plot_evaluations(npz, os.path.join(args.out, "eval_rewards.png"), args.env)
        print("Best model saved to", os.path.join(args.out, "best_model.zip"))


if __name__ == "__main__":
    main()

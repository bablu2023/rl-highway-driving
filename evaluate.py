"""Test the trained driver (or a random one) and record a video.

    python evaluate.py                     # trained agent, 20 episodes + results/agent.mp4
    python evaluate.py --random            # random driver, for comparison
    python evaluate.py --no-video
"""
import argparse
import os

import src  # noqa: F401

import numpy as np

from src.env import DEFAULT_ENV, make_env


def run_episodes(env, choose_action, episodes, record=0):
    """Drive `episodes` episodes. Returns per-episode stats and video frames."""
    stats, frames = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=1000 + ep)  # fixed seeds = fair comparison
        done, total, speeds, crashed = False, 0.0, [], False
        while not done:
            if ep < record:
                frames.append(env.render())
            action = choose_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            speeds.append(info.get("speed", np.nan))
            crashed = crashed or bool(info.get("crashed", False))
            done = terminated or truncated
        stats.append({"reward": total, "crashed": crashed, "speed": np.nanmean(speeds)})
        print(f"Episode {ep + 1:2d}: reward {total:6.2f}  "
              f"avg speed {np.nanmean(speeds):5.1f} m/s  {'CRASH' if crashed else 'ok'}")
    return stats, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=DEFAULT_ENV)
    p.add_argument("--model", default="results/dqn_highway.zip")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--random", action="store_true", help="drive randomly (baseline)")
    p.add_argument("--video", default=None, help="video path (default: results/agent.mp4 or random.mp4)")
    p.add_argument("--video-episodes", type=int, default=3)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    record = 0 if args.no_video else args.video_episodes
    env = make_env(args.env, render_mode="rgb_array" if record else None)

    if args.random:
        label = "Random driver"
        choose_action = lambda obs: env.action_space.sample()
    else:
        from stable_baselines3 import DQN
        label = "Trained DQN driver"
        model = DQN.load(args.model)
        # deterministic=True: always take the best action (no random exploring)
        choose_action = lambda obs: int(model.predict(obs, deterministic=True)[0])

    stats, frames = run_episodes(env, choose_action, args.episodes, record)
    env.close()

    rewards = [s["reward"] for s in stats]
    crash_rate = 100 * np.mean([s["crashed"] for s in stats])
    speed = np.nanmean([s["speed"] for s in stats])
    print(f"\n{label} over {args.episodes} episodes: mean reward {np.mean(rewards):.2f}, "
          f"crash rate {crash_rate:.0f}%, avg speed {speed:.1f} m/s")

    if frames:
        import imageio
        path = args.video or ("results/random.mp4" if args.random else "results/agent.mp4")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        imageio.mimsave(path, frames, fps=5)
        print(f"Saved video ({len(frames)} frames) to {path}")


if __name__ == "__main__":
    main()

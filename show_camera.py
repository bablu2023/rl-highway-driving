"""Save a picture of what the camera-based agent actually sees.

    python show_camera.py            # -> results/camera_view.png

Top: the normal colour view of the road (what we watch in the videos).
Bottom: the 4 stacked grayscale camera frames the neural network receives.
"""
import argparse
import os

import src  # noqa: F401

import matplotlib
matplotlib.use("Agg")  # save to file, no display needed
import matplotlib.pyplot as plt

from src.env import DEFAULT_ENV, make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=DEFAULT_ENV)
    p.add_argument("--steps", type=int, default=8, help="drive a few steps first so the frames differ")
    p.add_argument("--out", default="results/camera_view.png")
    args = p.parse_args()

    env = make_env(args.env, render_mode="rgb_array", obs="camera")
    obs, info = env.reset(seed=0)
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(1)  # IDLE: just keep driving
        if terminated or truncated:
            break
    colour_view = env.render()
    env.close()

    print("Camera observation shape (frames, width, height):", obs.shape)

    fig = plt.figure(figsize=(12, 5))
    top = fig.add_subplot(2, 1, 1)
    top.imshow(colour_view)
    top.set_title("What we see (colour render)")
    top.axis("off")
    for i in range(obs.shape[0]):
        ax = fig.add_subplot(2, obs.shape[0], obs.shape[0] + i + 1)
        ax.imshow(obs[i].T, cmap="gray")  # stored as width x height, so transpose to display
        ax.set_title(f"Camera frame t-{obs.shape[0] - 1 - i}" if i < obs.shape[0] - 1 else "Camera frame (now)")
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=110)
    print("Saved", args.out)


if __name__ == "__main__":
    main()

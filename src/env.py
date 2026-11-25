"""Environment: a self-driving car on a multi-lane highway (highway-env)."""
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)  # makes "highway-v0", "highway-fast-v0", ... available

# highway-fast-v0 is the same task as highway-v0 but simulates faster,
# so training takes much less time.
DEFAULT_ENV = "highway-fast-v0"

# The car's 5 possible high-level actions (DiscreteMetaAction)
ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

# What the agent "sees":
#   "kinematics": a 5 x 5 table of positions/speeds of the nearest cars (default)
#   "camera":     grayscale top-view camera images, like a real camera feed
CAMERA_OBSERVATION = {
    "type": "GrayscaleObservation",
    "observation_shape": (128, 64),        # width x height of each camera frame (pixels)
    "stack_size": 4,                       # last 4 frames, so the agent can see motion
    "weights": [0.2989, 0.5870, 0.1140],   # how to convert RGB colours to grayscale
    "scaling": 1.75,                       # camera zoom
}
OBSERVATION_TYPES = ("kinematics", "camera")


def make_env(env_id=DEFAULT_ENV, render_mode=None, config=None, obs="kinematics"):
    """Create the driving environment.

    obs="camera" gives the agent camera images instead of the table of positions.
    render_mode="rgb_array" returns full-colour frames, which we use to record videos.
    config lets you change the road, e.g. {"lanes_count": 3, "vehicles_count": 30}.
    """
    if obs not in OBSERVATION_TYPES:
        raise ValueError(f"obs must be one of {OBSERVATION_TYPES}, got {obs!r}")
    config = dict(config or {})
    if obs == "camera":
        config["observation"] = CAMERA_OBSERVATION
    return gym.make(env_id, render_mode=render_mode, config=config or None)

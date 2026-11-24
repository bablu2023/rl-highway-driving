"""Environment: a self-driving car on a multi-lane highway (highway-env)."""
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)  # makes "highway-v0", "highway-fast-v0", ... available

# highway-fast-v0 is the same task as highway-v0 but simulates faster,
# so training takes much less time.
DEFAULT_ENV = "highway-fast-v0"

# The car's 5 possible high-level actions (DiscreteMetaAction)
ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}


def make_env(env_id=DEFAULT_ENV, render_mode=None, config=None):
    """Create the driving environment.

    render_mode="rgb_array" returns camera images, which we use to record videos.
    config lets you change the road, e.g. {"lanes_count": 3, "vehicles_count": 30}.
    """
    return gym.make(env_id, render_mode=render_mode, config=config)

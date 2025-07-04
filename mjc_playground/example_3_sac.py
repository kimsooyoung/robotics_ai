
import time
import imageio
import itertools
import numpy as np
from typing import Callable, List, NamedTuple, Optional, Union

import mediapy as media
import matplotlib.pyplot as plt

from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

from mujoco_playground import registry


env_name = "HumanoidWalk"  # @param ["AcrobotSwingup", "AcrobotSwingupSparse", "BallInCup", "CartpoleBalance", "CartpoleBalanceSparse", "CartpoleSwingup", "CartpoleSwingupSparse", "CheetahRun", "FingerSpin", "FingerTurnEasy", "FingerTurnHard", "FishSwim", "HopperHop", "HopperStand", "HumanoidStand", "HumanoidWalk", "HumanoidRun", "PendulumSwingup", "PointMass", "ReacherEasy", "ReacherHard", "SwimmerSwimmer6", "WalkerRun", "WalkerStand", "WalkerWalk"]
CAMERAS = {
    "AcrobotSwingup": "fixed",
    "AcrobotSwingupSparse": "fixed",
    "BallInCup": "cam0",
    "CartpoleBalance": "fixed",
    "CartpoleBalanceSparse": "fixed",
    "CartpoleSwingup": "fixed",
    "CartpoleSwingupSparse": "fixed",
    "CheetahRun": "side",
    "FingerSpin": "cam0",
    "FingerTurnEasy": "cam0",
    "FingerTurnHard": "cam0",
    "FishSwim": "fixed_top",
    "HopperHop": "cam0",
    "HopperStand": "cam0",
    "HumanoidStand": "side",
    "HumanoidWalk": "side",
    "HumanoidRun": "side",
    "PendulumSwingup": "fixed",
    "PointMass": "cam0",
    "ReacherEasy": "fixed",
    "ReacherHard": "fixed",
    "SwimmerSwimmer6": "tracking1",
    "WalkerRun": "side",
    "WalkerWalk": "side",
    "WalkerStand": "side",
}
camera_name = CAMERAS[env_name]

env_cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=env_cfg)

# env_cfg = registry.get_default_config('CartpoleBalance')
# env = registry.load('CartpoleBalance')

from mujoco_playground.config import dm_control_suite_params
sac_params = dm_control_suite_params.brax_sac_config(env_name)
print(f"sac_params: {sac_params}")

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
    clear_output(wait=True)

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, sac_params["num_timesteps"] * 1.25])
    plt.ylim([0, 1100])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

    plt.gcf().savefig("sac_training.png")

sac_training_params = dict(sac_params)
network_factory = sac_networks.make_sac_networks
if "network_factory" in sac_params:
  del sac_training_params["network_factory"]
  network_factory = functools.partial(
      sac_networks.make_sac_networks,
      **sac_params.network_factory
  )

train_fn = functools.partial(
    sac.train, **dict(sac_training_params),
    network_factory=network_factory,
    progress_fn=progress
)

from mujoco_playground import wrapper

# params is the model parameters
make_inference_fn, params, metrics = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# Visualize Rollout
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1

for _ in range(n_episodes):
    state = jit_reset(rng)
    rollout.append(state)
    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)

render_every = 1
frames = env.render(rollout[::render_every])
rewards = [s.reward for s in rollout]

# Save video using imageio
video_path = "rollout_video_sac.mp4"
imageio.mimsave(video_path, frames, fps=int(1.0 / env.dt / render_every))

print(f"Video saved to {video_path}")

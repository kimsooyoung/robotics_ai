import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import functools
import json
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from IPython.display import clear_output, display
from orbax import checkpoint as ocp

from mujoco_playground import manipulation, wrapper
from mujoco_playground.config import manipulation_params

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

env_name = "LeapCubeRotateZAxis"
env_cfg = manipulation.get_default_config(env_name)

ppo_params = manipulation_params.brax_ppo_config(env_name)
networks = ppo_networks.make_ppo_networks(
    **ppo_params["network_factory"]
)

# Use the correct experiment name (you can hardcode or dynamically select the latest dir)
exp_name = "LeapCubeRotateZAxis-20250704-xxxxxx"  # replace with your actual one
ckpt_path = epath.Path("checkpoints") / exp_name

with open(ckpt_path / "params.pkl", "rb") as f:
    data = pickle.load(f)

normalizer_params = data["normalizer_params"]
policy_params = data["policy_params"]
value_params = data["value_params"]

make_inference_fn = networks.make_inference_fn
inference_fn = make_inference_fn(policy_params, deterministic=True)
jit_inference_fn = jax.jit(inference_fn)

eval_env = manipulation.load(env_name, config=env_cfg)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

rng = jax.random.PRNGKey(1234)
rollout = [state := jit_reset(rng)]
actions = []
rewards = []
cube_angvel = []
cube_angacc = []
torques = []

for i in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)
    rewards.append({k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")})
    actions.append({
        "policy_output": ctrl,
        "motor_targets": state.info["motor_targets"],
    })
    torques.append(jp.linalg.norm(state.data.actuator_force))
    cube_angvel.append(env.get_cube_angvel(state.data))
    cube_angacc.append(env.get_cube_angacc(state.data))
    if state.done:
        print("Done detected, stopping rollout.")
        break
print(rollout[-1].info["success_count"])

render_every = 1
fps = 1.0 / eval_env.dt / render_every
print(f"fps: {fps}")

traj = rollout[::render_every]

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

eval_env.mj_model.stat.meansize = 0.02
eval_env.mj_model.stat.extent = 0.25
eval_env.mj_model.vis.global_.azimuth = 140
eval_env.mj_model.vis.global_.elevation = -25
frames = eval_env.render(
    traj, height=480, width=640, scene_option=scene_option, camera="side"
)
# media.show_video(frames, fps=fps)

# Save video using imageio
video_path = "rollout_video_cube_rotate.mp4"
imageio.mimsave(video_path, frames, fps=int(1.0 / env.dt))

print(f"Video saved to {video_path}")

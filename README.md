# robotics_ai
Hands-on Codes for AI based Robotics Research 



Mujoco Playground Env Setup

```
conda create -n env_mujoco_playground python=3.11

pip install mujoco
pip install mujoco_mjx
pip install brax
pip install wandb
pip install playground
pip install imageio[ffmpeg]
pip install -U "jax[cuda12]"


# Installation check
python ./mjc_playground/install_check.py
```

- Mujoco Playground Env Setup

```
conda create -n env_mujoco_playground python=3.11


sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev

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

For training_vision

```
conda create -n env_mujoco_playground_vision python=3.11
conda activate env_mujoco_playground_vision

conda update -n base -c defaults conda

conda install cuda==12.5.1 
conda install cudnn cuda-nvcc cmake
conda install xorg-xorgproto 
pip install "jax[cuda12_local]" mujoco-mjx mujoco brax

conda install conda-forge::libnvjitlink-dev
conda install -c nvidia cuda-nvcc
conda install -c conda-forge cudatoolkit-dev
conda install -c conda-forge libgcc-ng
conda install -c conda-forge compilers clang sysroot


cd <your-directory>
git clone https://github.com/shacklettbp/madrona_mjx.git
cd madrona_mjx
git submodule update --init --recursive

mkdir build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCUDAToolkit_ROOT=$CONDA_PREFIX \
  -DCMAKE_CUDA_ARCHITECTURES=all \
  -DLOAD_VULKAN=OFF

cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_C_COMPILER=clang \
         -DCMAKE_CXX_COMPILER=clang++ \
         -DCMAKE_PREFIX_PATH=$CONDA_PREFIX

make -j8

cd ..
pip install -e .
```
# Baseline A: OpenVLA without SlotSSM

The isolated Conda prefix is `.conda/envs/openvla-a`. It contains PyTorch 2.2
with CUDA 12.1, Transformers 4.40.1, TensorFlow 2.15, MuJoCo 2.3.7,
robosuite 1.4.1, and the LIBERO runtime dependencies.

From the repository root, first validate headless simulation without loading a
model:

```bash
conda activate /data/lty/code/naive-e-slotssm/.conda/envs/openvla-a
MPLCONFIGDIR=/tmp/matplotlib \
XDG_CACHE_HOME=/tmp/cache \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
python experiments/robot/libero/run_openvla_baseline.py \
  --policy dummy --max-steps 2 --output artifacts/dummy-smoke.mp4
```

For GPU inference, use EGL and the downloaded official LIBERO-Spatial model:

```bash
conda activate /data/lty/code/naive-e-slotssm/.conda/envs/openvla-a
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HOME=/data/lty/code/naive-e-slotssm/artifacts/hf-cache \
MPLCONFIGDIR=/tmp/matplotlib \
XDG_CACHE_HOME=/tmp/cache \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python experiments/robot/libero/run_openvla_baseline.py \
  --policy openvla \
  --checkpoint artifacts/checkpoints/openvla-7b-finetuned-libero-spatial \
  --task-suite libero_spatial --task-id 0 --episode-id 0 \
  --device-map balanced_low_0 --gpu-memory-gib 20 \
  --output artifacts/openvla-baseline-a.mp4
```

The script intentionally uses only the base OpenVLA policy. It does not import
or instantiate any object-slot or SlotSSM class.

`--device-map balanced_low_0` uses Accelerate to shard the model over all GPUs
exposed by `CUDA_VISIBLE_DEVICES`, while keeping extra room on GPU 0. The 20 GiB
limit leaves roughly 4 GiB per RTX 4090 for MuJoCo/EGL, inputs, and activations.

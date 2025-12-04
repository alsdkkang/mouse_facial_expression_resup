# Lab Server Specifications (abizaidlab-ml)

## Hardware Overview
- **Hostname**: `abizaidlab-ml`
- **IP Address**: `134.117.140.113`
- **OS**: Ubuntu 20.04.6 LTS (Focal Fossa)
- **CPU**: AMD Ryzen 7 5800X 8-Core Processor (16 Threads)
- **GPU**: NVIDIA GeForce RTX 3080 Ti (12GB VRAM)
- **RAM**: 32GB (Total), ~25GB Available
- **Storage**: ~1TB NVMe SSD (~29GB available on root, check `/home` or other mounts if applicable)

## Performance Notes for Deep Learning
- **GPU**: The RTX 3080 Ti is a high-end consumer GPU. It supports mixed-precision training (FP16) which should be used to maximize the 12GB VRAM.
  - *Recommended Batch Size*: 16-32 (depending on image size and model depth).
  - *CUDA Version*: 12.6 (Ensure PyTorch is compatible, e.g., PyTorch 2.x).
- **CPU**: 8 cores / 16 threads is sufficient for data loading. Ensure `num_workers` in dataloaders is set to around 4-8 to avoid bottlenecks.
- **Memory**: 32GB is decent, but monitor usage if loading large datasets into memory. Use lazy loading (reading from disk on the fly) if possible.

## Environment Setup
The server has `conda` installed.
- **Python**: 3.8+ recommended.
- **Drivers**: NVIDIA Driver 560.35.05 is installed.

### Recommended Docker/Conda Setup
```bash
# Create a new environment
conda create -n mouse_face python=3.9 -y
conda activate mouse_face

# Install PyTorch (check https://pytorch.org/ for exact command matching CUDA 12.x)
pip install torch torchvision torchaudio
```

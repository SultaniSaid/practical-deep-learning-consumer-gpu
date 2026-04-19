# Practical Deep Learning Course — Fork for DirectML GPU Benchmarking on Windows

This repository is a fork of [k3sh4v/practical-deep-learning-consumer-gpu](https://github.com/k3sh4v/practical-deep-learning-consumer-gpu).

- Original upstream README: [k3sh4v/practical-deep-learning-consumer-gpu/blob/main/README.md](https://github.com/k3sh4v/practical-deep-learning-consumer-gpu/blob/main/README.md)
- Original course: [fastai/fastbook](https://github.com/fastai/fastbook) by [Jeremy Howard](https://github.com/jph00)

## What this fork adds

This fork is focused on making the repo work well for Windows 11 on ARM, including VS Code ARM and AMD Python compatibility. It also includes new benchmark and performance tooling to show where the GPU helps most.

- A dedicated benchmark notebook: `lesson1_gpu_vs_cpu.ipynb`
- An additional GPU-favoring example with explicit performance metrics
- Local dataset caching inside `data/` so downloads are reused across runs
- DirectML Adreno GPU detection and timing tests
- CPU vs GPU comparisons for real FastAI workloads
- Helper scripts for torch-directml device validation
- Updated `requirements.txt` for this Windows ARM / AMD Python environment

## Recommended setup

1. Clone this fork:
   ```powershell
   git clone https://github.com/SultaniSaid/practical-deep-learning-consumer-gpu.git
   cd practical-deep-learning-consumer-gpu
   ```
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run notebooks from VS Code or Jupyter.

## FastAI notebooks

The repo currently includes:

- `lesson1_collabdata.ipynb` — Collab example with DirectML device support
- `lesson1_visiondata.ipynb` — Vision training examples
- `lesson1_textdata.ipynb` — Text training examples
- `lesson1_tabulardata.ipynb` — Tabular training examples
- `lesson1_segmentdata.ipynb` — Segmentation examples
- `lesson1_gpu_vs_cpu.ipynb` — GPU vs CPU benchmark lesson

## DirectML and GPU support

This fork targets `torch-directml` on Windows with supported DirectX 12 GPUs.

### DirectML-supported devices include:

- NVIDIA GPUs with DirectX 12 support
- AMD GPUs with DirectX 12 support
- Intel integrated GPUs with DirectX 12 support
- Qualcomm Adreno GPUs on Windows running DirectML

### Notes for this fork

- This repo uses `torch-directml` for GPU acceleration.
- On Windows ARM64, the Adreno GPU may be available through DirectML.
- Some ops may still fallback to CPU during training, especially optimizer updates.

## Local dataset caching

Datasets are cached locally in `data/` to avoid repeated downloads.

If you want to force a fresh download, delete the `data/` folder and rerun the notebook.

## How to verify DirectML GPU usage

1. Open Task Manager on Windows.
2. Go to the Performance tab.
3. Select the GPU that matches your DirectML device.
4. Watch GPU usage while running the notebook.

## Want to contribute?

This fork is intended to make the original repo more Windows/DirectML friendly. Contributions, issue reports, and benchmark improvements are welcome.

## License

Follow the same open-source license as the upstream repository.


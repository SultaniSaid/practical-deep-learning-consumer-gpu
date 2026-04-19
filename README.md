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
2. Install Python 3.11 AMD64, even on Windows 11 ARM64.
   - `torch-directml` does not provide native `winarm64` wheels, so using the AMD64 installer with emulation is required for compatibility.
   - This repository is tested with Python 3.11 AMD64 in VS Code on Windows 11 ARM.
3. Create and activate a Python virtual environment.
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Run notebooks from VS Code or Jupyter.

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

This repo uses `torch-directml` for GPU acceleration. DirectML allows running PyTorch on any DirectX 12 compatible GPU, including Qualcomm Adreno (ARM64), AMD, and Intel.

## Technical Architecture (v2.0 "Vanguard")

The FastAI library is primarily designed for CUDA (NVIDIA) or CPU. To bridge this gap for Windows consumer hardware, this repository includes a centralized **Vanguard Adaptation Layer** ([`dml_fastai_utils.py`](./dml_fastai_utils.py)) that performs "monkey-patching" on FastAI core classes and provides high-level adapters for Transformers.

### Key Infrastructure:
- **Normalization Patching**: Moves mean/std tensors to the DirectML device during initialization, preventing "device mismatch" errors.
- **Freeze/Unfreeze Stability**: Overrides `Learner.freeze_to` to safely manage gradients and clear optimizer states, supporting both standard models and HuggingFace wrappers.
- **HuggingFace Synergy**: Includes `HFModelWrapper` and `HFCallback` to bridge HuggingFace models into FastAI with full DirectML device awareness (handling attention masks and logit extraction).
- **Portability Layer**: Standardized data path handling via `get_local_path()` ensures your notebooks work out-of-the-box on any Windows machine.

### Caveats & Performance

- **Mixed Precision**: `to_fp16()` is currently disabled in most learners because DirectML support for half-precision operators varies across hardware vendors.
- **Operator Fallbacks**: If you see warnings like `The operator 'aten::...' will fall back to run on the CPU`, it means DirectML does not have a native implementation for that specific operation yet. These parts of the model will run on your CPU, which may cause performance slowdowns.

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


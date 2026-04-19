# Fine-Tuning Gemma-4-E4B-it on Adreno GPU (DirectML)

This guide Adapt our **"DirectML Vanguard"** breakthroughs for the **Gemma-4** family of multimodal models.

## 🏆 The Gemma-4 Training Breakthrough

Training the brand-new **Gemma-4-E4B-it** (Effective 4 Billion parameters) on Windows ARM64 presents extreme memory and architectural challenges. Without specific repairs, the model will either crash (OOM) or "Freeze" (Loss stays flat).

### The Fix: Vanguard for Multimodal Transformers
Our dedicated training script `train_gemma4_dml.py` implements the following **Gemma-4 repairs**:
1.  **Architecture Unlock**: Uses `trust_remote_code=True` to enable the brand-new `gemma4` multimodal layers.
2.  **Vanguard Adapter Re-Sync**: After moving the 4B parameter model to the DirectML device, we manually iterate through the Hybrid-Attention layers to forcefully re-enable gradients for the **LoRA adapters**.
3.  **Memory Hardening**: High-count **Gradient Accumulation (8)** is used to simulate stable training while keeping the physical VRAM footprint within the 16GB limit of the Snapdragon X Elite.

---

## 🚀 Quick Start: Fine-Tuning Gemma-4

### Prerequisites
Ensure you have the multimodal LLM stack installed:
```bash
pip install torch-directml transformers peft datasets
```

### Running the Vanguard Training
We have provided a unified script that handles the repairs and multimodal initialization automatically:

#### A) Train on Adreno GPU (16GB VRAM Stress Test)
```bash
python train_gemma4_dml.py --device gpu --steps 5
```

#### B) Train on CPU-only (Stability Path)
```bash
python train_gemma4_dml.py --device cpu --steps 5
```

---

## 📊 Hardware Considerations
*   **VRAM Pressure**: Gemma-4-E4B in FP32 consumes ~16GB. You may see temporary system slowdowns during the backward pass on 16GB RAM devices.
*   **Disk Space**: The model requires ~8GB of free space. We have cleared the Hugging Face cache to accommodate this.
*   **Accuracy**: Within 5-10 steps, you should see the training loss begin to descend. This confirms that the Vanguard patch has "unlocked" the weights for learning.

---

*Verified on Qualcomm Snapdragon X Elite (Adreno X1-85 GPU)*

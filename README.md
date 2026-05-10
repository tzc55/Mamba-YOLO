# Mamba-YOLO: A 660-FPS NMS-Free Spatio-Temporal Architecture for Real-Time UAV Micro-Fire Detection

Official PyTorch implementation of the **Mamba-YOLO** architecture.

## 🔔 Open Source Status
> **Note:** The full paper is currently under peer review.

To facilitate the review process and demonstrate the authenticity of our proposed architecture, we have adopted a phased release strategy:

* ✅ **Core Architecture (Available Now):** We have released the core network implementation (`mamba_yolo_arch.py`) and the architectural configuration files (`.yaml`). These files detail the exact mathematical operations and PyTorch implementations of our core contributions:
  * The **Mamba-3** based Time-Series State Observer.
  * The **Spatial Attention Handshake Protocol**.
  * The high-resolution **P2 micro-target prediction head** and the ablation configuration.
* ⏳ **Training Pipeline & Weights (Coming Soon):** The complete end-to-end training scripts, inference codes, and pre-trained weights are currently running on our internal servers and undergoing code-cleaning. They will be fully released immediately upon the acceptance of the paper.
* ⏳ **Tiny Fire Benchmark (Coming Soon):** The highly curated sub-pixel fire dataset used in our experiments will be made available simultaneously with the full code release.

## 📌 Architecture Overview
Mamba-YOLO is a spatio-temporal closed-loop architecture designed for extreme early-stage wildland fire detection on resource-constrained UAVs. It achieves an unprecedented single-frame inference latency of **2.16 ms (over 660 FPS)** while strictly maintaining zero-miss detection for micro-targets.

*(Optional: Insert your architecture diagram here)*

## 📧 Contact
During the review process, the complete uncleaned codebase is available from the corresponding author on reasonable request.

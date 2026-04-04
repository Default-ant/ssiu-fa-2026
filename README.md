# SSIU-FA: Structural Similarity-Inspired Unfolding with Frequency-Aware Attention

This project is a high-performance, lightweight implementation and improvement of the 2025 IEEE Transactions on Image Processing (TIP) paper: **\"Structural Similarity-Inspired Unfolding for Lightweight Image Super-Resolution\" (Ni et al., June 2025).**

## 🎯 Our Objective
We aim to **beat the published SSIU results** by introducing **Frequency-Aware Attention (FA-ESAM)**. 

### **Published SSIU (2025) Baseline (x4):**
| Dataset | PSNR | SSIM | MS-SSIM |
| :--- | :--- | :--- | :--- |
| **Set5** | 32.18 dB | 0.8950 | 0.9808 |

---

## 🚀 Key Differences & Improvements
1.  **Base SSIU Architecture**: Implementing Mixed-Scale Gating Modules (MSGM) according to the June 2025 methodology.
2.  **Our Improvement (FA-ESAM)**: Adding **Frequency-Aware Priors** to the Sparse Attention Module. This focuses the network's energy on high-frequency textured regions, improving both PSNR and edge clarity.
3.  **High-Efficiency Training**: Optimized to converge on a subset of DIV2K/Flickr2K in **under 2 hours**.

## 📁 Repository Structure
- `ssiu_model.py`: Core implementation of the SSIU architecture (MSGM, ESAM modules).
- `improved_ssiu.py`: Integration of our Frequency-Aware enhancements.
- `train_ssiu.py`: High-efficiency training script with 2025 optimizations.
- `validate_ssiu.py`: Multi-dataset validation script (Set5, Set14, BSD100, etc.).

---
*(Current Status: Implementing the base architecture...)*

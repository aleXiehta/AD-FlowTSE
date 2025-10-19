# 🌀 AD-FlowTSE: Adaptive Discriminative Flow-Matching Target Speaker Extraction

**ICASSP 2026 Submission**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b.svg)](https://arxiv.org/)
<!-- [![Conference](https://img.shields.io/badge/ICASSP-2026-orange.svg)] -->


## 🧠 Introduction

Generative target-speaker extraction (TSE) methods often produce more natural outputs than predictive models.
Recent diffusion- or flow-matching-based approaches typically rely on a **fixed number of reverse steps** with **uniform step size**.

We introduce **Adaptive Discriminative Flow Matching TSE (AD-FlowTSE)** — a generative framework that extracts target speech with an *adaptive* step size.

Unlike prior FM-based speech enhancement and TSE methods that transport between the **mixture (or a normal prior)** and the **clean-speech distribution**, AD-FlowTSE defines the flow between the **background** and the **source**, governed by the *mixing ratio (MR)* of the source and background forming the mixture.

This design enables **MR-aware initialization**, where the model starts at an adaptive point along the background–source trajectory rather than using a fixed reverse schedule across all noise levels.

💡 *Experiments show that AD-FlowTSE achieves strong performance even with a single reverse step, and auxiliary MR estimation further enhances accuracy.*
Experimental results highlight that aligning the transport path with mixture composition and adapting step size to noise conditions yields **efficient and accurate TSE**.


## 🧩 Dataset Preparation

Follow the official data-preparation pipeline from [**SpeakerBeam**](https://github.com/BUTSpeechFIT/speakerbeam).
After preparation, ensure your dataset follows the same directory structure (mixture, clean, and reference files).


## 🏋️ Pre-trained Checkpoints

Pre-trained **AD-FlowTSE** models and **mixing-ratio predictors** are available in the `exp/` folder.
You can fine-tune or directly evaluate them using your dataset.


## ⚙️ Training Instructions

### 🔹 1️⃣ Train the Mixing-Ratio Predictor

```bash
python train_t_predicter.py \
  --config config/<config_FlowTSE_alpha.yaml | config_FlowTSE_alpha_noisy.yaml>
```

### 🔹 2️⃣ Train AD-FlowTSE

```bash
python train.py \
  --config config/<config_FlowTSE_large.yaml | config_FlowTSE_large_noisy.yaml>
```

---

## 🧪 Evaluation

Run evaluation with different MR-predictor variants:

```bash
python eval.py \
  --config config/<config_FlowTSE_large.yaml | config_FlowTSE_large_noisy.yaml> \
  --t_predicter <ECAPAMLP | GT | ZERO | ONE | RAND>
```

## 🧰 Credits

Our **UDiT** backbone is ported and modified from
[**SoloAudio**](https://github.com/WangHelin1997/SoloAudio/tree/main/model).
We thank the authors for releasing their high-quality implementation.


## 📖 Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{hsieh2026adflowtse,
  title     = {Adaptive Discriminative Flow Matching for Target Speaker Extraction},
  author    = {Tsun-An Hsieh and Paris Smaragdis and Minje Kim},
  booktitle = {submitted to Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026},
}
```
# AD-FlowTSE: Adaptive Discriminative Flow-Matching Target Speaker Extraction

**ICASSP 2026 Submission**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2510.16995-b31b1b.svg)](https://arxiv.org/abs/2510.16995)
[![Demo](https://img.shields.io/badge/Demo-Online-4caf50.svg)](https://alexiehta.github.io/demo/ad_flowtse/ad_flowtse_demo.html)
<!-- [![Conference](https://img.shields.io/badge/ICASSP-2026-orange.svg)] -->


## Overview

Generative target-speaker extraction (TSE) methods often produce more natural outputs than predictive models.
Recent diffusion- or flow-matching-based approaches typically rely on a **fixed number of reverse steps** with **uniform step size**.

We introduce **Adaptive Discriminative Flow Matching TSE (AD-FlowTSE)** — a generative framework that extracts target speech with an *adaptive* step size.

Unlike prior FM-based speech enhancement and TSE methods that transport between the **mixture (or a normal prior)** and the **clean-speech distribution**, AD-FlowTSE defines the flow between the **background** and the **source**, governed by the *mixing ratio (MR)* of the source and background forming the mixture.

This design enables **MR-aware initialization**, where the model starts at an adaptive point along the background–source trajectory rather than using a fixed reverse schedule across all noise levels.

💡 Experiments show that AD-FlowTSE delivers efficient and accurate TSE by achieving strong performance even with a single reverse step, further enhanced by auxiliary MR estimation, path alignment with mixture composition, and noise-adaptive step sizes.


## Dataset Preparation

Follow the official data-preparation pipeline from [**SpeakerBeam**](https://github.com/BUTSpeechFIT/speakerbeam).
After preparation, ensure your dataset follows the same directory structure (mixture, clean, and reference files).


## Pre-trained Checkpoints

Pre-trained **AD-FlowTSE** models and **mixing-ratio predictors** are available [here](https://uillinoisedu-my.sharepoint.com/:f:/g/personal/tsunanh2_illinois_edu/IgAwyb6PVqp1R79MTY1xuOFyAcRg1cYn_jagtB0wotmXlCg?e=Haqhj7).


## Training Instructions

### Train the Mixing-Ratio Predictor

```bash
python train_t_predicter.py \
  --config config/<config_FlowTSE_alpha.yaml | config_FlowTSE_alpha_noisy.yaml>
```

### Train AD-FlowTSE

```bash
python train.py \
  --config config/<config_FlowTSE_large.yaml | config_FlowTSE_large_noisy.yaml>
```

---

## Evaluation

Run evaluation with different MR-predictor variants:

```bash
python eval.py \
  --config config/<config_FlowTSE_large.yaml | config_FlowTSE_large_noisy.yaml> \
  --t_predicter <ECAPAMLP | GT | ZERO | ONE | RAND>
```

## Credits

Our **UDiT** backbone is ported and modified from
[**SoloAudio**](https://github.com/WangHelin1997/SoloAudio/tree/main/model).
We thank the authors for releasing their high-quality implementation.


## Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{hsieh2026adflowtse,
  title     = {Adaptive Deterministic Flow Matching for Target Speaker Extraction},
  author    = {Tsun-An Hsieh and Minje Kim},
  booktitle = {Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026},
}
```

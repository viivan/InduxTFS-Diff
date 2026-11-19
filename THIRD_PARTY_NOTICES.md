# Third-Party Notices

This software includes components that are derived from or reimplement the
following third-party open-source projects. Each component remains under its
original license, and the copyright notices and license terms below are kept
as required by those licenses.

## 1. DiT (Diffusion Transformer)

- File: `models/DiT.py`
- Source: https://github.com/facebookresearch/DiT
- Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0), https://creativecommons.org/licenses/by-nc/4.0/
- Note: This component is licensed for **non-commercial use only**. It is
  distributed here with the original copyright header retained. The MIT
  license of this repository does not apply to this file.
- References used by the original work:
  - GLIDE: https://github.com/openai/glide-text2im
  - MAE: https://github.com/facebookresearch/mae

## 2. TimeGAN

- Files: `models/TimeGAN.py`, `evaluation/discriminative_metric.py`
- Paper: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
  "Time-series Generative Adversarial Networks," NeurIPS 2019,
  https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
- Official reference implementation: https://github.com/jsyoon0823/TimeGAN
- License (official implementation): Apache License 2.0,
  https://www.apache.org/licenses/LICENSE-2.0
- The PyTorch version in this repository is adapted from publicly available
  PyTorch reimplementations of the same architecture.

## 3. WaveStitch

- Files: `models/WaveStitch.py`, `models/SSSD.py`, `models/s4.py`
- Source: https://github.com/adis98/WaveStitch
- Copyright (c) 2024 adis98
- License: MIT License

## 4. TS2Vec

- Directory: `evaluation/ts2vec/`
- Source: https://github.com/yuezhihan/ts2vec
- License: MIT License

## 5. Structured State Space (S4)

- File: `models/s4.py`
- Source: https://github.com/state-spaces/s4
- License: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0
- The optional CUDA Cauchy extension and the `pykeops` backend are
  accelerators; when neither is available the code falls back to a pure
  PyTorch implementation.

## 6. v-diffusion (reference architecture)

- Referenced in: `models/InduxTFS_Diff.py`, `models/Diff_MTS.py`,
  `models/MetaIndux_TS.py`
- Source: https://github.com/crowsonkb/v-diffusion-jax
- License: MIT License

---

If you redistribute this software, preserve this file together with the
copyright headers embedded in the individual source files.

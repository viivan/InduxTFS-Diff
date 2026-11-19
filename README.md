# InduxTFS-Diff

A PyTorch framework for generating multivariate industrial time series. It
bundles a diffusion-based generator with several baseline generative
architectures behind a single command-line interface, covering training,
sampling, and evaluation on fixed-length sliding windows. No pretrained
checkpoints or datasets are included.

## 1. Setup

- **Python**: 3.11 recommended.
- **Dependencies**:

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended; CPU works for small runs. The S4 Cauchy kernel uses
`pykeops` when available and otherwise falls back to a pure-PyTorch
implementation.

## 2. Data

Datasets are not distributed with this repository. Place the files yourself
under the project-level `data/` directory.

- **C-MAPSS** (e.g. `FD001`, `FD004`):

```
data/train_<FD>.txt
data/test_<FD>.txt
data/RUL_<FD>.txt
```

- **FEMTO Bearing**: place the dataset under `data/FEMTOBearingDataSet/` as
  expected by `data/FEMTODataset.py`.

## 3. Operating main.py

`main.py` is the single entry point and is controlled by command-line
arguments.

### Key arguments

- `--model_name`: model architecture. Main model: `InduxTFS_Diff`. Variants:
  `InduxTFS_NoTSME`, `InduxTFS_FixedFusion`, `InduxTFS_NoLTA`,
  `InduxTFS_NoFreq`, `InduxTFS_NoWavelet`. Baselines: `Diff_MTS`,
  `MetaIndux_TS`, `DiT`, `DiffWave`, `SSSD`, `WaveStitch`, `TimeGAN`.
- `--dataset`: subset identifier, e.g. `FD001`, `FD004`, `FEMTO`.
- `--state`: operation mode — `train`, `sample`, `eval`, `all`.
- `--window_size`: sliding-window length (e.g. `48`).
- `--input_size`: number of sensor channels (e.g. `14`).
- `--T`: number of diffusion timesteps (e.g. `500`).
- `--epoch`, `--lr`: training epochs and learning rate.
- `-b`, `--batch-size`: batch size (default `64`).
- `--sample_type`: `ddpm` or `ddim`.

### 3.1 Train

Trains the model, then generates synthetic samples from the training labels.
Checkpoints are saved under `outputs/`.

```bash
python main.py --dataset FD001 --model_name InduxTFS_Diff --state train \
    --window_size 48 --input_size 14 --T 500 --epoch 50 --lr 2e-3
```

### 3.2 Sample

Loads a trained checkpoint and generates samples, then runs evaluation.
Synthetic data is saved as `.npz` under `outputs/syn_data/`.

```bash
python main.py --dataset FD001 --model_name InduxTFS_Diff --state sample \
    --window_size 48 --input_size 14 --T 500
```

### 3.3 Evaluate

Loads previously generated samples and reports predictive, discriminative,
context-FID, and downstream classifier scores.

```bash
python main.py --dataset FD001 --model_name InduxTFS_Diff --state eval \
    --window_size 48 --input_size 14
```

### 3.4 All-in-one

Runs training, sampling, and evaluation in sequence.

```bash
python main.py --dataset FD001 --model_name InduxTFS_Diff --state all \
    --window_size 48 --input_size 14 --T 500 --epoch 50 --lr 2e-3
```

## 4. Default run

Running without arguments uses the built-in defaults (`InduxTFS_Diff`,
`FD004`, `state=all`, window `48`, `T=500`, `epoch=50`):

```bash
python main.py
```

## License

Released under the MIT License; see [LICENSE](LICENSE). Some files derive from
third-party projects and retain their original licenses and notices — in
particular, `models/DiT.py` is licensed under CC BY-NC 4.0 (non-commercial use
only). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

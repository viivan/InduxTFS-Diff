import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

ROOT = Path(__file__).resolve().parent / 'FEMTOBearingDataSet'

TRAIN_BEARINGS = ['Bearing1_2', 'Bearing2_1', 'Bearing2_2', 'Bearing3_1', 'Bearing3_2']
HOLDOUT_BEARINGS = ['Bearing1_1']

RUL_CAP_S = 125.0
FILE_DURATION_S = 0.1
SR_HZ = 25600
CACHE_DIR = ROOT / 'cache_femto'

FEATURE_NAMES = [
    'ax_rms', 'ay_rms',
    'ax_peak', 'ay_peak',
    'ax_ptp', 'ay_ptp',
    'ax_kurt', 'ay_kurt',
    'ax_scent', 'ay_scent',
    'ax_dommag', 'ay_dommag',
    'ax_shape', 'ay_shape',
]


def extract_features_per_file(acc):
    feats = []
    for ch in range(2):
        x = acc[:, ch].astype(np.float64)
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        ptp = float(np.max(x) - np.min(x))
        s = float(x.std())
        kurt = float(np.mean(((x - x.mean()) / (s + 1e-12)) ** 4)) - 3.0
        X = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / SR_HZ)
        centroid = float(np.sum(freqs * X) / (np.sum(X) + 1e-12))
        dom_mag = float(X.max())
        shape = float(rms / (np.mean(np.abs(x)) + 1e-12))
        feats += [rms, peak, ptp, kurt, centroid, dom_mag, shape]
    return np.array(feats, dtype=np.float32)


def load_bearing_features(bearing, cache_dir=CACHE_DIR, force=False):
    cache_file = cache_dir / f'features_{bearing}.npz'
    if cache_file.exists() and not force:
        return np.load(cache_file)['feats']

    d = ROOT / 'Training_set' / 'Learning_set' / bearing
    if not d.exists():
        raise FileNotFoundError(f'未找到轴承目录: {d}')
    files = sorted(d.glob('acc_*.csv'))
    if len(files) == 0:
        raise FileNotFoundError(f'{d} 下没有 acc_*.csv 文件')
    feats = []
    for i, f in enumerate(files):
        df = pd.read_csv(f, header=None)
        acc = df.iloc[:, 4:6].values.astype(np.float32)
        feats.append(extract_features_per_file(acc))
        if (i + 1) % 400 == 0:
            print(f'    {bearing}: {i+1}/{len(files)}')
    feats = np.stack(feats, 0).astype(np.float32)
    cache_dir.mkdir(exist_ok=True)
    np.savez(cache_file, feats=feats)
    print(f'    已缓存 {bearing}: {feats.shape}')
    return feats


class FEMTODataset:

    def __init__(self, sequence_length=80, train_bearings=None, holdout_bearings=None,
                 rul_cap_s=RUL_CAP_S, feature_columns=FEATURE_NAMES):
        self.sequence_length = sequence_length
        self.train_bearings = train_bearings if train_bearings is not None else TRAIN_BEARINGS
        self.holdout_bearings = (holdout_bearings if holdout_bearings is not None
                                 else HOLDOUT_BEARINGS)
        self.rul_cap_s = rul_cap_s
        self.feature_columns = feature_columns
        self.train_data = None
        self.test_data = None

        self._build()

    def _build(self):
        rows = []
        for bearing in self.train_bearings + self.holdout_bearings:
            feats = load_bearing_features(bearing)
            n = feats.shape[0]
            rul_uncapped = (np.arange(n - 1, -1, -1, dtype=np.float32)
                            * FILE_DURATION_S)
            df = pd.DataFrame(feats, columns=self.feature_columns)
            df['id'] = bearing
            df['step'] = np.arange(n)
            df['RUL'] = rul_uncapped
            rows.append(df)
        all_df = pd.concat(rows, ignore_index=True)

        scaler = MinMaxScaler()
        train_mask = all_df['id'].isin(self.train_bearings)
        cols_norm = self.feature_columns
        scaler.fit(all_df.loc[train_mask, cols_norm].values)
        all_df.loc[:, cols_norm] = scaler.transform(all_df.loc[:, cols_norm].values)

        self.train_data = all_df[train_mask].reset_index(drop=True)
        self.test_data = all_df[~train_mask].reset_index(drop=True)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def _slice(self, data, eval_only=False):
        feature_list, label_list = [], []
        for bearing in sorted(data['id'].unique()):
            d = data[data['id'] == bearing]
            vals = d[self.feature_columns].values
            ruls = d['RUL'].values
            if len(vals) < self.sequence_length:
                continue
            for j in range(0, len(vals) - self.sequence_length + 1):
                label = float(ruls[j + self.sequence_length - 1])
                if eval_only and label > self.rul_cap_s:
                    continue
                feature_list.append(vals[j:j + self.sequence_length])
                label_list.append(min(label, self.rul_cap_s))
        if len(feature_list) == 0:
            return (torch.empty(0), torch.empty(0))
        return (torch.from_numpy(np.array(feature_list, dtype=np.float32)),
                torch.from_numpy(np.array(label_list, dtype=np.float32).reshape(-1, 1)))

    def get_feature_slice(self, data):
        return self._slice(data, eval_only=False)[0]

    def get_label_slice(self, data):
        return self._slice(data, eval_only=False)[1]

    def get_eval_slice(self, data):
        return self._slice(data, eval_only=True)

    def get_last_data_slice(self, data):
        feature_list, label_list = [], []
        for bearing in sorted(data['id'].unique()):
            d = data[data['id'] == bearing]
            vals = d[self.feature_columns].values
            ruls = d['RUL'].values
            if len(vals) == 0:
                continue
            if len(vals) < self.sequence_length:
                pad = np.repeat(vals[:1], self.sequence_length - len(vals), axis=0)
                vals = np.concatenate([pad, vals], axis=0)
            feature_list.append(vals[-self.sequence_length:])
            label_list.append(min(float(ruls[-1]), self.rul_cap_s))
        return (torch.from_numpy(np.array(feature_list, dtype=np.float32)),
                torch.from_numpy(np.array(label_list, dtype=np.float32).reshape(-1, 1)))

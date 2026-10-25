"""EnvisionHGdetector models without TensorFlow, Holistic or the package's pins (runs on macOS).

Port of the two classifiers in envisionhgdetector 3.0.x (Pouw, Shaikh et al.; MIT License, Copyright (c) 2024
Wim Pouw; github.com/WimPouw/envisionhgdetector):
  CNN       25 frames x 92 world-landmark values -> [P(motion), P(Gesture | motion), P(Move | motion)]
  LightGBM  5 frames -> 100 engineered features -> P(Gesture)
Both read MediaPipe pose world landmarks 0-22 as (x, y, z, visibility). The CNN forward pass is numpy; the
LightGBM model is a plain Booster text file plus the scaler's mean and scale, so no pickle crosses versions.

One-time export, on a machine where the package is installed (only h5py, joblib, lightgbm, sklearn needed):
    python envision_port.py export
Check the port against the package (needs the package and TensorFlow):
    python envision_port.py check
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models' / 'envision'
CNN_NPZ = MODELS / 'cnn_default.npz'
LGBM_TXT = MODELS / 'lightgbm_default.txt'
LGBM_META = MODELS / 'lightgbm_default.json'

SEQ_CNN, SEQ_LGBM, N_LANDMARKS = 25, 5, 23
KEY_JOINTS = [11, 12, 13, 14, 15, 16]          # shoulders, elbows, wrists
L_WRIST, R_WRIST = 15, 16
VIS_LANDMARKS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
UPPER_VIS = [11, 12, 13, 14, 15, 16]
BN_EPS = 1e-3                                   # Keras BatchNormalization default


def package_dir():
    import importlib.util
    spec = importlib.util.find_spec('envisionhgdetector')
    if spec is None:
        raise SystemExit('envisionhgdetector is not installed here; export on a machine that has it')
    return Path(spec.submodule_search_locations[0])


def export():
    """Write the CNN weights (.npz) and the LightGBM booster (.txt + scaler .json) from the package files."""
    import h5py
    import joblib
    src = package_dir() / 'model'
    MODELS.mkdir(parents=True, exist_ok=True)
    weights = {}
    with h5py.File(src / 'cnn_default.h5', 'r') as f:
        f.visititems(lambda name, obj: weights.__setitem__(name.split('/')[0] + '.' + name.split('/')[-1][:-2],
                                                           np.asarray(obj)) if isinstance(obj, h5py.Dataset) else None)
    np.savez(CNN_NPZ, **weights)
    data = joblib.load(src / 'lightgbm_default.pkl')
    model = data['model']
    booster = model.booster_ if hasattr(model, 'booster_') else model
    booster.save_model(str(LGBM_TXT))
    classes = [str(c) for c in data['label_encoder'].classes_] if data.get('label_encoder') is not None else None
    LGBM_META.write_text(json.dumps({'mean': data['scaler'].mean_.tolist(), 'scale': data['scaler'].scale_.tolist(),
                                     'classes': classes}), encoding='utf-8')
    print(f'wrote {CNN_NPZ.name} ({len(weights)} arrays), {LGBM_TXT.name}, {LGBM_META.name}; classes {classes}')


def _conv_same(x, kernel, bias, stride=1):
    """Keras Conv1D, padding='same': x (L, C_in), kernel (K, C_in, C_out)."""
    k = kernel.shape[0]
    n_out = -(-len(x) // stride)
    pad = max((n_out - 1) * stride + k - len(x), 0)
    xp = np.pad(x, ((pad // 2, pad - pad // 2), (0, 0)))
    out = sum(xp[i:i + (n_out - 1) * stride + 1:stride] @ kernel[i] for i in range(k))
    return out + bias


def _maxpool_same(x):
    """Keras MaxPooling1D(pool_size=2, strides=2, padding='same')."""
    if len(x) % 2:
        x = np.vstack([x, x[-1:]])              # the pad is ignored by max, so repeating the last row is equal
    return x.reshape(-1, 2, x.shape[1]).max(axis=1)


class EnvisionCNN:
    """Hierarchical CNN: motion head (sigmoid) and Gesture/Move head (softmax), from 25 x 92 world landmarks."""

    def __init__(self, path=CNN_NPZ):
        w = dict(np.load(path))
        self.blocks = []
        for i in range(3):
            def bn(j):
                n = 'batch_normalization' + (f'_{j}' if j else '')
                scale = w[n + '.gamma'] / np.sqrt(w[n + '.moving_variance'] + BN_EPS)
                return scale, w[n + '.beta'] - w[n + '.moving_mean'] * scale
            def conv(j):
                n = 'conv1d' + (f'_{j}' if j else '')
                return w[n + '.kernel'], w[n + '.bias']
            self.blocks.append((conv(2 * i), bn(2 * i), conv(2 * i + 1), bn(2 * i + 1)))
        self.dense = (w['dense.kernel'], w['dense.bias'])
        self.motion = (w['has_motion.kernel'], w['has_motion.bias'])
        self.kind = (w['gesture_probs.kernel'], w['gesture_probs.bias'])

    def __call__(self, seq):
        """seq (25, 92) -> {'motion', 'gesture', 'move'}: P(motion) and P(Gesture | motion), P(Move | motion)."""
        x = np.asarray(seq, dtype=np.float32)
        for (k, b), (s, t), (k_sc, b_sc), (s_sc, t_sc) in self.blocks:
            main = _maxpool_same(np.maximum(_conv_same(x, k, b) * s + t, 0))
            short = _conv_same(x, k_sc, b_sc, stride=2) * s_sc + t_sc
            x = np.maximum(main + short, 0)
        h = np.maximum(np.concatenate([x.mean(axis=0), x.max(axis=0)]) @ self.dense[0] + self.dense[1], 0)
        motion = 1 / (1 + np.exp(-(h @ self.motion[0] + self.motion[1])[0]))
        z = h @ self.kind[0] + self.kind[1]
        z = np.exp(z - z.max())
        z /= z.sum()
        return {'motion': float(motion), 'gesture': float(z[0]), 'move': float(z[1])}


def lgbm_features(video, window_size=SEQ_LGBM, stride=2):
    """(n_frames, 92) world landmarks -> (n_windows, 100), the package's vectorized extract_sequence_features."""
    video = np.asarray(video, dtype=np.float32)
    n_frames = len(video)
    if n_frames < window_size:
        return np.empty((0, 100), dtype=np.float32)
    seq_4d = video.reshape(n_frames, N_LANDMARKS, 4)
    seq_3d, visibility = seq_4d[:, :, :3], seq_4d[:, :, 3]
    kj_flat = seq_3d[:, KEY_JOINTS, :].reshape(n_frames, -1)
    starts = np.arange(0, n_frames - window_size + 1, stride)
    ends = starts + window_size - 1
    out = np.zeros((len(starts), 100), dtype=np.float32)

    def window_mean(arr):
        cumsum = np.vstack([np.zeros((1, arr.shape[1])), np.cumsum(arr, axis=0)])
        return (cumsum[ends + 1] - cumsum[starts]) / window_size

    def norm(a):
        return np.linalg.norm(a, axis=1, keepdims=True)

    out[:, 0:18] = window_mean(kj_flat)
    frame_vels = np.zeros_like(kj_flat)
    frame_vels[1:] = kj_flat[1:] - kj_flat[:-1]
    out[:, 18:36] = window_mean(frame_vels)
    out[:, 36] = window_mean(norm(frame_vels[:, 12:15]))[:, 0]
    out[:, 37] = window_mean(norm(frame_vels[:, 15:18]))[:, 0]
    if window_size >= 3:
        wrist_windows = np.lib.stride_tricks.sliding_window_view(kj_flat[:, 12:18], window_size, axis=0)[starts]
        out[:, 38:44] = np.ptp(wrist_windows, axis=2)
    l_wrist, r_wrist = seq_3d[:, L_WRIST, :], seq_3d[:, R_WRIST, :]
    l_has, r_has = np.any(l_wrist, axis=1, keepdims=True), np.any(r_wrist, axis=1, keepdims=True)
    l_pinky = np.where(l_has, seq_3d[:, 17, :] - l_wrist, 0.0)
    l_index = np.where(l_has, seq_3d[:, 19, :] - l_wrist, 0.0)
    l_thumb = np.where(l_has, seq_3d[:, 21, :] - l_wrist, 0.0)
    r_pinky = np.where(r_has, seq_3d[:, 18, :] - r_wrist, 0.0)
    r_index = np.where(r_has, seq_3d[:, 20, :] - r_wrist, 0.0)
    r_thumb = np.where(r_has, seq_3d[:, 22, :] - r_wrist, 0.0)
    for j, a in enumerate([l_pinky, l_index, l_thumb, r_pinky, r_index, r_thumb]):
        out[:, 44 + 3 * j:47 + 3 * j] = window_mean(a)
    for j, (a, b) in enumerate([(l_pinky, l_thumb), (l_index, l_thumb), (l_pinky, l_index),
                                (r_pinky, r_thumb), (r_index, r_thumb), (r_pinky, r_index)]):
        out[:, 62 + j] = window_mean(norm(a - b))[:, 0]
    frame_accels = np.zeros_like(kj_flat)
    frame_accels[2:] = kj_flat[2:] - 2 * kj_flat[1:-1] + kj_flat[:-2]
    out[:, 68] = window_mean(norm(frame_accels[:, 12:15]))[:, 0]
    out[:, 69] = window_mean(norm(frame_accels[:, 15:18]))[:, 0]
    if n_frames > 1 and window_size > 2:
        vels = np.diff(kj_flat, axis=0)
        lw_speed, rw_speed = np.linalg.norm(vels[:, 12:15], axis=1), np.linalg.norm(vels[:, 15:18], axis=1)
        if len(lw_speed) >= window_size - 1:
            out[:, 70] = np.std(np.lib.stride_tricks.sliding_window_view(lw_speed, window_size - 1)[starts], axis=1)
            out[:, 71] = np.std(np.lib.stride_tricks.sliding_window_view(rw_speed, window_size - 1)[starts], axis=1)
    l_sh, r_sh = seq_3d[:, 11, :], seq_3d[:, 12, :]
    l_sh_has, r_sh_has = np.any(l_sh, axis=1, keepdims=True), np.any(r_sh, axis=1, keepdims=True)
    out[:, 72] = window_mean(np.where(l_sh_has, l_wrist[:, 1:2] - l_sh[:, 1:2], 0.0))[:, 0]
    out[:, 73] = window_mean(np.where(r_sh_has, r_wrist[:, 1:2] - r_sh[:, 1:2], 0.0))[:, 0]
    both = (np.any(l_wrist, axis=1) & np.any(r_wrist, axis=1))[:, None]
    out[:, 74] = window_mean(np.where(both, norm(l_wrist - r_wrist), 0.0))[:, 0]
    out[:, 75] = window_mean(np.where(l_sh_has, norm(l_wrist - l_sh), 0.0))[:, 0]
    out[:, 76] = window_mean(np.where(r_sh_has, norm(r_wrist - r_sh), 0.0))[:, 0]
    if n_frames > 1:
        lw_disp = np.linalg.norm(np.diff(l_wrist, axis=0), axis=1)
        rw_disp = np.linalg.norm(np.diff(r_wrist, axis=0), axis=1)
        cs = np.concatenate([[0.0], np.cumsum(lw_disp + rw_disp)])
        out[:, 77] = cs[ends] - cs[starts]
    center = (l_sh + r_sh) / 2
    mirror = np.array([-1, 1, 1], dtype=np.float32)
    out[:, 78] = window_mean(np.where(both, 1.0 / (1.0 + norm((l_wrist - center) + (r_wrist - center) * mirror)),
                                      0.0))[:, 0]
    if n_frames > 1:
        cs_l = np.concatenate([[0.0], np.cumsum(lw_disp)])
        cs_r = np.concatenate([[0.0], np.cumsum(rw_disp)])
        lm, rm = cs_l[ends] - cs_l[starts], cs_r[ends] - cs_r[starts]
        out[:, 79] = np.abs(lm - rm) / (lm + rm + 1e-6)
    out[:, 80:92] = window_mean(visibility[:, VIS_LANDMARKS])
    out[:, 92:98] = window_mean(visibility[:, UPPER_VIS])
    out[:, 98] = np.min(np.lib.stride_tricks.sliding_window_view(visibility[:, 15], window_size)[starts], axis=1)
    out[:, 99] = np.min(np.lib.stride_tricks.sliding_window_view(visibility[:, 16], window_size)[starts], axis=1)
    return out


class EnvisionLGBM:
    """LightGBM on the 100 engineered features of the last 5 frames -> P(Gesture)."""

    def __init__(self, model_path=LGBM_TXT, meta_path=LGBM_META):
        import lightgbm
        self.booster = lightgbm.Booster(model_file=str(model_path))
        meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
        self.mean, self.scale = np.array(meta['mean']), np.array(meta['scale'])
        classes = meta.get('classes') or ['Gesture', 'NoGesture']   # the package's label encoder sorts names
        self.gesture_col = classes.index('Gesture')

    def features(self, seq):
        return lgbm_features(seq[-SEQ_LGBM:])[-1]

    def __call__(self, seq=None, features=None):
        """seq (>= 5, 92), or precomputed features (100,) -> P(Gesture)."""
        f = self.features(seq) if features is None else features
        p = np.asarray(self.booster.predict(((f - self.mean) / self.scale).reshape(1, -1)))
        p = p.reshape(1, -1)[0]
        return float(p[self.gesture_col]) if len(p) > 1 else float(p[0])


def check(n=200, seed=0):
    """Compare the port with the package on random and on recorded-like inputs; prints max absolute differences."""
    import joblib
    import tensorflow as tf  # noqa: F401  (only to run the package's own model)
    from envisionhgdetector.model_combined import build_cnn_model
    src = package_dir() / 'model'
    ref = build_cnn_model()
    ref.load_weights(str(src / 'cnn_default.h5'))
    cnn, lgbm = EnvisionCNN(), EnvisionLGBM()
    data = joblib.load(src / 'lightgbm_default.pkl')
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.3, (1, 92)).astype(np.float32)
    seqs = base + np.cumsum(rng.normal(0, 0.02, (n, SEQ_CNN, 92)), axis=1).astype(np.float32)
    seqs[:, :, 3::4] = rng.uniform(0, 1, (n, SEQ_CNN, 23))
    want = ref.predict(seqs, verbose=0)
    got = np.array([[(o := cnn(s))['motion'], o['gesture'], o['move']] for s in seqs])
    feats = np.array([lgbm.features(s) for s in seqs])
    model = data['model']
    p_ref = (model.predict_proba if hasattr(model, 'predict_proba') else model.predict)(data['scaler'].transform(feats))
    p_ref = np.asarray(p_ref).reshape(n, -1)[:, lgbm.gesture_col]
    p_got = np.array([lgbm(features=f) for f in feats])
    print(f'CNN max |diff| {np.abs(want - got).max():.2e} over {n} sequences (motion range '
          f'{want[:, 0].min():.2f}-{want[:, 0].max():.2f})')
    print(f'LightGBM max |diff| {np.abs(p_ref - p_got).max():.2e} (P(Gesture) range {p_ref.min():.2f}-{p_ref.max():.2f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['export', 'check'])
    args = ap.parse_args()
    export() if args.cmd == 'export' else check()


if __name__ == '__main__':
    main()

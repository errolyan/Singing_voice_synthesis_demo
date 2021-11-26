# -*- coding: utf-8 -*-

import os
from os.path import join, basename, exists
import joblib
import torch
import pysptk
import pyworld
import librosa
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import librosa.display
import soundfile as sf
# import IPython
# from IPython.display import Audio
from nnmnkwii.io import hts
from nnmnkwii import paramgen
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.frontend import merlin as fe
from nnsvs.multistream import multi_stream_mlpg, split_streams
from nnsvs.gen import (predict_timelag, predict_duration, predict_acoustic, postprocess_duration,gen_waveform, get_windows)
from nnsvs.frontend.ja import xml2lab, _lazy_init
from nnsvs.gen import _midi_to_hz

_lazy_init(dic_dir="/usr/lib/sinsy/dic")
sample_rate = 48000
frame_period = 5
fftlen = pyworld.get_cheaptrick_fft_size(sample_rate)
alpha = pysptk.util.mcepalpha(sample_rate)
hop_length = int(0.001 * frame_period * sample_rate)

## 建模
# model_dir = "./20200502_kiritan_singing-00-svs-world"
model_dir = "./model"

use_cuda = True
if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

### Time-lag model 时间模型
timelag_config = OmegaConf.load(join(model_dir, "timelag", "model.yaml"))
timelag_model = hydra.utils.instantiate(timelag_config.netG).to(device)
checkpoint = torch.load(join(model_dir, "timelag", "latest.pth"), map_location=lambda storage, loc: storage)
timelag_model.load_state_dict(checkpoint["state_dict"])
timelag_in_scaler = joblib.load(join(model_dir, "in_timelag_scaler.joblib"))
timelag_out_scaler = joblib.load(join(model_dir, "out_timelag_scaler.joblib"))
timelag_model.eval();

### Duration model  时长模型

duration_config = OmegaConf.load(join(model_dir, "duration", "model.yaml"))
duration_model = hydra.utils.instantiate(duration_config.netG).to(device)
checkpoint = torch.load(join(model_dir, "duration", "latest.pth"), map_location=lambda storage, loc: storage)
duration_model.load_state_dict(checkpoint["state_dict"])
duration_in_scaler = joblib.load(join(model_dir, "in_duration_scaler.joblib"))
duration_out_scaler = joblib.load(join(model_dir, "out_duration_scaler.joblib"))
duration_model.eval();

### Acoustic model 语音建模

acoustic_config = OmegaConf.load(join(model_dir, "acoustic", "model.yaml"))
acoustic_model = hydra.utils.instantiate(acoustic_config.netG).to(device)
checkpoint = torch.load(join(model_dir, "acoustic", "latest.pth"), map_location=lambda storage, loc: storage)
acoustic_model.load_state_dict(checkpoint["state_dict"])
acoustic_in_scaler = joblib.load(join(model_dir, "in_acoustic_scaler.joblib"))
acoustic_out_scaler = joblib.load(join(model_dir, "out_acoustic_scaler.joblib"))
acoustic_model.eval();

## 合成

### 选取xml文件

# NOTE: 01.xml and 02.xml were not included in the training data
# 03.xml - 37.xml were used for training.
labels = xml2lab("kiritan_singing/musicxml/01.xml").round_()

question_path = join(model_dir, "jp_qst001_nnsvs.hed")
binary_dict, continuous_dict = hts.load_question_set(question_path, append_hat_for_LL=False)

# pitch indices in the input features
pitch_idx = len(binary_dict) + 1
pitch_indices = np.arange(len(binary_dict), len(binary_dict)+3)
log_f0_conditioning = True


### Predict time-lag

lag = predict_timelag(device, labels, timelag_model, timelag_in_scaler,
    timelag_out_scaler, binary_dict, continuous_dict, pitch_indices,
    log_f0_conditioning)
print(lag.shape)

### Predict phoneme durations

durations = predict_duration(device, labels, duration_model,
    duration_in_scaler, duration_out_scaler, lag, binary_dict, continuous_dict,
    pitch_indices, log_f0_conditioning)
print(durations.shape)

# Normalize phoneme durations to satisfy constraints by the musical score
duration_modified_labels = postprocess_duration(labels, durations, lag)

### Predict acoustic features

acoustic_features = predict_acoustic(device, duration_modified_labels, acoustic_model,
    acoustic_in_scaler, acoustic_out_scaler, binary_dict, continuous_dict,
    "coarse_coding", pitch_indices, log_f0_conditioning)
print(acoustic_features.shape)

### Visualize acoustic features
'''
Before generating a wavefrom, let's visualize acoustic features to understand how the acoustic model works. Since acoustic features contain multiple differnt features (*multi-stream*, e.g., mgc, lf0, vuv and bap), let us first split acoustic features.
'''

stream_sizes = acoustic_config.stream_sizes
has_dynamic_features = acoustic_config.has_dynamic_features
# (mgc, lf0, vuv, bap) with delta and delta-delta except for vuv
stream_sizes, has_dynamic_features

feats = multi_stream_mlpg(acoustic_features, acoustic_out_scaler.var_, get_windows(3), stream_sizes, has_dynamic_features)
# get static features
mgc, diff_lf0, vuv, bap = split_streams(feats, [60, 1, 1, 5])

"""#### Visualize F0"""

# relative f0 -> absolute f0
# need to extract pitch sequence from the musical score
linguistic_features = fe.linguistic_features(duration_modified_labels,binary_dict, continuous_dict,add_frame_features=True,subphone_features="coarse_coding")
f0_score = _midi_to_hz(linguistic_features, pitch_idx, False)[:, None]
lf0_score = f0_score.copy()
nonzero_indices = np.nonzero(lf0_score)
lf0_score[nonzero_indices] = np.log(f0_score[nonzero_indices])
lf0_score = interp1d(lf0_score, kind="slinear")

f0 = diff_lf0 + lf0_score
f0[vuv < 0.5] = 0
f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

#### Visualize spectrogram

# Trim and visualize (to save memory and time)
logsp = np.log(pysptk.mc2sp(mgc[-2500:, :], alpha=alpha, fftlen=fftlen))
librosa.display.specshow(logsp.T, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="linear", cmap="jet");

#### Visualize aperiodicity

aperiodicity = pyworld.decode_aperiodicity(bap[-2500:, :].astype(np.float64), sample_rate, fftlen)
librosa.display.specshow(aperiodicity.T, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="linear", cmap="jet");

### Generate waveform
#Finally, let's generate waveform and listen to the sample.


generated_waveform = gen_waveform(duration_modified_labels, acoustic_features, acoustic_out_scaler,binary_dict, continuous_dict, acoustic_config.stream_sizes,acoustic_config.has_dynamic_features,"coarse_coding", log_f0_conditioning,pitch_idx, num_windows=3,post_filter=True, sample_rate=sample_rate, frame_period=frame_period,relative_f0=True)

# trim trailing/leading silences for covenience
generated_waveform = librosa.effects.trim(generated_waveform)[0]

# 保存
sf.write('./result/test.wav', generated_waveform, sample_rate)

"""## Listen to the generated sample"""
librosa.display.waveplot(generated_waveform, sample_rate, x_axis="time")
# IPython.display.display(Audio(generated_waveform, rate=sample_rate))

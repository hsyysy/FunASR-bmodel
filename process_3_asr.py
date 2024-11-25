#!/bin/python3
import numpy as np
import time
import os
import pickle
from funasr import AutoModel
from funasr.utils.vad_utils import slice_padding_audio_samples

file_path = "./audio/vad_example.wav"
#file_path = "./audio/20240711090630019.wav"
filename = os.path.splitext(file_path)[0]

dev_id = 5
model_dir = "./bmodel/"
model_asr = AutoModel(model=model_dir+"speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", device="cpu", disable_update=True, disable_pbar=True, dev_id=dev_id)
#model_asr = AutoModel(model=model_dir+"speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",device="cpu", disable_update=True, disable_pbar=True, dev_id=dev_id)

with open(filename+'_speechs.pkl', 'rb') as f:
    speechs = pickle.load(f)

with open(filename+"_vad.pkl", "rb") as f:
    sorted_data = pickle.load(f)


st = time.time()

results_sorted = []
n = len(sorted_data)

if not len(sorted_data):
    print("empty speech")
    exit()

batch_size = 300000
batch_size_threshold_ms = 60000
if len(sorted_data) > 0 and len(sorted_data[0]) > 0:
    batch_size = max(batch_size, sorted_data[0][0][1] - sorted_data[0][0][0])

beg_idx = 0

st = time.time()
for j in range(n):
    batch_size_ms_cum = (sorted_data[j][0][1] - sorted_data[j][0][0])
    if j < n - 1 and (
        batch_size_ms_cum + sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size and (
        sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size_threshold_ms and (
        j + 1 - beg_idx < 10):  # 10 is upper limit of asr_bmodel's batch:
        continue
    end_idx = j + 1
    speech_j, speech_lengths_j = slice_padding_audio_samples(speechs[0], len(speechs[0]), sorted_data[beg_idx:end_idx])
    results = model_asr.inference(speech_j,batch_size=batch_size)
    beg_idx = end_idx
    if len(results) < 1:
        continue
    results_sorted.extend(results)
print("asr total cost  =",time.time() - st)

with open(filename+'_asr.pkl', 'wb') as f:
    pickle.dump(results_sorted, f)

#!/bin/python3
import numpy as np
import time
import os
import pickle

from funasr import AutoModel
from funasr.utils.vad_utils import slice_padding_audio_samples
from funasr.models.campplus.utils import sv_chunk

from process_0_info import get_file_dev_id

file_path, dev_id = get_file_dev_id()

filename = os.path.splitext(file_path)[0]

dev_id = 5
model_dir = "./bmodel/"
model_spk = AutoModel(model=model_dir+"speech_campplus_sv_zh-cn_16k-common", device="cpu", disable_update=True, disable_pbar=True, dev_id=dev_id)

with open(filename+'_speechs.pkl', 'rb') as f:
    speechs = pickle.load(f)

with open(filename+"_vad.pkl", "rb") as f:
    sorted_data = pickle.load(f)

with open(filename+'_asr.pkl', 'rb') as f:
    results_sorted = pickle.load(f)

st = time.time()

segments = []
n = len(sorted_data)
for ii in range(n):
    speech_j, speech_lengths_j = slice_padding_audio_samples(speechs[0], len(speechs[0]), [sorted_data[ii]])
    segment = sv_chunk([[sorted_data[ii][0][0]/1000.0,sorted_data[ii][0][1]/1000.0,speech_j[0]]])
    segments.extend(segment)
    spk_res_i = model_spk.inference([seg[2] for seg in segment])
    spk_output = [spk_res_ij["spk_embedding"].numpy() for spk_res_ij in spk_res_i]
    results_sorted[ii]['spk_embedding'] = np.stack([t.squeeze(0) for t in spk_output], axis=0)

restored_data = [0] * n
vadsegments = [0] * n
for j in range(n):
    index = sorted_data[j][1]
    restored_data[index] = results_sorted[j]
    vadsegments[sorted_data[j][1]] = sorted_data[j][0]
result = {}

for j in range(n):
    for k, v in restored_data[j].items():
        if k.startswith("timestamp"):
            if k not in result:
                result[k] = []
            for t in restored_data[j][k]:
                t[0] += vadsegments[j][0]
                t[1] += vadsegments[j][0]
            result[k].extend(restored_data[j][k])
        elif k == 'spk_embedding':
            if k not in result:
                result[k] = restored_data[j][k]
            else:
                result[k] = np.concatenate([result[k], restored_data[j][k]], axis=0)
        elif 'text' in k:
            if k not in result:
                result[k] = restored_data[j][k]
            else:
                result[k] += " " + restored_data[j][k]
        else:
            if k not in result:
                result[k] = restored_data[j][k]
            else:
                result[k] += restored_data[j][k]

print("spk total cost  =",time.time() - st)
with open(filename+'_spk.pkl', 'wb') as f:
    pickle.dump(result, f)

with open(filename+'_spk_segments.pkl', 'wb') as f:
    pickle.dump(segments, f)

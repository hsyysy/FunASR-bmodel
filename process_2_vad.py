#!/bin/python3
import numpy as np
import time
import pickle
import os
from funasr import AutoModel

from process_0_info import get_file_dev_id

file_path, dev_id = get_file_dev_id()

filename = os.path.splitext(file_path)[0]
dev_id = 5
model_dir = "./bmodel/"
model_vad = AutoModel(model=model_dir+"speech_fsmn_vad_zh-cn-16k-common-pytorch", device="cpu", disable_update=True, disable_pbar=True, dev_id=dev_id)

with open(filename+'_speechs.pkl', 'rb') as f:
    speechs = pickle.load(f)

st = time.time()
res = model_vad.inference(speechs, disable_pbar=True)[0]

vadsegments = res["value"]
n = len(vadsegments)
data_with_index = [(vadsegments[i], i) for i in range(n)]
sorted_data = sorted(data_with_index, key=lambda x: x[0][1] - x[0][0])
print(sorted_data)

print("vad total cost  =",time.time() - st)

with open(filename+"_vad.pkl", "wb") as f:
    pickle.dump(sorted_data, f)

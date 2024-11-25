#!/bin/python3
import numpy as np
import time
import pickle
import os

from funasr import AutoModel

from process_0_info import get_file_dev_id

file_path, dev_id, model_dir = get_file_dev_id()

filename = os.path.splitext(file_path)[0]

model_punc = AutoModel(model=model_dir+"punc_ct-transformer_zh-cn-common-vocab272727-pytorch", device="cpu", disable_update=True, disable_pbar=True, dev_id=dev_id)

with open(filename+'_spk.pkl', 'rb') as f:
    result = pickle.load(f)

st = time.time()
result["raw_text"] = result["text"]
punc_res = model_punc.inference(result["text"])
result["text"] = punc_res[0]["text"]
result["punc_array"] = punc_res[0]["punc_array"]
print("punc total cost =",time.time() - st)

with open(filename+'_punc.pkl', 'wb') as f:
    pickle.dump(result, f)

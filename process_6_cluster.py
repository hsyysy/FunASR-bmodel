#!/bin/python3
import numpy as np
import time
import os
import pickle

from funasr.utils.timestamp_tools import timestamp_sentence
from funasr.models.campplus.cluster_backend import ClusterBackend
from funasr.models.campplus.utils import postprocess, distribute_spk

from process_0_info import get_file_dev_id

file_path, _, _ = get_file_dev_id()

filename = os.path.splitext(file_path)[0]

with open(filename+'_punc.pkl', 'rb') as f:
    result = pickle.load(f)

with open(filename+'_spk_segments.pkl', 'rb') as f:
    segments = pickle.load(f)

st = time.time()
segments = sorted(segments, key=lambda x: x[0])
spk_embedding = result['spk_embedding']
cb_model = ClusterBackend()
labels = cb_model(spk_embedding)
sv_output = postprocess(segments, None, labels, spk_embedding)
sentence_list = timestamp_sentence(result['punc_array'], result['timestamp'], result["raw_text"])
distribute_spk(sentence_list, sv_output)
result['sentence_info'] = sentence_list
if "spk_embedding" in result:
    del result['spk_embedding']
print("clustering cost =",time.time() - st)

with open(filename+'_cluster.pkl', 'wb') as f:
    pickle.dump(result, f)

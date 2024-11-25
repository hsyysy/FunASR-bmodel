#!/bin/python3
import pickle
import os

from process_0_info import get_file_dev_id

file_path, _, _ = get_file_dev_id()

filename = os.path.splitext(file_path)[0]

with open(filename+'_cluster.pkl', 'rb') as f:
    result = pickle.load(f)

# show result
for si in result["sentence_info"]:
    print("["+format(si["start"],"7d")+"]["+format(si["end"],"7d")+"], spk="+format(si["spk"],'2d')+", text: "+si["text"])

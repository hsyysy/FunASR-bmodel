#!/bin/python3
import pickle
import os

file_path = "./audio/vad_example.wav"
#file_path = "./audio/20240711090630019.wav"
filename = os.path.splitext(file_path)[0]

with open(filename+'_cluster.pkl', 'rb') as f:
    result = pickle.load(f)

# show result
for si in result["sentence_info"]:
    print("["+format(si["start"],"7d")+"]["+format(si["end"],"7d")+"], spk="+format(si["spk"],'2d')+", text: "+si["text"])

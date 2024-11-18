import numpy as np

with open("log.txt","r") as f:
    lines = f.readlines()

pstr = [
        "vad prepare time",
        "vad fbank time",
        "vad forward ComputeDecibel time",
        "vad forward ComputeScores time",
        "vad forward DetectFrames time",
        "vad forward post time",
        "vad forward time",
        "vad post time",
        "vad total time",
        "data prepare time",
        "speech prepare time",
        "asr loading file time",
        "asr init beamsearch time",
        "asr extract fbank feats time",
        "asr encode time",
        "asr predict time",
        "asr inner post time",
        "asr inference time",
        "spk extract fbank feats time",
        "spk forward time",
        "spk embedding time",
        "asr and spk post time",
        "punc time",
        "clustering import time",
        "clustering umap time",
        "clustering HDBSCAN time",
        "clustering cb_model time",
        "clustering post time",
        "clustering total time",
        ]
strlen = max([len(i) for i in pstr])
ptime = {}
pnum = {}
for pstr0 in pstr:
    ptime[pstr0] = 0
    pnum[pstr0] = 0
for a in lines:
    for pstr0 in pstr:
        if a.startswith(pstr0):
            ptime[pstr0] += float(a.split(' ')[-1])
            pnum[pstr0] += 1

for pstr0 in pstr:
    print(pstr0.ljust(strlen+1),':',format(ptime[pstr0],'6.2f')+", N =",format(pnum[pstr0],'4d'))

import numpy as np

def safe_str_to_float(s):
    try:
        return float(s)
    except ValueError:
        return None

with open("log.txt","r") as f:
    lines = f.readlines()

pstr = [
        "vad inner prepare time",
        "vad inner fbank time",
        "vad inner forward ComputeDecibel time",
        "vad inner forward ComputeScores time",
        "vad inner forward DetectFrames time",
        "vad inner forward total time",
        "vad total time",
        "asr load files time",
        "asr inner extract fbank feats time",
        "asr inner encode time",
        "asr inner predict time",
        "asr inner post time",
        "asr inference time",
        "spk extract fbank feats time",
        "spk forward time",
        "spk embedding time",
        "asr and spk post time",
        "punc time",
        "clustering inner import time",
        "clustering inner umap time",
        "clustering inner HDBSCAN time",
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
for idx,a in enumerate(lines):
    for pstr0 in pstr:
        if a.startswith(pstr0):
            num = safe_str_to_float(a.split(' ')[-1])
            if num:
                ptime[pstr0] += num
            else:
                print(format(idx,'5d')+': '+a,end="")
            pnum[pstr0] += 1

for pstr0 in pstr:
    print(pstr0.ljust(strlen+1),':',format(ptime[pstr0],'6.2f')+", N =",format(pnum[pstr0],'4d'))

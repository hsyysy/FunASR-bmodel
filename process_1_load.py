#!/bin/python3
import numpy as np
import time
import pickle
import os
import wave

from process_0_info import get_file_dev_id

file_path, _, _= get_file_dev_id()
filename = os.path.splitext(file_path)[0]

st = time.time()
with wave.open(file_path, "rb") as wav_file:
    params = wav_file.getparams()
    fs = wav_file.getframerate()
    frames = wav_file.readframes(wav_file.getnframes())
    audio_bytes = bytes(frames)
speechs = [ np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0 ]

print("load files cost =",time.time() - st)

with open(filename+'_speechs.pkl', 'wb') as f:
    pickle.dump(speechs, f)
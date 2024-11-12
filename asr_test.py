import numpy as np
import torch
import time

# load model
print("importing AutoModel...")
from funasr import AutoModel
print("importing finished")
dev_id = 5
print("loading model")
print("-"*120)
model_vad = AutoModel(model="bmodel/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", device="cpu", disable_update=True, disable_pbar=True, dev_id=dev_id)
print("-"*120)
model_asr = AutoModel(model="bmodel/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", device="cpu", disable_update=True, disable_pbar=True, dev_id=dev_id)
print("-"*120)
model_spk = AutoModel(model="bmodel/iic/speech_campplus_sv_zh-cn_16k-common", device="cpu", disable_update=True, disable_pbar=True, dev_id=dev_id)
print("-"*120)
model_punc = AutoModel(model="bmodel/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", device="cpu", disable_update=True, disable_pbar=True, dev_id=dev_id)
print("-"*120)
print("model loaded")
print()
all_st = time.time()

# step.0 load audio
st = time.time()
import wave
with wave.open("./audio/vad_example.wav", "rb") as wav_file:
#with wave.open("./audio/20240711090630019.wav", "rb") as wav_file:
    params = wav_file.getparams()
    fs = wav_file.getframerate()
    frames = wav_file.readframes(wav_file.getnframes())
    audio_bytes = bytes(frames)

speechs = [ np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0 ]
print("load files cost =",time.time() - st)

# step.1 vad
st = time.time()
res = model_vad.inference(speechs, disable_pbar=True)[0]
print("vad total cost  =",time.time() - st)

# step.2 asr and spk
from funasr.utils.vad_utils import slice_padding_audio_samples
from funasr.models.campplus.utils import sv_chunk

st = time.time()
key = res["key"]
vadsegments = res["value"]
speech = speechs[0]
speech_lengths = len(speech)
n = len(vadsegments)
data_with_index = [(vadsegments[i], i) for i in range(n)]
sorted_data = sorted(data_with_index, key=lambda x: x[0][1] - x[0][0])
results_sorted = []

if not len(sorted_data):
    logging.info("decoding, utt: {}, empty speech".format(key))
    exit()

batch_size = 300000
if len(sorted_data) > 0 and len(sorted_data[0]) > 0:
    batch_size = max(batch_size, sorted_data[0][0][1] - sorted_data[0][0][0])

batch_size_threshold_ms = 60000
beg_idx = 0

all_segments = []
spk_time = 0
for j in range(n):
    batch_size_ms_cum = (sorted_data[j][0][1] - sorted_data[j][0][0])
    if j < n - 1 and (
        batch_size_ms_cum + sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size and (
        sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size_threshold_ms and (
        j + 1 - beg_idx < 10):  # 10 is upper limit of asr_bmodel's batch:
        continue
    end_idx = j + 1
    speech_j, speech_lengths_j = slice_padding_audio_samples(speech, speech_lengths, sorted_data[beg_idx:end_idx])
    results = model_asr.inference(speech_j,batch_size=batch_size)
    spk_st = time.time()
    for _b in range(len(speech_j)):
        vad_segments = [[sorted_data[beg_idx:end_idx][_b][0][0]/1000.0,
                         sorted_data[beg_idx:end_idx][_b][0][1]/1000.0,
                         np.array(speech_j[_b])]]
        segments = sv_chunk(vad_segments)
        all_segments.extend(segments)
        spk_res_i = model_spk.inference([seg[2] for seg in segments])
        spk_output = [spk_res_ij["spk_embedding"] for spk_res_ij in spk_res_i]
        results[_b]['spk_embedding'] = torch.stack([t.squeeze(0) for t in spk_output], dim=0)
    beg_idx = end_idx
    if len(results) < 1:
        continue
    results_sorted.extend(results)
    spk_time += time.time()-spk_st
print("spk total cost  =",spk_time)

restored_data = [0] * n
for j in range(n):
    index = sorted_data[j][1]
    restored_data[index] = results_sorted[j]
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

print("asr total cost  =",time.time() - st - spk_time)

# step.3 compute punc model

st = time.time()
raw_text = result["text"]
punc_res = model_punc.inference(result["text"])
result["text"] = punc_res[0]["text"]
print("punc total cost =",time.time() - st)

# step.4 speaker embedding cluster after resorted
from funasr.utils.timestamp_tools import timestamp_sentence
from funasr.models.campplus.cluster_backend import ClusterBackend
from funasr.models.campplus.utils import postprocess, distribute_spk

st = time.time()
all_segments = sorted(all_segments, key=lambda x: x[0])
spk_embedding = result['spk_embedding']
cb_model = ClusterBackend()
labels = cb_model(spk_embedding)
sv_output = postprocess(all_segments, None, labels, spk_embedding)
sentence_list = timestamp_sentence(punc_res[0]['punc_array'], result['timestamp'], raw_text)
distribute_spk(sentence_list, sv_output)
result['sentence_info'] = sentence_list
if "spk_embedding" in result:
    del result['spk_embedding']
result["key"] = key
print("clustering cost =",time.time() - st)
print("generation cost =",time.time() - all_st)

# show result
print("-"*120)
for si in result["sentence_info"]:
    print("["+format(si["start"],"7d")+"]["+format(si["end"],"7d")+"], spk="+format(si["spk"],'2d')+", text: "+si["text"])

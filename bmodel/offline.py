#!/bin/python3

import sys
sys.path.append("..")

from funasr import AutoModel
import time
import multiprocessing 

file_path = "../audio/test_audio_20241017.wav"
dev_id = 0
target = "BM1684X"

# offline asr demo
model = AutoModel(
        #model="speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/models/"+target, ## 语音识别模型
        model="speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404/models/"+target,       ## 语音识别模型
        vad_model="speech_fsmn_vad_zh-cn-16k-common/models/"+target,                                   ## 语音端点检测模型
        punc_model="punc_ct-transformer_zh-cn-common-vocab272727/models/"+target,                      ## 标点恢复模型
        spk_model="speech_campplus_sv_zh-cn_16k-common/models/"+target,                                ## 说话人识别模型
        device="cpu",
        disable_update=True,
        disable_pbar=True,
        dev_id=dev_id,
)

# inference
start_time = time.time()
res = model.generate(input=file_path, batch_size_s=300,)
end_time = time.time()
if "sentence_info" in res[0].keys():
    for si in res[0]["sentence_info"]:
        print("["+format(si["start"],"7d")+"]["+format(si["end"],"7d")+"], spk="+format(si["spk"],'2d')+", text: "+si["text"])
else:
    print(res[0]["text"])
print("generate time:", end_time-start_time)

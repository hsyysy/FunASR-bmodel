#!/bin/python3
from funasr import AutoModel

modelnames = [
        #"iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        #"iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
        #"iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        #"iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        #"iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        #"iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
        #"iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
]

for modelname in modelnames:
    model = AutoModel(model=modelname, device="cpu", disable_update=True)
    res = model.export(quantize=True)

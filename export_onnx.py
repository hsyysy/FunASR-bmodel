#!/bin/python3
from funasr import AutoModel
import os

modelname = "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
model = AutoModel(model=modelname, device="cpu", disable_update=True, output_dir=output_dir)
res = model.export(quantize=False, opset_version=13)

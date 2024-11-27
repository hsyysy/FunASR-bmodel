## model from export.py
model_dir=$HOME/funasr_onnx_model

model=./bmodel/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404

online_model=${model_dir}/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online

vad_dir=./bmodel/speech_fsmn_vad_zh-cn-16k-common-pytorch

punc_dir=./bmodel/punc_ct-transformer_zh-cn-common-vocab272727-pytorch

quantize=false

./runtime/onnxruntime/build/bin/funasr-onnx-2pass \
    --mode 2pass \
    --model-dir         ${model}    \
    --online-model-dir  ${online_model} \
    --vad-dir           ${vad_dir}  \
    --punc-dir          ${punc_dir} \
    --quantize          ${quantize} \
    --vad-quant         ${quantize} \
    --punc-quant        ${quantize} \
    --lm-dir  $HOME/modelscope/speech_ngram_lm_zh-cn-ai-wesp-fst \
    --itn-dir $HOME/modelscope/fst_itn_zh \
    --wav-path ./audio/test_audio_20241017.wav

    #--wav-path ./audio/test_audio_20241017.wav

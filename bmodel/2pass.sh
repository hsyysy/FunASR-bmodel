hub=$HOME/.cache/modelscope/hub
target=BM1684X

# bmodel
model=speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/models/$target
vad_dir=speech_fsmn_vad_zh-cn-16k-common/models/$target
punc_dir=punc_ct-transformer_zh-cn-common-vocab272727/models/$target

# onnx model
online_model=${hub}/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online

# other model
lm_dir=${hub}/iic/speech_ngram_lm_zh-cn-ai-wesp-fst
itn_dir=${hub}/thuduj12/fst_itn_zh

quantize=false

../runtime/onnxruntime/build/bin/funasr-onnx-2pass \
    --mode offline \
    --model-dir         ${model}    \
    --online-model-dir  ${online_model} \
    --vad-dir           ${vad_dir}  \
    --punc-dir          ${punc_dir} \
    --quantize          ${quantize} \
    --vad-quant         ${quantize} \
    --punc-quant        ${quantize} \
    --lm-dir            ${lm_dir} \
    --itn-dir           ${itn_dir} \
    --wav-path ../audio/test_audio_20241017.wav

    #--wav-path ../test_asr.wav
    #--wav-path ../audio/test_audio_20241017.wav
    #--wav-path ../audio/vad_example.wav
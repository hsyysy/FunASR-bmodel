runtime=./bmodel
target=BM1684X

# bmodel
model=${runtime}/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/models/$target
vad_dir=${runtime}/speech_fsmn_vad_zh-cn-16k-common/models/$target
punc_dir=${runtime}/punc_ct-transformer_zh-cn-common-vocab272727/models/$target

# other model
lm_dir=$HOME/.cache/modelscope/hub/iic/speech_ngram_lm_zh-cn-ai-wesp-fst
itn_dir=$HOME/.cache/modelscope/hub/thuduj12/fst_itn_zh

#quantize=true
quantize=false

./runtime/onnxruntime/build/bin/funasr-onnx-offline \
    --model-dir  ${model}    \
    --vad-dir    ${vad_dir}  \
    --punc-dir   ${punc_dir} \
    --quantize   ${quantize} \
    --vad-quant  ${quantize} \
    --punc-quant ${quantize} \
    --lm-dir     ${lm_dir} \
    --itn-dir    ${itn_dir} \
    --wav-path ./audio/test_audio_20241017.wav

    #--wav-path ./test_asr.wav
    #--wav-path ./audio/test_audio_20241017.wav
    #--wav-path ./audio/vad_example.wav

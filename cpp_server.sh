model_dir=./bmodel
target=BM1684X

# bmodel
model=${model_dir}/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/models/$target
vad_dir=${model_dir}/speech_fsmn_vad_zh-cn-16k-common/models/$target
punc_dir=${model_dir}/punc_ct-transformer_zh-cn-common-vocab272727/models/$target

# onnx model
online_model=$HOME/funasr_onnx_model/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online

# other model
lm_dir=$HOME/.cache/modelscope/hub/iic/speech_ngram_lm_zh-cn-ai-wesp-fst
itn_dir=$HOME/.cache/modelscope/hub/thuduj12/fst_itn_zh

#quantize=true
quantize=false

./runtime/websocket/build/bin/funasr-wss-server-2pass \
    --model-dir         ${model}    \
    --online-model-dir  ${online_model} \
    --vad-dir           ${vad_dir}  \
    --punc-dir          ${punc_dir} \
    --quantize          ${quantize} \
    --vad-quant         ${quantize} \
    --punc-quant        ${quantize} \
    --lm-dir            ${lm_dir} \
    --itn-dir           ${itn_dir} \
    --keyfile  ./runtime/ssl_key/server.key \
    --certfile ./runtime/ssl_key/server.crt \
    --hotword "" \
    --port 10211

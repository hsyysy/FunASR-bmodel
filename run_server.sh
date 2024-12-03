export PYTHONPATH=$PWD:$PYTHONPATH
model_dir=../../../bmodel
target=BM1684X

asr_model="${model_dir}/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/models/${target}"
asr_model_online="${model_dir}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/models/${target}"
vad_model="${model_dir}/speech_fsmn_vad_zh-cn-16k-common/models/${target}"
punc_model="${model_dir}/punc_ct-transformer_zh-cn-common-vocab272727/models/${target}"


cd runtime/python/websocket
#python3 funasr_wss_server.py \
python3 funasr_wss_server_with_id.py \
    --asr_model $asr_model \
    --asr_model_online $asr_model_online \
    --vad_model $vad_model \
    --punc_model $punc_model \
    --device cpu \
    --dev_id 5 \
    --port 12333

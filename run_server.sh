export PYTHONPATH=$PWD:$PYTHONPATH

modeldir="../../../bmodel/iic"

asr_model="${modeldir}/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
#asr_model="bmodel/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

asr_model_online="${modeldir}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
vad_model="${modeldir}/speech_fsmn_vad_zh-cn-16k-common-pytorch"
punc_model="${modeldir}/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

cd runtime/python/websocket
#python3 funasr_wss_server.py \
python3 funasr_wss_server_with_id.py \
    --asr_model $asr_model \
    --asr_model_online $asr_model_online \
    --vad_model $vad_model \
    --punc_model $punc_model \
    --device cpu \
    --dev_id 0 \
    --port 12333

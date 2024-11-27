model_dir="$HOME/funasr_onnx_model"

model=./bmodel/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404

vad_dir=./bmodel/speech_fsmn_vad_zh-cn-16k-common-pytorch

punc_dir=./bmodel/punc_ct-transformer_zh-cn-common-vocab272727-pytorch

#quantize=true
quantize=false

./runtime/onnxruntime/build/bin/funasr-onnx-offline \
    --model-dir  ${model}    \
    --vad-dir    ${vad_dir}  \
    --punc-dir   ${punc_dir} \
    --quantize   ${quantize} \
    --vad-quant  ${quantize} \
    --punc-quant ${quantize} \
    --bladedisc false \
    --lm-dir  $HOME/modelscope/speech_ngram_lm_zh-cn-ai-wesp-fst \
    --itn-dir $HOME/modelscope/fst_itn_zh \
    --wav-path ./audio/vad_example.wav

    #--wav-path ./test_asr.wav
    #--wav-path ./audio/test_audio_20241017.wav
    #--wav-path ./audio/vad_example.wav

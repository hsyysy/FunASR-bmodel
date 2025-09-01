cd ../runtime/python/websocket
#python3 funasr_wss_client.py \
python3 funasr_wss_client_with_id.py \
    --mode 2pass \
    --audio_in ../../../audio/test_audio_20241017.wav \
    --port 12333 \
    --hotword ../../../bmodel/hotwords.txt \
    --chunk_interval 10

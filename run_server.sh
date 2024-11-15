export $PYTHONPATH=$PWD:$PYTHONPATH
cd runtime/python/websocket
python3 funasr_wss_server_with_id.py --device cpu --dev_id 0 --port 12333

请切换仓库到`export_contextual`分支，运行`python3 export_onnx.py`即可于目录`output`中获取`encoder.onnx`。
然后将文件`funasr/models/contextual_paraformer/export_meta.py`中的`part = "encoder"`修改为`part = "decoder"`后，
再度执行`python3 export_onnx.py`即可于目录`output`中获取`decoder.onnx`。

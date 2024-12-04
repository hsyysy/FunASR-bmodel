## 模型编译

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，具体可参考[模型导出](./docs/Export_Guide.md)。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR编译环境，具体可参考[TPU-MLIR环境搭建](https://github.com/sophgo/sophon-demo/blob/release/docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》( 请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

## 下载mlir
从sftp上获取TPU-MLIR压缩包
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/FunASR/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/tpu-mlir_v1.13.beta.0-20241203.tar.gz
tar -xf tpu-mlir_v1.13.beta.0-20241203.tar.gz
```

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp32bmodel_encoder.sh bm1684x #bm1684x/bm1688
./scripts/gen_fp32bmodel_decoder.sh bm1684x #bm1684x/bm1688
./scripts/gen_fp32bmodel_lstm.sh bm1684x #bm1684x/bm1688
./scripts/gen_fp32bmodel_embedding.sh bm1684x #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684X`等文件夹下生成转换好的FP32 BModel文件。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/gen_fp16bmodel_encoder.sh bm1688 #bm1684x/bm1688
./scripts/gen_fp16bmodel_decoder.sh bm1688 #bm1684x/bm1688
```

​执行上述命令会在`models/BM1688/`等文件夹下生成转换好的FP32 BModel文件。

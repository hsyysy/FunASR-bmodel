# FunASR-bmodel

## 目录

- [FunASR-bmodel](#funasr)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型和链接库](#3-准备模型和链接库)
    - [3.1 使用提供的模型](#31-使用提供的模型)
    - [3.2 编译模型](#32-编译模型)
  - [4.运行测试](#4-运行测试)
    - [4.1 环境准备](#41-环境准备)
    - [4.2 离线识别](#42-离线识别)
    - [4.3 在线识别](#43-在线识别)

## 1. 简介

FunASR是一个流行的语音识别框架，本目录将常用的几个模型移植到算丰平台。

该例程支持在V23.09LTS SP3及以上的BM1684X SOPHONSDK, 或在v1.7.0及以上的BM1688 SOPHONSDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行，也支持在BM1688 Soc设备（如SE9-16）上运行。

## 2.特性

* 支持BM1684X(x86 PCIe、SoC)，BM1688(SoC)
* 支持FP32(BM1684X/BM1688)模型编译和推理

## 3. 准备模型

该模型目前支持在BM1684X和BM1688上运行，已提供编译好的bmodel。

### 3.1 使用提供的模型

​本例程在各个模型的目录内`scripts`目录下提供了下载脚本`download.sh`

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt-get update
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

支持的模型
|  模型名称  |  Python  | C++ |
|------------|----------|-----|
|[speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404](speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404)|支持|支持|
|[speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online](speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online)|支持|不支持|
|[speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404](speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404)|支持|不支持|
|[speech_fsmn_vad_zh-cn-16k-common](speech_fsmn_vad_zh-cn-16k-common)|支持|支持|
|[punc_ct-transformer_zh-cn-common-vocab272727](punc_ct-transformer_zh-cn-common-vocab272727)|支持|支持|
|[speech_campplus_sv_zh-cn_16k-common](speech_campplus_sv_zh-cn_16k-common)|支持|不支持|

## 3.2 编译模型
请参考模型目录内的readme.md编译

## 4. 运行测试

### 4.1 环境准备
Python需要先安装tpu_perf, 在x86平台使用算能计算卡（PCIE模式）时执行
```bash
pip3 install tpu_perf-1.2.35-py3-none-manylinux2014_x86_64.whl
```
在算能边缘计算盒子（SOC模式）上执行
```bash
pip3 install tpu_perf-1.2.35-py3-none-manylinux2014_aarch64.whl
```
另外还需要安装外部依赖库
```bash
pip3 install -r requirements.txt
```
C++编译所需的外部库请参考对应readme.md文件。
C++程序还需要[speech_ngram_lm_zh-cn-ai-wesp-fst](https://www.modelscope.cn/models/iic/speech_ngram_lm_zh-cn-ai-wesp-fst)和[fst_itn_zh](https://www.modelscope.cn/models/thuduj12/fst_itn_zh)，可在原[FunASR](https://github.com/modelscope/FunASR)仓库的推理过程中自动下载，或者从链接中下载，并在脚本中配置目录。

### 4.2 离线识别
Python离线识别可参考offline.py，配置好音频文件路径，以及平台和设备ID后，即可执行推理。

C++编译可参考[runtime/onnxruntime/readme.md](../runtime/onnxruntime/readme.md)编译，并用offline.sh测试运行。

**注意：** 编译前需要在[com-define.h](../runtime/onnxruntime/include/com-define.h)文件中配置DEV_ID，即设备ID。

### 4.3 在线识别
Python在线识别可参考run_server.sh开启服务，通过run_client.sh测试

C++编译可参考[runtime/websocket/readme.md](../runtime/websocket/readme_zh.md)编译，执行cpp_server.sh开启服务，通过cpp_client.sh测试，由于目前online模型暂未适配C++，因此online识别部分由cpu通过onnx推理完成。(在原[FunASR](https://github.com/modelscope/FunASR)仓库运行bmodel/export_onnx.py可转出onnx模型)

另有2pass.sh脚本用于本地测试在线识别服务。

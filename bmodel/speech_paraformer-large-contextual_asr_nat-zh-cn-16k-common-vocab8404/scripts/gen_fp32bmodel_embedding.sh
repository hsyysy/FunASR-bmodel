#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

name=embedding

outdir=../models/$target_dir

function gen_mlir()
{
    model_transform.py \
        --model_name $name \
        --model_def ../models/onnx/${name}.onnx \
        --input_shapes [[1,1]] \
        --mlir ${name}.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir ${name}.mlir \
        --quantize F32 \
        --chip $target \
        --model ${name}_fp32.bmodel
    mv ${name}_fp32.bmodel $outdir
}

pushd $model_dir
if [ ! -d "$outdir" ]; then
    echo $pwd
    mkdir $outdir
fi

gen_mlir
gen_fp32bmodel

popd

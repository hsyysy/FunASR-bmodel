#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

name=encoder
outdir=../models/$target_dir

function gen_mlir()
{
    model_transform.py \
        --model_name ${name} \
        --model_def ../models/onnx/${name}.onnx \
        --input_shapes [[1,20,560],[1],[3],[1,10,560]] \
        --test_input input_encoder.npz \
        --test_result ${name}_top_results.npz \
        --dynamic \
        --mlir ${name}.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir ${name}.mlir \
        --quantize F32 \
        --chip $target \
        --dynamic \
        --disable_layer_group \
        --model ${name}_fp32.bmodel \
        --test_input ${name}_in_f32.npz \
        --test_reference ${name}_top_results.npz \
        --tolerance 0.99,0.99

    mv ${name}_fp32.bmodel $outdir/
}

pushd $model_dir

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
gen_mlir
gen_fp32bmodel

popd

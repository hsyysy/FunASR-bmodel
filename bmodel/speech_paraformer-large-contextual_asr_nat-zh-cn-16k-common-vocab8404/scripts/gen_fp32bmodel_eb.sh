#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

name=eb

outdir=../models/$target_dir

function gen_mlir()
{
    model_transform.py \
        --model_name $name \
        --model_def ../models/onnx/model_${name}.onnx \
        --input_shapes [[100,10]] \
        --dynamic \
        --test_input ${name}_input_1b.npz \
        --test_result ${name}_output.npz \
        --mlir ${name}.mlir

        #--output_names /Transpose_output_0,/bias_encoder/ConstantOfShape_output_0 \
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir ${name}.mlir \
        --quantize F32 \
        --dynamic \
        --disable_layer_group \
	--compare_all \
        --chip $target \
        --test_input ${name}_in_f32.npz \
        --test_reference ${name}_output.npz \
        --tolerance 0.99,0.99 \
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

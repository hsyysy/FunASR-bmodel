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
        --input_shapes [[$1,2000,560],[$1]] \
        --test_input ${name}_input_$1b.npz \
        --test_result ${name}_top_results.npz \
        --dynamic \
        --mlir ${name}_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir ${name}_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --dynamic \
        --disable_layer_group \
        --model ${name}_fp16_$1b.bmodel \
        #--test_input ${name}_in_f32.npz \
        #--test_reference ${name}_top_results.npz \
        #--tolerance 0.99,0.99 \

    mv ${name}_fp16_$1b.bmodel $outdir/
}

pushd $model_dir

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp16bmodel 1

# batch_size=10
gen_mlir 10
gen_fp16bmodel 10

popd

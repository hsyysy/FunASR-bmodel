#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
fi

name=decoder
outdir=../models/$target_dir

function gen_mlir()
{
    model_transform.py \
        --model_name ${name} \
        --model_def ../models/onnx/${name}.onnx \
        --input_shapes [[$1,2000,512],[$1],[$1,600,512],[$1],[$1,100,512]] \
        --test_input ${name}_input_$1b.npz \
        --test_result ${name}_top_results.npz \
        --dynamic \
        --shape_influencing_input_names enc_len,pre_token_length \
        --mlir ${name}_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir ${name}_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --dynamic \
        --disable_layer_group \
        --model ${name}_fp32_$1b.bmodel \
        #--test_input ${name}_in_f32.npz \
        #--test_reference ${name}_top_results.npz \
        #--tolerance 0.99,0.99 \

    mv ${name}_fp32_$1b.bmodel $outdir/
}

pushd $model_dir

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

# batch_size=10
gen_mlir 10
gen_fp32bmodel 10

popd
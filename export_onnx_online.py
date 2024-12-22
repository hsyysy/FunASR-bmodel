from funasr import AutoModel
import torch
import os

def init_cache(cache: dict = {}, export = 1, **kwargs):
    chunk_size = kwargs.get("chunk_size", [5, 10, 5])
    encoder_chunk_look_back = kwargs.get("encoder_chunk_look_back", 4)
    decoder_chunk_look_back = kwargs.get("decoder_chunk_look_back", 1)
    batch_size = 1

    enc_output_size = kwargs["encoder_conf"]["output_size"]
    feats_dims = kwargs["frontend_conf"]["n_mels"] * kwargs["frontend_conf"]["lfr_m"]
    cache_encoder = {
            "start_idx": 0, "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
            "cif_alphas": torch.zeros((batch_size, 1)), "chunk_size": chunk_size,
            "encoder_chunk_look_back": encoder_chunk_look_back, "last_chunk": False, "opt": None,
            "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)),
            "tail_chunk": False
    }
    cache["encoder"] = cache_encoder
    cache["encoder"]["start_idx"] = torch.tensor([0])
    cache["encoder"]["chunk_size"] = torch.tensor(chunk_size)
    if export == 1: return cache
    del cache["encoder"]
    cache_decoder = {
            "decode_fsmn": None, "decoder_chunk_look_back": decoder_chunk_look_back, "opt": None,
            "chunk_size": chunk_size
    }
    cache["decoder"] = cache_decoder
    cache["decoder"]["chunk_size"] = torch.tensor(chunk_size)
    cache["frontend"] = {}
    cache["prev_samples"] = torch.empty(0)

    return cache

modelname = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
model_name = "iic/" + modelname
output_dir = "bmodel/" + modelname + "/models/onnx/"
os.makedirs(output_dir, exist_ok=True)

## online model encoder
model_class = "ParaformerStreamingEncoder"
model, kwargs = AutoModel.build_model(
        model=model_name,
        model_class=model_class,
        disable_update=True,
        device="cpu",
)

cache = dict()
init_cache(cache, export=1, **kwargs)
speech = torch.randn(1, 30, 560)
speech_lengths = torch.tensor([30], dtype=torch.int32)
is_final = torch.tensor([0], dtype=torch.int32)
dummy_input = (speech, cache, is_final)
input_names = ["speech", "start_idx", "cif_hidden",
               "cif_alphas", "chunk_size", "encoder_chunk_look_back", "last_chunk",
               "feats", "tail_chunk"]
dynamic_axes = {'speech':{1:'speech_len'}, "feats":{1:'feats_len'}}
torch.onnx.export(model, dummy_input, output_dir + 'encoder.onnx', input_names=input_names, dynamic_axes=dynamic_axes, do_constant_folding=False)

## online model decoder 0
model_class = "ParaformerStreamingDecoder"
model, kwargs = AutoModel.build_model(
        model=model_name,
        model_class=model_class,
        disable_update=True,
        device="cpu",
)

cache = dict()
init_cache(cache, export=2, **kwargs)
encoder_out = torch.randn(1, 30, 512)
encoder_out_lens = torch.tensor([30], dtype=torch.int32)
pre_acoustic_embeds = torch.randn(1, 30, 512)
pre_token_length = torch.tensor([30], dtype=torch.int32)
is_final = torch.tensor([0], dtype=torch.int32)
dummy_input = (encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length, cache, is_final)
input_names = ["encoder_out", "encoder_out_lens", "pre_acoustic_embeds",
               "pre_token_length", "chunk_size", "decoder_chunk_look_back"]
dynamic_axes = {'encoder_out':{1:'feat_len'}, 'encoder_out_lengths':{0:'feat_len'}, 'pre_acoustic_embeds':{1:'unknown_len'}, 'pre_token_length':{0:'unknown_len'}}
torch.onnx.export(model, dummy_input, output_dir + 'decoder0.onnx', input_names=input_names, dynamic_axes=dynamic_axes, do_constant_folding=False)

## online model decoder 1
model_class = "ParaformerStreamingDecoder"
model, kwargs = AutoModel.build_model(
        model=model_name,
        model_class=model_class,
        disable_update=True,
        device="cpu",
)

cache = dict()
init_cache(cache, export=3, **kwargs)
encoder_out = torch.randn(1, 30, 512)
encoder_out_lens = torch.tensor([30], dtype=torch.int32)
pre_acoustic_embeds = torch.randn(1, 20, 512)
pre_token_length = torch.tensor([20], dtype=torch.int64)
is_final = torch.tensor([0], dtype=torch.int32)
cache['decoder']['decode_fsmn'] = [ torch.randn((1, 512, 25)) for _ in range(16) ]
dummy_input = (encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length, cache, is_final)
# input_names = ["encoder_out", "encoder_out_lens", "pre_acoustic_embeds",
#             "pre_token_length", "chunk_size", "decoder_chunk_look_back"]
#input_names = ["encoder_out", "pre_acoustic_embeds"]
#dynamic_axes = {'encoder_out':{1:'feat_len'}, 'encoder_out_lengths':{0:'feat_len'}, 'pre_acoustic_embeds':{1:'unknown_len'}, 'pre_token_length':{0:'unknown_len'}}
            #dynamic_axes = {'encoder_out':{1:'feat_len'}, 'pre_acoustic_embeds':{1:'unknown_len'}}
# for i in range(16):
#     input_names.append("fsmn_cache" + str(i))
#     dynamic_axes['fsmn_cache' + str(i)] = {2:'fsmn_len'}
# input_names.append("pre_acoustic_embeds")
dynamic_axes = {'onnx::MatMul_0':{1:'feat_len'}, 'input.1':{1:'unknown_len'}}
for i in range(16):
    dynamic_axes["onnx::Slice_" + str(i+4)] = {2:'fsmn_len'}
torch.onnx.export(model, dummy_input, output_dir + 'decoder1.onnx',
        dynamic_axes=dynamic_axes, do_constant_folding=False)

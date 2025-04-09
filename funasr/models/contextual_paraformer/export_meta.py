#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import types

from funasr.register import tables
from funasr.models.seaco_paraformer.export_meta import ContextualEmbedderExport


class ContextualEmbedderExport2(ContextualEmbedderExport):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.embedding = model.bias_embed
        model.bias_encoder.batch_first = False
        self.bias_encoder = model.bias_encoder
    
    def export_dummy_inputs(self):
        hotword = torch.tensor(
            [
                [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
                [100, 101, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
                [100, 101, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        # hotword_length = torch.tensor([10, 2, 1], dtype=torch.int32)
        return (hotword)


def export_rebuild_model(model, **kwargs):
    is_onnx = kwargs.get("type", "onnx") == "onnx"

    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    # little difference with bias encoder with seaco paraformer
    embedder_class = ContextualEmbedderExport2
    embedder_model = embedder_class(model, onnx=is_onnx)

    if kwargs["decoder"] == "ParaformerSANMDecoder":
        kwargs["decoder"] = "ParaformerSANMDecoderOnline"
    decoder_class = tables.decoder_classes.get(kwargs["decoder"] + "Export")
    model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)
    model.feats_dim = 560

    import copy

    backbone_model = copy.copy(model)

    # backbone
    part = "encoder"
    if part == "encoder":
        backbone_model.forward = types.MethodType(export_backbone_forward_encoder, backbone_model)
        backbone_model.export_dummy_inputs = types.MethodType(
            export_backbone_dummy_inputs_encoder, backbone_model
        )
        backbone_model.export_input_names = types.MethodType(
            export_backbone_input_names_encoder, backbone_model
        )
        backbone_model.export_output_names = types.MethodType(
            export_backbone_output_names_encoder, backbone_model
        )
        backbone_model.export_dynamic_axes = types.MethodType(
            export_backbone_dynamic_axes_encoder, backbone_model
        )
        backbone_model.export_name = types.MethodType(export_backbone_name_encoder, backbone_model)
    elif part == "decoder":
        backbone_model.forward = types.MethodType(export_backbone_forward_decoder, backbone_model)
        backbone_model.export_dummy_inputs = types.MethodType(
            export_backbone_dummy_inputs_decoder, backbone_model
        )
        backbone_model.export_input_names = types.MethodType(
            export_backbone_input_names_decoder, backbone_model
        )
        backbone_model.export_output_names = types.MethodType(
            export_backbone_output_names_decoder, backbone_model
        )
        backbone_model.export_dynamic_axes = types.MethodType(
            export_backbone_dynamic_axes_decoder, backbone_model
        )
        backbone_model.export_name = types.MethodType(export_backbone_name_decoder, backbone_model)
    
    embedder_model.export_name = "model_eb"
    #backbone_model.export_name = "model"

    return backbone_model, embedder_model


def export_backbone_forward_ori(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    bias_embed: torch.Tensor,
):
    batch = {"speech": speech, "speech_lengths": speech_lengths}

    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    pre_acoustic_embeds, pre_token_length, _, _ = self.predictor(enc, mask)
    pre_token_length = pre_token_length.floor().type(torch.int32)

    decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length, bias_embed)
    decoder_out = torch.log_softmax(decoder_out, dim=-1)

    return decoder_out, pre_token_length


def export_backbone_forward_encoder(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ):
    batch = {"speech": speech, "speech_lengths": speech_lengths}
    
    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    hidden, alphas, token_num = self.predictor(enc, mask)
    return enc, hidden, alphas, token_num


def export_backbone_forward_decoder(
            self,
            enc: torch.Tensor,
            enc_len: torch.Tensor,
            pre_acoustic_embeds: torch.Tensor,
            pre_token_length: torch.Tensor,
            bias_embed: torch.Tensor,
    ):
    decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length, bias_embed)
    decoder_out = torch.log_softmax(decoder_out, dim=-1)

    return decoder_out


def export_backbone_dummy_inputs_ori(self):
    speech = torch.randn(2, 30, self.feats_dim)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    bias_embed = torch.randn(2, 1, 512)
    return (speech, speech_lengths, bias_embed)


def export_backbone_dummy_inputs_encoder(self):
    speech = torch.randn(2, 30, self.feats_dim)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    return (speech, speech_lengths) #export encoder


def export_backbone_dummy_inputs_decoder(self):
    enc = torch.randn(2, 30, 512)
    enc_len = torch.tensor([6, 30], dtype=torch.int32)
    pre_acoustic_embeds = torch.randn(2, 9, 512)
    pre_token_length = torch.tensor([6, 9], dtype=torch.int32)
    bias_embed = torch.randn(2, 1, 512)
    return (enc, enc_len, pre_acoustic_embeds, pre_token_length, bias_embed) #export decoder


def export_backbone_input_names_ori(self):
    return ["speech", "speech_lengths", "bias_embed"]


def export_backbone_input_names_encoder(self):
    return ['speech', 'speech_lengths'] #export encoder


def export_backbone_input_names_decoder(self):
    return ['enc', 'enc_len', 'pre_acoustic_embeds', 'pre_token_length', 'bias_embed'] #export decoder


def export_backbone_output_names_ori(self):
    return ["logits", "token_num"]


def export_backbone_output_names_encoder(self):
    return ['enc', 'hidden', 'alphas', 'token_num'] #export encoder


def export_backbone_output_names_decoder(self):
    return ['logits'] #export decoder


def export_backbone_dynamic_axes_ori(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "bias_embed": {0: "batch_size", 1: "num_hotwords"},
        "logits": {0: "batch_size", 1: "logits_length"},
    }


def export_backbone_dynamic_axes_encoder(self):
    #encoder
    return {
        'speech': {
            0: 'batch_size',
            1: 'feats_length'
        },
        'speech_lengths': {
            0: 'batch_size',
        },
        'enc': {
            0: 'batch_size',
            1: 'feats_length'
        },
        'hidden': {
            0: 'batch_size',
            1: 'logits_length_plus'
        },
        'alphas': {
            0: 'batch_size',
            1: 'logits_length_plus'
        },
        # 'token_num': {
        #     0: 'batch_size',
        # },
    }


def export_backbone_dynamic_axes_decoder(self):
    #decoder
    return {
        'enc': {
            0: 'batch_size',
            1: 'feats_length'
        },
        'enc_len': {
            0: 'batch_size',
        },
        'pre_acoustic_embeds': {
            0: 'batch_size',
            1: 'unknown_length'
        },
        'pre_token_length': {
            0: 'batch_size',
        },
        'bias_embed': {
            0: 'batch_size',
            1: 'num_hotwords'
        },
        'logits': {
            0: 'batch_size',
            1: 'logits_length'
        },
    }

def export_backbone_name_ori(self):
    return "model.onnx"


def export_backbone_name_encoder(self):
    return 'encoder.onnx'


def export_backbone_name_decoder(self):
    return 'decoder.onnx'

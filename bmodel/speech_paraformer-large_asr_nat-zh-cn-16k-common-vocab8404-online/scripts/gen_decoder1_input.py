import numpy as np
import torch

encoder_out = torch.randn(1, 20, 512)
pre_acoustic_embeds = torch.randn(1, 10, 512)
pre_token_length = torch.tensor([10], dtype=torch.int64)
input = {"onnx::MatMul_0":encoder_out.detach().numpy(), "input.1":pre_acoustic_embeds.detach().numpy(), "inp":pre_token_length.detach().numpy()}
for i in range(16):
    input["onnx::Slice_" + str(i+4)] = (torch.randn((1, 512, 50))).detach().numpy()
np.savez("input_decoder1.npz", **input)

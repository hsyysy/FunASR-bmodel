import numpy as np
import torch

enc = torch.randn(1, 2000, 512)
enc_len = torch.tensor([2000], dtype=torch.int32)
pre_acoustic_embeds = torch.randn(1, 600, 512)
pre_token_length = torch.tensor([600], dtype=torch.int32)
bias_embed = torch.randn(1, 1, 512)
enc = enc.detach().numpy()
enc_len = enc_len.detach().numpy()
pre_acoustic_embeds = pre_acoustic_embeds.detach().numpy()
pre_token_length = pre_token_length.detach().numpy()
bias_embed = bias_embed.detach().numpy()
np.savez("decoder_input_1b.npz", enc=enc, enc_len=enc_len, pre_acoustic_embeds=pre_acoustic_embeds, pre_token_length=pre_token_length, bias_embed=bias_embed)

#enc = torch.randn(10, 2000, 512)
#enc_len = torch.arange(start=200, end=2200, step=200, dtype=torch.int32)
#pre_acoustic_embeds = torch.randn(10, 600, 512)
#pre_token_length = torch.arange(start=60, end=660, step=60, dtype=torch.int32)
#bias_embed = torch.randn(10, 1, 512)
#enc = enc.detach().numpy()
#enc_len = enc_len.detach().numpy()
#pre_acoustic_embeds = pre_acoustic_embeds.detach().numpy()
#pre_token_length = pre_token_length.detach().numpy()
#bias_embed = bias_embed.detach().numpy()
#np.savez("decoder_input_10b.npz", enc=enc, enc_len=enc_len, pre_acoustic_embeds=pre_acoustic_embeds, pre_token_length=pre_token_length, bias_embed=bias_embed)


import numpy as np
import torch

enc = torch.randn(1, 20, 512).detach().numpy()
enc_len = torch.tensor([20], dtype=torch.int32).detach().numpy()
acoustic_embeds = torch.randn(1, 10, 512).detach().numpy()
acoustic_embeds_len = torch.tensor([10], dtype=torch.int32).detach().numpy()
in_cache_0 = torch.randn(1,512,10).detach().numpy()
in_cache_1 = torch.randn(1,512,10).detach().numpy()
in_cache_2 = torch.randn(1,512,10).detach().numpy()
in_cache_3 = torch.randn(1,512,10).detach().numpy()
in_cache_4 = torch.randn(1,512,10).detach().numpy()
in_cache_5 = torch.randn(1,512,10).detach().numpy()
in_cache_6 = torch.randn(1,512,10).detach().numpy()
in_cache_7 = torch.randn(1,512,10).detach().numpy()
in_cache_8 = torch.randn(1,512,10).detach().numpy()
in_cache_9 = torch.randn(1,512,10).detach().numpy()
in_cache_10 = torch.randn(1,512,10).detach().numpy()
in_cache_11 = torch.randn(1,512,10).detach().numpy()
in_cache_12 = torch.randn(1,512,10).detach().numpy()
in_cache_13 = torch.randn(1,512,10).detach().numpy()
in_cache_14 = torch.randn(1,512,10).detach().numpy()
in_cache_15 = torch.randn(1,512,10).detach().numpy()
np.savez("input_decoder_1b.npz", enc=enc, enc_len=enc_len, acoustic_embeds=acoustic_embeds, acoustic_embeds_len=acoustic_embeds_len,
        in_cache_0=in_cache_0,
        in_cache_1=in_cache_1,
        in_cache_2=in_cache_2,
        in_cache_3=in_cache_3,
        in_cache_4=in_cache_4,
        in_cache_5=in_cache_5,
        in_cache_6=in_cache_6,
        in_cache_7=in_cache_7,
        in_cache_8=in_cache_8,
        in_cache_9=in_cache_9,
        in_cache_10=in_cache_10,
        in_cache_11=in_cache_11,
        in_cache_12=in_cache_12,
        in_cache_13=in_cache_13,
        in_cache_14=in_cache_14,
        in_cache_15=in_cache_15,
        )

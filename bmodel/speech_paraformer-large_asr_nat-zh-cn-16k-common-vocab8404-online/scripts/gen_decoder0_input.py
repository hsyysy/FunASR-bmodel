import numpy as np
import torch

encoder_out = torch.randn(1, 20, 512)
pre_acoustic_embeds = torch.randn(1, 10, 512)
np.savez("input_decoder0.npz", encoder_out=encoder_out.detach().numpy(),
                              pre_acoustic_embeds=pre_acoustic_embeds.detach().numpy())

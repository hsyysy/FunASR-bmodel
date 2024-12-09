import numpy as np
import torch

speech = torch.randn(1, 30, 560)
start_idx = torch.tensor([2000])
chunk_size = torch.tensor([5, 10, 5])
feats = torch.zeros((1, chunk_size[0] + chunk_size[2], 560))

np.savez("input_encoder.npz", speech=speech.detach().numpy(),
                              start_idx=start_idx.detach().numpy(),
                              chunk_size=chunk_size.detach().numpy(),
                              feats=feats.detach().numpy())

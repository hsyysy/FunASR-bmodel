import numpy as np
import torch

speech = torch.randn(1, 20, 560)
speech_lengths = torch.tensor([20], dtype=torch.int32)
speech = speech.detach().numpy()
speech_lengths = speech_lengths.detach().numpy()
np.savez("input_model_1b.npz", speech=speech, speech_lengths=speech_lengths)

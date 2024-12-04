import numpy as np
import torch

speech = torch.randn(1, 1100, 560)
speech_lengths = torch.tensor([1100], dtype=torch.int32)
speech = speech.detach().numpy()
speech_lengths = speech_lengths.detach().numpy()
np.savez("input_1batch.npz", speech=speech, speech_lengths=speech_lengths)

speech = torch.randn(10, 1100, 560)
speech_lengths = torch.arange(start=200, end=1200, step=100, dtype=torch.int32)
speech = speech.detach().numpy()
speech_lengths = speech_lengths.detach().numpy()
np.savez("input_10batch.npz", speech=speech, speech_lengths=speech_lengths)


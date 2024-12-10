import numpy as np
import torch

speech = torch.randn(1, 2000, 560)
speech_lengths = torch.tensor([2000], dtype=torch.int32)
speech = speech.detach().numpy()
speech_lengths = speech_lengths.detach().numpy()
np.savez("encoder_input_1b.npz", speech=speech, speech_lengths=speech_lengths)

#speech = torch.randn(10, 2000, 560)
#speech_lengths = torch.arange(start=200, end=2200, step=200, dtype=torch.int32)
#speech = speech.detach().numpy()
#speech_lengths = speech_lengths.detach().numpy()
#np.savez("encoder_input_10b.npz", speech=speech, speech_lengths=speech_lengths)


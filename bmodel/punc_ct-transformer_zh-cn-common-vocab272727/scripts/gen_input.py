#!/bin/python3
import torch
import numpy as np

length = 200
text_indexes = torch.randint(0, 272727, (1, length)).type(torch.int32)
text_indexes = text_indexes.detach().numpy()#.astype(np.float32)
text_lengths = torch.tensor([length], dtype=torch.int32)
text_lengths = text_lengths.detach().numpy()#.astype(np.float32)
np.savez("input_0.npz", inputs=text_indexes, text_lengths=text_lengths)

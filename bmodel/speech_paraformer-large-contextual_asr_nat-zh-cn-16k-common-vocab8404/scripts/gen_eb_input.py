import numpy as np
import torch

hotword = torch.randint(high=8404, size=(100, 10), dtype=torch.int32)
hotword = hotword.detach().numpy()
np.savez("eb_input_1b.npz", hotword=hotword)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
torch.nn ONLY takes in mini-batches. 
Thereby, if I want to feed a single data to test-run,
then I need to unsqueeze(0) to transform the input to have a fake batch. 

"""

d = [1,2,3,4,5]
t = torch.tensor(d, dtype=float)
print(t.size())
t_ = t.unsqueeze(0)
print(t_.size())   # fake batch 1
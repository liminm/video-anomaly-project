import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MemoryModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres # Hard-shrinkage for sparse addressing

        # The "Memory" is a matrix of learned normal patterns
        # shape: (2000 items, 512 features)
        self.memory = nn.Parameter(torch.Tensor(mem_dim, fea_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory.size(1))
        self.memory.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # input shape: (Batch, Channel, Height, Width)
        s = input.data.shape
        l = len(s)

        # Flatten input to (N, C)
        # x is the Query
        x = input.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, s[1])
        
        # Calculate Cosine Similarity between Input (Query) and Memory Keys
        # y = softmax(x * M^T)
        att_weight = F.linear(x, self.memory) # Feat * Mem^T
        att_weight = F.softmax(att_weight, dim=1) # Attention scores

        # Sparse Addressing: Ignore weak memories
        # If the match is weak, zero it out. Force model to pick a STRONG match.
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # Re-normalize
            att_weight = F.normalize(att_weight, p=1, dim=1)

        # Retrieve Memory: M_hat = Weight * Memory
        output = F.linear(att_weight, self.memory.t()) # Att * Mem
        
        # Reshape back to image
        output = output.view(s[0], s[2], s[3], s[1])
        output = output.permute(0, 3, 1, 2)
        
        return output, att_weight

def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input - lambd)) * input / (torch.abs(input - lambd) + epsilon)
    return output
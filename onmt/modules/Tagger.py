import torch
import torch.nn as nn

"""
Tagger file
"""


class Tagger(nn.Module):
    def __init__(self):
        super(Tagger, self).__init__()


    def forward(self, memory_bank):

        return memory_bank
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Tagger file
"""

class Tagger(nn.Module):
    def __init__(self, rnn_type, hidden_size,
                 dropout=0.0):
        super(Tagger, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, memory_bank):
        src_len, bsize, rnn_size = memory_bank.size()

        # final_layer, _ = self.rnn(memory_bank, None)
        # t_copy = F.sigmoid(self.linear(memory_bank))
        t_copy = self.linear(memory_bank)
        t_copy = self.linear2(F.tanh(t_copy))
        t_copy = F.log_softmax(t_copy, dim=-1)
        # print("WEIGHT", self.linear.weight)
        # memory_bank.detach_()
        # t_copy = self.linear(memory_bank)
        # t_copy = F.sigmoid(t_copy)
        # print(t_copy[:, 0][:10])
        # final_layer = F.tanh(self.linear1(memory_bank))
        # t_copy = F.sigmoid(self.linear(final_layer))
        return t_copy
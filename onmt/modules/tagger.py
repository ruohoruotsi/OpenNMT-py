import torch.nn as nn

"""
Tagger file
"""


class Tagger(nn.Module):
    def __init__(self, rnn_type, hidden_size,
                 dropout=0.0):
        super(Tagger, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, memory_bank):
        src_len, bsize, rnn_size = memory_bank.size()
        # t_copy = self.linear(memory_bank)
        # t_copy = self.linear2(torch.tanh(t_copy))
        t_copy = self.linear2(memory_bank)
        return t_copy

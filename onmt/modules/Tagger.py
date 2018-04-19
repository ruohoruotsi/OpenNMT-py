import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Tagger file
"""

def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.modules.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq


class Tagger(nn.Module):
    def __init__(self, rnn_type, hidden_size,
                 dropout=0.0):
        super(Tagger, self).__init__()
        self.rnn, self.no_pack_padded_seq  = \
            rnn_factory(rnn_type,
                        input_size=hidden_size,
                        hidden_size=hidden_size//2,
                        num_layers=1,
                        dropout=dropout,
                        bidirectional=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, memory_bank):
        src_len, bsize, rnn_size = memory_bank.size()

        # memory_bank.detach_()
        # final_layer, _ = self.rnn(memory_bank, None)
        # t_copy = F.sigmoid(self.linear(final_layer
        # print(memory_bank)
        # print("WEIGHT", self.linear.weight)
        # t_copy = self.linear(memory_bank.view(-1, rnn_size)).view(src_len, bsize, -1)

        t_copy = self.linear(memory_bank)
        t_copy = F.sigmoid(t_copy)
        # print(t_copy[:, 0][:10])
        # exit()
        # print(t_copy.shape)
        return t_copy#t_copy
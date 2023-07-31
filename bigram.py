import torch
import torch.nn as nn
from torch.nn import functional as F
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets):
        logits = self.token_embedding_table(index) # (batch, time, count)
        return logits

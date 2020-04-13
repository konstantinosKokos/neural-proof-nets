import torch
from torch.nn import Module, Embedding, Dropout
from torch import Tensor, LongTensor


class ComplexEmbedding(Module):
    def __init__(self, num_classes: int, dim: int, dropout_rate: float = 0.1):
        super(ComplexEmbedding, self).__init__()
        self.amplitudes = Embedding(num_embeddings=num_classes, embedding_dim=dim)
        self.frequencies = Embedding(num_embeddings=num_classes, embedding_dim=dim)
        self.dropout = Dropout(dropout_rate)

    def forward(self, words: LongTensor) -> Tensor:
        return self.embed(words, 0)

    def embed(self, words: LongTensor, start_from: int) -> Tensor:
        rank = len(words.shape)

        if rank == 2:
            # words := B x S
            positions = torch.arange(start=start_from + 1, end=words.shape[1] + 1 + start_from,
                                     device=words.device).view(1, -1).repeat(words.shape[0], 1)

            amplitudes = self.amplitudes(words)
            frequencies = self.frequencies(words)
            phases = frequencies * positions.unsqueeze(-1)

            real = amplitudes * torch.cos(phases)
            imag = amplitudes * torch.sin(phases)

            return self.dropout(torch.cat([real, imag], dim=-1))

        elif rank == 1:
            # words := B
            words = words.unsqueeze(1)
            ret = self.embed(words, start_from)
            return ret.squeeze(1)



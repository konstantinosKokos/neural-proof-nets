import torch
from torch.nn import Module, Embedding, Dropout
from torch import Tensor, LongTensor
from torch import bmm


class ComplexEmbedding(Module):
    def __init__(self, num_classes: int, dim: int, dropout_rate: float = 0.1):
        super(ComplexEmbedding, self).__init__()
        self.amplitudes = Embedding(num_embeddings=num_classes, embedding_dim=dim)
        self.frequencies = Embedding(num_embeddings=num_classes, embedding_dim=dim)
        self.dropout = Dropout(dropout_rate)

    def forward(self, words: LongTensor) -> Tensor:
        return self.embed(words, 0)

    def embed(self, token_ids: LongTensor, start_from: int) -> Tensor:
        rank = len(token_ids.shape)

        if rank == 2:
            # tokens := B x S
            positions = torch.arange(start=start_from + 1, end=token_ids.shape[1] + 1 + start_from,
                                     device=token_ids.device).view(1, -1).repeat(token_ids.shape[0], 1)

            amplitudes = self.amplitudes(token_ids)
            frequencies = self.frequencies(token_ids)
            phases = frequencies * positions.unsqueeze(-1)

            real = amplitudes * torch.cos(phases)
            imag = amplitudes * torch.sin(phases)

            return self.dropout(torch.cat([real, imag], dim=-1))

        elif rank == 1:
            # tokens := B
            token_ids = token_ids.unsqueeze(1)
            ret = self.embed(token_ids, start_from)
            return ret.squeeze(1)

    def invert(self, token_reprs: Tensor, start_from: int) -> Tensor:
        rank = len(token_reprs.shape)

        if rank == 3:
            # words := B x S x D

            # S
            positions = torch.arange(start=start_from + 1, end=token_reprs.shape[1] + 1 + start_from,
                                     device=token_reprs.device)

            # A x D
            freq_map = self.frequencies(torch.arange(self.frequencies.num_embeddings, device=token_reprs.device))
            # A x D x S
            phase_map = freq_map.unsqueeze(-1) * positions
            # A x D x S
            phase_map = torch.cat([torch.cos(phase_map), torch.sin(phase_map)], dim=1)

            # A x D
            amplitude_map = self.amplitudes.weight
            amplitude_map = torch.cat([amplitude_map, amplitude_map], dim=-1)

            # A x D x S
            inversion_map = amplitude_map.unsqueeze(-1) * phase_map

            # S x D x A
            inversion_map = inversion_map.permute(-1, -2, -3)
            # S x B x D
            token_reprs = token_reprs.transpose(-3, -2)
            return self.dropout(bmm(token_reprs, inversion_map)).transpose(-2, -3)

        elif rank == 2:
            # tokens := B x D
            token_reprs = token_reprs.unsqueeze(1)
            ret = self.invert(token_reprs, start_from)
            return ret.squeeze(1)


from typing import *

from torch import Tensor, LongTensor
from torch.nn import Module, Sequential, Linear, functional, LayerNorm, Dropout

from PermutationParser.neural.multi_head_atn import MultiHeadAttention


class FFN(Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        super(FFN, self).__init__()
        self.linear_one = Linear(d_model, d_ff, bias=False)
        self.linear_two = Linear(d_ff, d_model, bias=False)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = functional.gelu(self.linear_one(x))
        x = self.dropout(x)
        return self.linear_two(x)


class EncoderLayer(Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int, dropout_rate: float) \
            -> None:
        super(EncoderLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.ffn = FFN(d_model=d_model, d_ff=d_intermediate)
        self.ln_mha = LayerNorm(normalized_shape=d_model)
        self.ln_ffn = LayerNorm(normalized_shape=d_model)
        self.dropout = Dropout(dropout_rate)

    def forward(self, inps: Tuple[Tensor, LongTensor]) -> Tuple[Tensor, LongTensor]:
        encoder_input, encoder_mask = inps

        encoder_input = self.dropout(encoder_input)
        mha_x = self.mha(encoder_input, encoder_input, encoder_input, encoder_mask)
        mha_x = self.dropout(mha_x)
        mha_x = encoder_input + mha_x
        mha_x = self.ln_mha(mha_x)

        ffn_x = self.ffn(mha_x)
        ffn_x = self.dropout(ffn_x)
        ffn_x = ffn_x + mha_x
        ffn_x = self.ln_ffn(ffn_x)
        return ffn_x, encoder_mask


def make_encoder(num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int,
                 dropout: float = 0.1) -> Sequential:
    return Sequential(*[EncoderLayer(num_heads, d_model, d_k, d_v, d_intermediate, dropout)
                        for _ in range(num_layers)])


class DecoderLayer(Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int, dropout: float) \
            -> None:
        super(DecoderLayer, self).__init__()
        self.dropout_rate = dropout
        self.mask_mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.ffn = FFN(d_model, d_intermediate)
        self.ln_m_mha = LayerNorm(normalized_shape=d_model)
        self.ln_mha = LayerNorm(normalized_shape=d_model)
        self.ln_ffn = LayerNorm(normalized_shape=d_model)
        self.dropout = Dropout(dropout)

    def forward(self, inps: Tuple[Tensor, LongTensor, Tensor, LongTensor]) \
            -> Tuple[Tensor, LongTensor, Tensor, LongTensor]:
        encoder_output, encoder_mask, decoder_input, decoder_mask = inps

        t = decoder_input.shape[1]
        x_drop = self.dropout(decoder_input)
        m_mha_x = self.mask_mha(x_drop, x_drop, x_drop, decoder_mask)
        m_mha_x = self.dropout(m_mha_x)
        m_mha_x = m_mha_x + x_drop
        m_mha_x = self.ln_m_mha(m_mha_x)

        mha_x = self.mha(m_mha_x, encoder_output, encoder_output, encoder_mask[:, :t, :])
        mha_x = self.dropout(mha_x)
        mha_x = mha_x + m_mha_x
        mha_x = self.ln_mha(mha_x)

        ffn_x = self.ffn(mha_x)
        ffn_x = self.dropout(ffn_x)
        ffn_x = ffn_x + mha_x
        ffn_x = self.ln_ffn(ffn_x)

        return encoder_output, encoder_mask, ffn_x, decoder_mask

    def infer(self, inps: Tuple[Tensor, LongTensor, Tensor, LongTensor], t: int) \
            -> Tuple[Tensor, LongTensor, Tensor, LongTensor]:
        encoder_output, encoder_mask, decoder_input, decoder_mask = inps

        x_drop = self.dropout(decoder_input)
        m_mha_x = self.mask_mha(x_drop, x_drop, x_drop, decoder_mask)
        m_mha_x = self.dropout(m_mha_x)
        m_mha_x = m_mha_x + x_drop
        m_mha_x = self.ln_m_mha(m_mha_x)

        mha_x = self.mha(m_mha_x, encoder_output, encoder_output, encoder_mask[:, :t, :])
        mha_x = self.dropout(mha_x)
        mha_x = mha_x + m_mha_x
        mha_x = self.ln_mha(mha_x)

        ffn_x = self.ffn(mha_x)
        ffn_x = self.dropout(ffn_x)
        ffn_x = ffn_x + mha_x
        ffn_x = self.ln_ffn(ffn_x)

        return encoder_output, encoder_mask, ffn_x, decoder_mask


def make_decoder(num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int,
                 dropout: float = 0.1) -> Sequential:
    return Sequential(*[DecoderLayer(num_heads, d_model, d_k, d_v, d_intermediate, dropout)
                        for _ in range(num_layers)])

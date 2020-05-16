from typing import *

from torch import Tensor, LongTensor
from torch.nn import Module, Sequential, Linear, functional, LayerNorm, Dropout

from Parser.neural.multi_head_atn import MultiHeadAttention


class FFN(Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1, d_out: Optional[int] = None):
        super(FFN, self).__init__()
        self.linear_one = Linear(d_model, d_ff, bias=True)
        self.linear_two = Linear(d_ff, d_model, bias=True) if d_out is None else Linear(d_ff, d_out)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = functional.gelu(self.linear_one(x))
        x = self.dropout(x)
        return self.linear_two(x)


class EncoderLayer(Module):
    def __init__(self, num_heads: int, d_model: int, d_atn: int, d_v: int, d_intermediate: int, dropout_rate: float) \
            -> None:
        super(EncoderLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.mha = MultiHeadAttention(num_heads,
                                      d_q_in=d_model, d_k_in=d_model, d_v_in=d_model, d_atn=d_atn, d_v=d_v,
                                      d_out=d_model, dropout_rate=dropout_rate)
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
    def __init__(self, num_heads_enc: int, num_heads_dec: int, d_encoder: int, d_decoder: int,
                 d_atn_enc: int, d_atn_dec: int, d_v_enc: int, d_v_dec: int, d_interm: int, dropout_rate: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.dropout = Dropout(dropout_rate)
        self.mask_mha = MultiHeadAttention(num_heads=num_heads_dec, d_q_in=d_decoder, d_k_in=d_decoder,
                                           d_v_in=d_decoder, d_atn=d_atn_dec, d_v=d_v_dec,
                                           d_out=d_decoder, dropout_rate=dropout_rate)
        self.ln_masked_mha = LayerNorm(d_decoder)
        self.mha = MultiHeadAttention(num_heads=num_heads_enc, d_q_in=d_decoder, d_k_in=d_encoder,
                                      d_v_in=d_encoder, d_atn=d_atn_enc, d_v=d_v_enc,
                                      d_out=d_decoder, dropout_rate=dropout_rate)
        self.ln_mha = LayerNorm(d_decoder)
        self.ffn = FFN(d_model=d_decoder, d_ff=d_interm, dropout_rate=dropout_rate)
        self.ln_ffn = LayerNorm(d_decoder)

    def forward(self, inps: Tuple[Tensor, LongTensor, Tensor, LongTensor]) \
            -> Tuple[Tensor, LongTensor, Tensor, LongTensor]:
        encoder_out, encoder_mask, decoder_in, decoder_mask = inps

        t = decoder_in.shape[1]

        x_drop = self.dropout(decoder_in)
        dec_atn = self.mask_mha(x_drop, x_drop, x_drop, decoder_mask)
        dec_atn = dec_atn + x_drop
        dec_atn = self.ln_masked_mha(dec_atn)

        enc_dec_atn = self.mha(dec_atn, encoder_out, encoder_out, encoder_mask[:, :t, :])
        enc_dec_atn = self.dropout(enc_dec_atn)
        enc_dec_atn = dec_atn + enc_dec_atn
        enc_dec_atn = self.ln_mha(enc_dec_atn)

        out = self.ffn(enc_dec_atn)
        out = self.dropout(out)
        out = out + enc_dec_atn
        out = self.ln_ffn(out)
        return encoder_out, encoder_mask, out, decoder_mask


def make_decoder(num_layers: int, num_heads_enc: int, num_heads_dec: int, d_encoder: int, d_decoder: int,
                 d_atn_enc: int, d_atn_dec: int, d_v_enc: int, d_v_dec: int, d_interm: int, dropout_rate: float = 0.1):
    return Sequential(*[DecoderLayer(num_heads_enc=num_heads_enc, num_heads_dec=num_heads_dec,
                                     d_encoder=d_encoder, d_decoder=d_decoder, d_atn_enc=d_atn_enc, d_atn_dec=d_atn_dec,
                                     d_v_enc=d_v_enc, d_v_dec=d_v_dec, d_interm=d_interm, dropout_rate=dropout_rate)
                        for _ in range(num_layers)])
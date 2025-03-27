import torch
from torch import nn, Tensor

from xtransformer.transformer.config import AttentionConfig
from xtransformer.transformer.encoder_only import EncoderOnly
from xtransformer.transformer.transformer_decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super(Transformer, self).__init__()
        self.cfg = cfg
        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.encoder = EncoderOnly(cfg)
        self.decoder = TransformerDecoder(cfg)

    def forward(self, input_x: Tensor, input_mask: Tensor, target_x: Tensor, target_mask: Tensor) -> torch.Tensor:
        encoder_attention = self.encoder(input_x, input_mask)
        decoder_attention = self.decoder(target_x, target_mask, encoder_attention, input_mask)
        out = self.output_layer(decoder_attention)
        return out

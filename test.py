import torch

from xtransformer.embedding import Embedding
from xtransformer.embedding import EmbeddingConfig
from xtransformer.transformer import DecoderOnly
from xtransformer.transformer import EncoderOnly
from xtransformer.transformer.config import AttentionConfig
from xtransformer.transformer.transformer import Transformer

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_x = torch.tensor([[100, 256, 244, 88, 98, 555, 658, 1, 1, 1]]).to(device)
    input_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]).to(device)

    target_x = torch.tensor([[89, 33, 556, 43, 89, 213, 1, 1]]).to(device)
    target_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1]]).to(device)

    attention_cfg = AttentionConfig()
    emb_cfg = EmbeddingConfig()

    pos_emb = Embedding(emb_cfg).to(device)

    decoder_only_model = DecoderOnly(attention_cfg).to(device)
    encoder_only_model = EncoderOnly(attention_cfg).to(device)
    transformer_model = Transformer(attention_cfg).to(device)

    input_emb_x = pos_emb(input_x)
    target_emb_x = pos_emb(target_x)

    decoder_only_out = decoder_only_model(input_emb_x, input_mask)
    print("decoder_only_out:{}".format(decoder_only_out))
    encoder_only_out = encoder_only_model(input_emb_x, input_mask)
    print("encoder_only_out:{}".format(encoder_only_out))
    transformer_out = transformer_model(input_emb_x, input_mask, target_emb_x, target_mask)
    print("transformer_out:{}".format(transformer_out))

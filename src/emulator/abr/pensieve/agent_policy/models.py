import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn import Transformer

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, pad_idx, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = generate_square_subsequent_mask(src_seq_len, device)
    #src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class Seq2SeqWithEmbeddingmodClass(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 input_size: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_classes: int = 3231):  # Add num_classes parameter
        super(Seq2SeqWithEmbeddingmodClass, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.embed_layer1 = nn.Linear(input_size, 256)
        self.embed_layer2 = nn.Linear(256, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.de_embed_layer1 = nn.Linear(emb_size, 256)
        self.de_embed_layer2 = nn.Linear(256, num_classes)  # Change output layer to num_classes
        self.softmax = nn.Softmax(dim=-1)  # Apply softmax over the class dimension
        self.relu = nn.ReLU()

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor
    ):
        # 1) embed
        src = src.float()
        src = self.relu(self.embed_layer2(self.relu(self.embed_layer1(src))))
        # trg = self.relu(self.embed_layer2(self.relu(self.embed_layer1(trg))))

        # 2) (batch, seq, emb) -> (seq, batch, emb)
        src = src.permute(1, 0, 2)
        # trg = trg.permute(1, 0, 2)

        # 3) positional encoding
        src_pos = self.positional_encoding(src)  # shape [seq, batch, emb]
        # tgt_pos = self.positional_encoding(trg)

        # 4) revert to (batch, seq, emb)
        src_pos = src_pos.permute(1, 0, 2)  # [batch, seq, emb]
        # tgt_pos = tgt_pos.permute(1, 0, 2)

        # 5) run the transformer
        # 5a) switch shape back to [seq, batch, emb] for the encoder call
        # src_pos = src_pos.permute(1, 0, 2)  # => [seq, batch, emb]
        memory = self.transformer.encoder(
            src_pos,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask
        )  # shape [seq, batch, emb]

        # 5b) decode 
        # decode expects the memory as [seq, batch, emb] if batch_first=False 
        # so do the same for tgt_pos
        # tgt_pos = tgt_pos.permute(1, 0, 2)  # => [seq, batch, emb]
        # decoder_out = self.transformer.decoder(
        #     tgt_pos,
        #     memory,
        #     tgt_mask=tgt_mask,
        #     memory_key_padding_mask=memory_key_padding_mask,
        #     tgt_key_padding_mask=tgt_padding_mask
        # )  # => [seq, batch, emb]

        # 6) final projection
        # decoder_out = decoder_out.permute(1, 0, 2)  # => [batch, seq, emb]
        # logits = self.relu(self.de_embed_layer2(self.relu(self.de_embed_layer1(decoder_out))))
        # probs = self.softmax(logits)

        # now we can return both
        # return probs, memory
        return memory
    
class Seq2SeqWithEmbeddingmodClassMultiHead(nn.Module):
    def __init__(self,
                num_encoder_layers: int,
                num_decoder_layers: int,
                input_size: int,
                emb_size: int,
                nhead: int,
                dim_feedforward: int = 512,
                dropout: float = 0.1,
                num_heads: int = 5,
                max_bucket: int = 50):
        super().__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.embed_layer1 = nn.Linear(input_size, 256)
        self.embed_layer2 = nn.Linear(256, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.de_embed_layer1 = nn.Linear(emb_size, 256)
        
        # One output head per feature (features 1 to 5)
        self.output_heads = nn.ModuleList([
            nn.Linear(256, max_bucket) for _ in range(num_heads)
        ])
        self.relu = nn.ReLU()

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        src = self.relu(self.embed_layer2(self.relu(self.embed_layer1(src.float()))))
        # trg = self.relu(self.embed_layer2(self.relu(self.embed_layer1(trg.float()))))

        src = self.positional_encoding(src)
        # trg = self.positional_encoding(trg)

        memory = self.transformer.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask
        )  # (B, T, emb)
        return memory
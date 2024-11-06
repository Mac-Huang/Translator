import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, N=6, d_model=768, max_seq_len=512, d_ff=3072, head=8, dropout=0.1):
        super(Transformer, self).__init__()

        # Embedding layers with positional encoding
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab_size, d_model),
            PositionalEncoding(d_model, max_seq_len, dropout)
        )
        self.tgt_embed = nn.Sequential(
            Embeddings(tgt_vocab_size, d_model),
            PositionalEncoding(d_model, max_seq_len, dropout)
        )

        # Transformer components
        self.encoder = Encoder(
            EncoderLayer(d_model, MultiHeadAttention(d_model, head, dropout), 
                         FeedForward(d_model, d_ff, dropout), dropout), N)
        self.decoder = Decoder(
            DecoderLayer(d_model, MultiHeadAttention(d_model, head, dropout), 
                         MultiHeadAttention(d_model, head, dropout), 
                         FeedForward(d_model, d_ff, dropout), dropout), N)
        
        # Output generator
        self.generator = Generator(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        # print(f"[Transformer] Source shape before embedding: {src.shape}")
        encoded = self.encoder(self.src_embed(src), src_mask)
        # print(f"[Transformer] Encoded shape: {encoded.shape}")
        
        # print(f"[Transformer] Target shape before embedding: {tgt.shape}")
        decoded = self.decoder(self.tgt_embed(tgt), tgt_mask, encoded, src_mask)
        # print(f"[Transformer] Decoded shape: {decoded.shape}")
        
        output = self.generator(decoded)
        # print(f"[Transformer] Output shape after generator: {output.shape}")
        return output

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.dim_size)
        
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, dim_size, atten, ff, dropout):
        super(EncoderLayer, self).__init__()
        self.dim_size = dim_size
        self.atten = atten
        self.ff = ff
        self.sublayer = clones(SublayerConnection(dim_size, dropout), 2)
        
    def forward(self, x, src_mask):
        # print(f"[EncoderLayer] Input shape: {x.shape}")
        x = self.sublayer[0](x, lambda x: self.atten(x, x, x, src_mask))
        # print(f"[EncoderLayer] Shape after attention: {x.shape}")
        x = self.sublayer[1](x, self.ff)
        # print(f"[EncoderLayer] Output shape: {x.shape}")
        return x

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.dim_size)
        
    def forward(self, x, tgt_mask, memory, src_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask, memory, src_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, dim_size, self_atten, src_atten, ff, dropout):
        super(DecoderLayer, self).__init__()
        self.dim_size = dim_size
        self.self_atten = self_atten
        self.src_atten = src_atten
        self.ff = ff
        self.sublayer = clones(SublayerConnection(dim_size, dropout), 3)
        
    def forward(self, x, tgt_mask, memory, src_mask):
        # print(f"[DecoderLayer] Input shape: {x.shape}")
        x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, tgt_mask))
        # print(f"[DecoderLayer] Shape after self-attention: {x.shape}")
        x = self.sublayer[1](x, lambda x: self.src_atten(x, memory, memory, src_mask))
        # print(f"[DecoderLayer] Shape after encoder-decoder attention: {x.shape}")
        x = self.sublayer[2](x, self.ff)
        # print(f"[DecoderLayer] Output shape: {x.shape}")
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % head == 0, "d_model must be divisible by the number of heads"
        self.d_k = d_model // head
        self.head = head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        
    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # print(f"[Before Attention] Initial Mask shape: {mask.shape if mask is not None else 'None'}")

        # Debugging shape of scores and mask
        # print(f"[Attention] Scores shape: {scores.shape}")
        if mask is not None:
            if mask.dim() == 3:  # Batch x 1 x Seq length
                mask = mask.unsqueeze(1)
            # print(f"[Attention] Mask shape after potential unsqueeze: {mask.shape}")

            scores = scores.masked_fill(mask == 0, -1e9)
            
        p_attn = F.softmax(scores, dim=-1)
        if dropout:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        ##################
        if mask is not None:
            mask = mask.unsqueeze(1)
        ##################
        Q, K, V = [linear(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) 
                   for linear, x in zip(self.linears, (query, key, value))]
        
        # print(f"[MultiHeadAttention] After linear transformation - Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
        
        x, _ = self.attention(Q, K, V, mask, self.dropout)
        # print(f"[Attention] Output shape: {x.shape}")
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        
        output = self.linears[-1](x)
        # print(f"[MultiHeadAttention] Output shape: {output.shape}")
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class SublayerConnection(nn.Module):
    def __init__(self, dim_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(dim_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # Residual Connection

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class LayerNorm(nn.Module):
    def __init__(self, dim_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(dim_size))
        self.b = nn.Parameter(torch.zeros(dim_size))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(int(max_seq_len), d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # print(f"[PositionalEncoding] Input shape: {x.shape}")
        # print(f"[PositionalEncoding] Positional encoding shape: {self.pe[:, :x.size(1), :].shape}")
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def test_transformer():
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    model = Transformer(src_vocab_size, tgt_vocab_size, N=1)

    # Dummy input
    src = torch.randint(0, src_vocab_size, (32, 512))  # (batch_size, seq_length)
    tgt = torch.randint(0, tgt_vocab_size, (32, 512))
    src_mask = torch.ones((32, 1, 512))  # (batch_size, 1, seq_length)
    tgt_mask = torch.ones((32, 1, 512))  # (batch_size, 1, seq_length)

    output = model(src, tgt, src_mask, tgt_mask)
    # Output shape: torch.Size([32, 512, 10000])
    # print(f"Output shape: {output.shape}")

# def __main__():
#     test_transformer()

# if __name__ == "__main__":
#     __main__()

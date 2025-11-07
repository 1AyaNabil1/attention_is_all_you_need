from turtle import shape
import torch 
import torch.nn as nn
import math

class InputEmbedding(torch.nn.Module):

    def __init__(self, d_model, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:     # dropout to make our model less overfitting
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        """Let's build the positional encoding matrix of shape sequence length to d_model
        Why sequence length to d_model? because for each position we want a vector of size d_model (512) and the sequence length of them
        because the maximum length of the sentence is sequence length (5000)"""

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1) --> this vector can go from 0 to sentence length -1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # unsqueeze to make it (seq_len, 1)

        # Create the denominator of the formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # we used log for numerical stability

        # Sin() applied for the even positions 
        # Cos() applied for the odd positions

        # Apply the sine to even positions
        pe[:, 0::2] = torch.sin(position * div_term)  # 0::2 means start from 0 and take every second element

        # Apply the cosine to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)  # 1::2 means start from 1 and take every second element

        # Add a batch dimension --> so that we can apply it to the whole sentences so to all the batch of the sentences
        # Because now the shape is sequence length to the model but we  will have a batch of sentences
        # So what we do is we add a new dimension to this PE

        pe = pe.unsqueeze(0)  # shape (1, seq_len, d_model) 

        # Then we register pe as a buffer so that it is not considered as a model parameter
        self.register_buffer('pe', pe) 

    def forward(self, x):
        """this function used to add every position encoding to the input embeddings"""
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # we slice the pe to have the same length as the input x
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps 

        # the alpha will be the one that will scale the normalized value(multiplied)
        # the bias will be the one that will shift the normalized value (added)
        self.alpha = nn.Parameter(torch.ones((1,)))  # Multiplied by 1  
        self.bias = nn.Parameter(torch.zeros((1,)))  # Added to 0

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)  # mean over the last dimension
        std = x.std(dim = -1, keepdim=True)    # std over the last dimension

        # Apply the formula that is defined on obsidian :)
        normalized_x = (x - mean) / (std + self.eps)  # normalize the input

        return self.alpha * normalized_x + self.bias  # scale and shift
    

class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2 

    def forward(self, x):
        # we have input sentence of shape (batch_size, seq_len, d_model) then we will convert it  --> (batch_size, seq_len, d_ff) then convert it back --> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))  # apply relu activation in between the two linear layers
        # and this is our feed forward block DONE :")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h  #we need to divide the embedding dimension into h heads 
        assert d_model % h == 0, "d_model must be divisible by h" # this is called DK

        self.d_k = d_model // h  # dimension of each head
        self.w_q = nn.Linear(d_model, d_model)  # weight matrix for query WQ
        self.w_k = nn.Linear(d_model, d_model)  # weight matrix for key WK
        self.w_v = nn.Linear(d_model, d_model)  # weight matrix for value WV
        # DV = DK in practice but in the paper they are different, DV is the dimension of the first vector
        self.w_o = nn.Linear(d_model, d_model)  # weight matrix for output WO
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # get the dimension of the key

        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1) / math.sqrt(d_k))  # scaled dot product attention

        # Before applying softmax we need to apply the mask, we want to hide some interaction between words we have to use mask and then we apply softmax
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)  # fill the masked positions with -inf so that after softmax they become 0

        # (Batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)  # apply softmax on the last dimension
        if dropout is not None:
            attention_scores = dropout(attention_scores)  # apply dropout to the attention scores

        return (attention_scores @ value), attention_scores  # return the weighted sum of values and the attention scores(for visualization)

    def forward(self, query, key, value, mask=None): # when we need words do not interact with other words we mask them and this is what mask means
        query = self.w_q(query)  # shape (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(key)      # shape (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(value)  # shape (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # We are going from (Batch, Seq_Len, d_model) -to-> (Batch, Seq_Len, h, d_k) -to-> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_model) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)  # shape (Batch, seq_len, d_model) --> final output after linear layer
    
class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # the paper apply the sublayer and then the normalization, many implementations do normalization first then sublayer
        # but we will follow the paper implementation

class EncoderLayer(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):  # the source mask is the mask that we want to apply to the input of the enccoder layer, to hide the interaction of the padding word with other words 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # this is way it's called self-attention, because the role of the query key and the value is X itself it's the input itself , the sentence that is watching itself
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)]) # three sublayers in the decoder block

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x is the input to the decoder block, 
        encoder_output is the output of the encoder, 
        src_mask is the mask for the source sequence applied to the encoder (translating from English (source language)), 
        tgt_mask is the mask for the target sequence applied to the decoder (to Arabic (target language))
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # self-attention sublayer of the decoder 
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # cross-attention sublayer of the decoder
        x = self.residual_connections[2](x, self.feed_forward_block) # feed-forward sublayer of the decoder
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers # list of decoder blocks
        self.norm = LayerNormalization(features)   # final layer normalization

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module): # called the projection layer because it's projecting the embedding layer into the vocabulary space

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src) # source embedding
        src = self.src_pos(src) # source positional encoding
        return self.encoder(src, src_mask)  # encode the source sequence
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x) # we take from the embedding to the vocabulary size 
    
# Build a function to connect all the previous classes, we will use the same numbers that were mentioned on the paper
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers, we don't need to create two positional encoding layers because actually they do the same job but to be more clear we will create two different layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderLayer(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

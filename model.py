import torch 
import torch.nn 
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


import torch
import torch.nn as nn
from typing import Tuple

class BiLSTMEncoder(nn.Module): # nn is a clas in pytorch that provides base class for all neural network. [!#naming conv=PascalCase]
    """Bidirectional LSTM encoder for sequence-to-sequence models.

    Encodes input sequences into hidden states and outputs, suitable for tasks like machine translation.
    Uses a bidirectional LSTM to capture context from both directions, followed by a linear layer for output projection.

    Args:
        vocab_size (int): Size of the input vocabulary.
        embedding_dim (int): Dimension of token embeddings.
        hidden_dim (int): Dimension of LSTM hidden states per direction.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout probability for regularization.
        output_dim (int): Dimension of the output (e.g., target vocabulary size for classification).

    Attributes:
        embedding (nn.Embedding): Token embedding layer.
        lstm (nn.LSTM): Bidirectional LSTM layer.
        dropout (nn.Dropout): Dropout layer.
        fc (nn.Linear): Linear layer to project concatenated hidden states.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int, dropout: float, output_dim: int):
        super(BiLSTMEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout

        # Initialize layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, #input shape (batch_size,sequence_length, embedding_dim by default shape of lstm is sequence_length first.)
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0.0  # Dropout only if multiple layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  #as we are uing bidirectional LSTM, hidden_dim *2 

        # Initialize weights for stability  
        nn.init.xavier_uniform_(self.embedding.weight)  #xavier initialization for embedding.
        for name, param in self.lstm.named_parameters(): #xavier initialization for LSTM weights
            #name parameters means all the parameters of LSTM gates.
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the forward pass of the encoder.

        Args:
            x (torch.Tensor): Input token IDs, shape (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - output: LSTM outputs, shape (batch_size, sequence_length, hidden_dim * 2).
                - hidden: Final hidden states, shape (num_layers * 2, batch_size, hidden_dim).
                - cell: Final cell states, shape (num_layers * 2, batch_size, hidden_dim).

        Raises:
            ValueError: If input tensor shape or dimensions are invalid.
        """
        # Validate input
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (batch_size, sequence_length), got {x.shape}")
        if not torch.all(x >= 0) or not torch.all(x < self.vocab_size):
            raise ValueError(f"Input token IDs must be in [0, {self.vocab_size}), got min {x.min()}, max {x.max()}")

        # Embed input: (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(x)

        # Apply LSTM: (batch_size, sequence_length, embedding_dim) -> 
        # (batch_size, sequence_length, hidden_dim * 2), (num_layers * 2, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(embedded)

        # Concatenate final forward and backward hidden states: (batch_size, hidden_dim * 2)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # Apply dropout and linear layer: (batch_size, hidden_dim * 2) -> (batch_size, output_dim)
        dropout = self.dropout(final_hidden)
        out = self.fc(dropout)

        return out, hidden, cell
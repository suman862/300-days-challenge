import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AdditiveAttention(nn.Module):
    """Implements Bahdanau-style additive attention for sequence-to-sequence models.

    Computes attention scores between the decoder's hidden state and encoder outputs,
    producing a context vector and attention weights for use in decoding.

    Args:
        encoder_hidden_dim (int): Dimension of the encoder's hidden states.
        decoder_hidden_dim (int): Dimension of the decoder's hidden states.
        attention_dim (int): Dimension of the attention mechanism's hidden layer.

    Attributes:
        encoder_attn (nn.Linear): Linear layer to project encoder outputs.
        decoder_attn (nn.Linear): Linear layer to project decoder hidden state.
        v (nn.Parameter): Parameter vector to compute attention scores.
    """
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int):
        super(AdditiveAttention, self).__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_dim = attention_dim

        # Linear layers to project encoder and decoder states
        self.encoder_attn = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        self.decoder_attn = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        
        # Attention score parameter, initialized with Glorot initialization for stability
        self.v = nn.Parameter(torch.empty(attention_dim))
        nn.init.xavier_uniform_(self.v.unsqueeze(0))  # Shape: (1, attention_dim)

    def forward(self, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the attention context vector and weights.

        Args:
            encoder_outputs (torch.Tensor): Encoder hidden states, shape (batch_size, src_len, encoder_hidden_dim).
            decoder_hidden (torch.Tensor): Decoder hidden state, shape (batch_size, decoder_hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - context: Context vector, shape (batch_size, encoder_hidden_dim).
                - attn_weights: Attention weights, shape (batch_size, src_len).

        Raises:
            ValueError: If input tensor shapes or dimensions do not match expected values.
        """
        # Validate input shapes
        if encoder_outputs.dim() != 3:
            raise ValueError(
                f"Expected encoder_outputs to be 3D (batch_size, src_len, encoder_hidden_dim), got {encoder_outputs.shape}"
            )
        if decoder_hidden.dim() != 2:
            raise ValueError(
                f"Expected decoder_hidden to be 2D (batch_size, decoder_hidden_dim), got {decoder_hidden.shape}"
            )

        batch_size, src_len, enc_dim = encoder_outputs.size()
        if enc_dim != self.encoder_hidden_dim:
            raise ValueError(f"Encoder hidden dimension mismatch: expected {self.encoder_hidden_dim}, got {enc_dim}")
        if decoder_hidden.size(1) != self.decoder_hidden_dim:
            raise ValueError(f"Decoder hidden dimension mismatch: expected {self.decoder_hidden_dim}, got {decoder_hidden.size(1)}")
        if batch_size != decoder_hidden.size(0):
            raise ValueError(f"Batch size mismatch: encoder_outputs {batch_size}, decoder_hidden {decoder_hidden.size(0)}")

        # Repeat decoder hidden state to match source length: (batch_size, decoder_hidden_dim) -> (batch_size, src_len, decoder_hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Compute energy: (batch_size, src_len, encoder_hidden_dim) -> (batch_size, src_len, attention_dim)
        #                + (batch_size, src_len, decoder_hidden_dim) -> (batch_size, src_len, attention_dim)
        energy = torch.tanh(self.encoder_attn(encoder_outputs) + self.decoder_attn(decoder_hidden))

        # Compute attention scores: (batch_size, src_len, attention_dim) @ (attention_dim,) -> (batch_size, src_len)
        attention_scores = torch.matmul(energy, self.v)

        # Apply softmax to get attention weights: (batch_size, src_len)
        attn_weights = F.softmax(attention_scores, dim=1)

        # Compute context vector: (batch_size, 1, src_len) @ (batch_size, src_len, encoder_hidden_dim) -> (batch_size, 1, encoder_hidden_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights
import torch
import torch.nn as nn
import torch.nn
import torch.nn.functional as F

import math
import time

from Config import Config
from Vocabulary import decode
from Utils import generate_masks

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=Config.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=Config.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=Config.device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, vocab_size, n_embed, n_head, n_ff, n_layers, dropout):
        
        super().__init__()
        
        self.transformer = nn.Transformer(d_model=n_embed, nhead=n_head, num_decoder_layers=n_layers, num_encoder_layers=n_layers, dim_feedforward=n_ff, dropout=dropout, batch_first=True)
        self.pos_encoder = PositionalEncoding(n_embed, dropout)

        self.src_input_emb = nn.Embedding(vocab_size, n_embed)
        self.tgt_input_emb = nn.Embedding(vocab_size, n_embed)
        self.n_embed = n_embed
        
        self.ln = nn.Linear(n_embed, vocab_size)


    def forward(self, src, tgt, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask, mem_padding_mask): 
        src = self.src_input_emb(src) * math.sqrt(self.n_embed)
        tgt = self.tgt_input_emb(tgt) * math.sqrt(self.n_embed)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # forward(src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, src_is_causal=None, tgt_is_causal=None, memory_is_causal=False)
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=None, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask, src_is_causal=None, tgt_is_causal=True)
        output = self.ln(output)
        
        return output
    
    
class MTModel(nn.Module):
    
    def __init__(self, config: Config):
        
        super().__init__()
        
        self.config = config
        self.transformer = TransformerModel(config.vocab_size, config.n_embd, config.n_head, config.n_embd * 4, config.n_layer, config.dropout)
        
    def forward(self, src, tgt, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask, mem_padding_mask, targets = None):
        B, T = tgt.shape
        if tgt_mask == None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(Config.device) != 0

        logits = self.transformer(src, tgt, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask, mem_padding_mask)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            outs = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(outs, targets, ignore_index=self.config.PAD_TOKEN)

        return logits, loss
    
    def _sample_top_p_(self, probs: torch.tensor) -> torch.tensor:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = (probs_sum - probs_sort) > Config.p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        idx_next = torch.multinomial(probs_sort, num_samples=1)
        idx_next = torch.gather(probs_idx, -1, idx_next)

        return idx_next
    
    def generate(self, src, tgt, max_new_tokens):
 
        for _ in range(max_new_tokens):
            tgt = tgt[:, -self.config.block_size:]
            # Use torch.nn.functional.pad for padding
            tgt_pad = F.pad(tgt, (0, self.config.block_size - tgt.size(1)), value=self.config.PAD_TOKEN)
            
            # Ensure masks are generated correctly
            src_m, tgt_m, ca_m = generate_masks(src, tgt_pad)
            # Get the predictions; src, tgt, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask, mem_padding_mask, targets = None
            logits, _ = self(src, tgt_pad, None, src_m, None, tgt_m, src_m)
            logits = logits[:, tgt.size(1) - 1, :]  # Focus on the last time step


            probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities


            # Sample from the distribution
            tgt_next = self._sample_top_p_(probs)
            
            # Append sampled index to the running sequence
            tgt = torch.cat((tgt, tgt_next), dim=1)
            
            # Check for end token
            if (tgt_next == Config.END_TOK).all():
                break

        return tgt
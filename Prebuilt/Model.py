import torch
import torch.nn as nn
import torch.nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import math
import os

from Config import Config
from Blocks import TransformerDecoderLayer
from Vocabulary import decode
from Utils import generate_masks

class MultiheadAttentionWithWeights(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, *args, **kwargs):
        vargs = dict(kwargs, need_weights=True)
        output, weights = super().forward(*args, **vargs)
        self.attn_weights = weights
        return output, weights

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math::
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

        pe = torch.zeros(max_len, d_model, device=Config.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=Config.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=Config.device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        # x shape: [batch size, sequence length, embed dim]
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0)
        x = x + pe
        return x



class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def _detect_is_causal_mask(
        self,
        mask,
        is_causal,
        size,
    ) -> bool:
        """Return whether the given attention mask is causal.

        Warning:
        If ``is_causal`` is not ``None``, its value will be returned as is.  If a
        user supplies an incorrect ``is_causal`` hint,

        ``is_causal=False`` when the mask is in fact a causal attention.mask
        may lead to reduced performance relative to what would be achievable
        with ``is_causal=True``;
        ``is_causal=True`` when the mask is in fact not a causal attention.mask
        may lead to incorrect and unpredictable execution - in some scenarios,
        a causal mask may be applied based on the hint, in other execution
        scenarios the specified mask may be used.  The choice may not appear
        to be deterministic, in that a number of factors like alignment,
        hardware SKU, etc influence the decision whether to use a mask or
        rely on the hint.
        ``size`` if not None, check whether the mask is a causal mask of the provided size
        Otherwise, checks for any causal mask.
        """
        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None and mask is not None:
            sz = size if size is not None else mask.size(-2)
            causal_comparison = nn.Transformer.generate_square_subsequent_mask(sz, device=mask.device, dtype=mask.dtype)

            # Do not use `torch.equal` so we handle batched masks by
            # broadcasting the comparison.
            if mask.size() == causal_comparison.size():
                make_causal = bool((mask == causal_comparison).all())
            else:
                make_causal = False

        return make_causal

    def __init__(self, vocab_size, n_embed, n_head, n_ff, n_layers, dropout):
        
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embed, nhead=n_head, dim_feedforward=n_ff, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=n_embed, nhead=n_head, dim_feedforward=n_ff, dropout=dropout, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        
        self.pos_encoder = PositionalEncoding(n_embed, dropout)

        self.src_input_emb = nn.Embedding(vocab_size, n_embed)
        self.tgt_input_emb = nn.Embedding(vocab_size, n_embed)
        self.n_embed = n_embed
        
        self.ln = nn.Linear(n_embed, vocab_size)


    def forward(self, src, tgt, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask, mem_padding_mask, memory_mask=None): 
        src = self.src_input_emb(src) * math.sqrt(self.n_embed)
        tgt = self.tgt_input_emb(tgt) * math.sqrt(self.n_embed)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        memory = self.encoder(src=src, mask=src_mask, src_key_padding_mask=src_padding_mask)
        # forward(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=False)
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=None, memory_mask=memory_mask)
        output = self.ln(output)
        
        return output
    
    
class MTModel(nn.Module):
    
    def __init__(self, config: Config):
        
        super().__init__()
        
        self.config = config
        self.transformer = TransformerModel(config.vocab_size, config.n_embd, config.n_head, config.n_embd * 4, config.n_layer, config.dropout)
        
    def forward(self, src, tgt, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask, mem_padding_mask, memory_mask=None, targets = None):
        train_tgt = tgt
        
        B, T = train_tgt.shape
        if tgt_mask == None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=Config.device, dtype=bool)

        logits = self.transformer(src, train_tgt, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask, mem_padding_mask, memory_mask=memory_mask)
        
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
 
        for i in range(max_new_tokens):
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

    def visualize_attention(self, src, tgt, src_m, tgt_m, ca_m, output_dir='attention_plots', filename='attention_plot'):
        self.eval()
        
        with torch.no_grad():
            # src, tgt, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask, mem_padding_mask, targets = None
            output = self(src, tgt, src_mask=None, tgt_mask=None, 
                                    src_padding_mask=src_m, tgt_padding_mask=tgt_m, 
                                    mem_padding_mask=src_m)
        
        # Extract attention weights from the last decoder layer
        cross_attn_weights = self.transformer.decoder.layers[0].multihead_attn.attn_weights
        d_attn_weights = self.transformer.decoder.layers[0].attn_weights
        d_attn_mask = self.transformer.decoder.layers[0].mask

        print('Weights: ',d_attn_weights[:5, :5])
        print('Mask: ',d_attn_mask[:5, :5])
        # e_attn_weights = self.transformer.transformer.encoder.layers[-1].self_attn.attn_weights

        # print(cross_attn_weights, d_attn_weights, e_attn_weights)
        
        if d_attn_weights is None or cross_attn_weights is None:
            print("No attention weights captured. Ensure you're using the custom MultiheadAttention class.")
            return
        
        cross_attn_weights = cross_attn_weights.squeeze(0).cpu().numpy()[:150, :150]
        d_attn_weights = d_attn_weights.squeeze(0).cpu().numpy()[:150, :150]
        # e_attn_weights = e_attn_weights.squeeze(0).cpu().numpy()[:150, :150]
        
        src_tokens = list(range(d_attn_weights.shape[1]))[:150]
        tgt_tokens = list(range(d_attn_weights.shape[1]))[:150]
        
        def plot_and_save(weights, name):
            plt.figure(figsize=(12, 8))
            sns.heatmap(weights, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap='viridis')
            
            plt.title('Attention Weights Visualization')
            plt.xlabel('Source Tokens')
            plt.ylabel('Target Tokens')
            plt.tight_layout()
            
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f'{filename}-{name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        plot_and_save(cross_attn_weights, 'cross')
        plot_and_save(d_attn_weights, 'decoder')
        
        print(f"Attention plot saved to {output_dir}")
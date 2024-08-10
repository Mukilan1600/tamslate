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
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.src_input_emb.weight, -initrange, initrange)
        nn.init.uniform_(self.tgt_input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.ln.bias)
        nn.init.uniform_(self.ln.weight, -initrange, initrange)
        


    def forward(self, src, tgt, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask, mem_padding_mask, memory_mask=None): 
        
        memory = self._encode(src, src_mask, src_padding_mask)
        output = self._decode(tgt, tgt_mask, tgt_padding_mask, memory_mask, mem_padding_mask, memory)
        
        return output
    
    def _encode(self, src, src_mask, src_padding_mask):
        src = self.src_input_emb(src) * math.sqrt(self.n_embed)
        src = self.pos_encoder(src)
        memory = self.encoder(src=src, mask=src_mask, src_key_padding_mask=src_padding_mask)
        
        return memory
    
    def _decode(self, tgt, tgt_mask, tgt_padding_mask, memory_mask, mem_padding_mask, memory):
        tgt = self.tgt_input_emb(tgt) * math.sqrt(self.n_embed)        
        tgt = self.pos_encoder(tgt)
        
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=mem_padding_mask, memory_mask=memory_mask)
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
    
    def generate(self, src, tgt, max_new_tokens, beam_size=1):
        batch_size = src.size(0)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.config.block_size, device=Config.device, dtype=bool)
        src_m, _, __ = generate_masks(src, tgt)
        memory = self.transformer._encode(src, None, src_m)

        # Initialize the beam
        beam = [{'sequence': tgt, 'score': 0.0}] * batch_size

        for i in range(max_new_tokens):
            new_beam = []
            for b in beam:
                sequence = b['sequence']
                score = b['score']

                sequence = sequence[:, -self.config.block_size:]
                sequence_pad = F.pad(sequence, (0, self.config.block_size - sequence.size(1)), value=self.config.PAD_TOKEN)

                # Ensure masks are generated correctly
                src_m, tgt_m, ca_m = generate_masks(src, sequence_pad)
                logits = self.transformer._decode(sequence_pad, tgt_mask, tgt_m, None, src_m, memory)
                logits = logits[:, sequence.size(1) - 1, :]

                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, beam_size, dim=-1)

                for j in range(beam_size):
                    new_sequence = torch.cat((sequence, top_indices[:, j].unsqueeze(1)), dim=1)
                    new_score = score + torch.log(top_probs[:, j])
                    new_beam.append({'sequence': new_sequence, 'score': new_score})

            # Sort the beam and keep the top beam_size sequences
            new_beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)
            beam = new_beam[:beam_size]

            # Check if the best sequence has reached the end token
            if beam[0]['sequence'][:, -1].item() == Config.END_TOK:
                break

        # Return the sequence with the highest score
        return beam[0]['sequence']


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
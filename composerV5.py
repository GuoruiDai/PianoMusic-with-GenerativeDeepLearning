import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from model_utils import ComposerConfig


class FFN(nn.Module):
    def __init__(self, config: ComposerConfig):
        super().__init__()
        self.w1 = nn.Sequential(
            nn.RMSNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim * 4)  # Gate "value" projection
        )
        self.w2 = nn.Linear(config.embed_dim, config.embed_dim * 4)  # Gate "modulation" projection
        self.w3 = nn.Linear(config.embed_dim * 4, config.embed_dim)  # Output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x)) 


class FlashAttention(nn.Module):
    def __init__(self, config: ComposerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_attn_head
        self.head_dim = self.embed_dim // self.n_heads
        
        self.w_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.w_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.w_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        self.register_buffer('k_cache',
            torch.zeros(config.batch_size, config.training_context_len, self.n_heads, self.head_dim),
            persistent=False)
        self.register_buffer('v_cache',
            torch.zeros(config.batch_size, config.training_context_len, self.n_heads, self.head_dim),
            persistent=False)

    def _apply_rotary_emb(self, x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor], pos_idx: int):
        dtype = x.dtype
        batch_size, seq_len, n_heads, head_dim = x.shape
        freqs_cos, freqs_sin = freqs
        
        if self.training:
            freqs_cos_slice = freqs_cos[:seq_len]
            freqs_sin_slice = freqs_sin[:seq_len]
        else:
            freqs_cos_slice = freqs_cos[pos_idx:pos_idx+seq_len]
            freqs_sin_slice = freqs_sin[pos_idx:pos_idx+seq_len]

        # Reshape x to separate even and odd dimensions
        x_reshape = x.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
        x_even = x_reshape[..., 0]  # [batch, seq_len, n_heads, head_dim/2]
        x_odd = x_reshape[..., 1]   # [batch, seq_len, n_heads, head_dim/2]
        
        # Reshape freqs for broadcasting
        freqs_cos_slice = freqs_cos_slice.view(1, seq_len, 1, head_dim // 2)
        freqs_sin_slice = freqs_sin_slice.view(1, seq_len, 1, head_dim // 2)
        
        x_even_rotated = x_even * freqs_cos_slice - x_odd * freqs_sin_slice
        x_odd_rotated = x_even * freqs_sin_slice + x_odd * freqs_cos_slice
        
        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
        x_out = x_rotated.reshape(batch_size, seq_len, n_heads, head_dim)
        return x_out.to(dtype)
    
    def forward(self, x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor], pos_idx: int):
        batch_dim, seq_len, _ = x.shape
        
        q = self.w_q(x).view(batch_dim, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch_dim, seq_len, self.n_heads, self.head_dim)
        v = self.w_v(x).view(batch_dim, seq_len, self.n_heads, self.head_dim)
        
        q = self._apply_rotary_emb(q, freqs, pos_idx)
        k = self._apply_rotary_emb(k, freqs, pos_idx)
        
        if not self.training:
            self.k_cache[:, pos_idx:pos_idx+seq_len] = k  # add new k at pos
            self.v_cache[:, pos_idx:pos_idx+seq_len] = v
            k = self.k_cache[:, :pos_idx+seq_len]  # retrieve full past k
            v = self.v_cache[:, :pos_idx+seq_len]
        
        multi_head_q = q.permute(0, 2, 1, 3)  # (batch, heads, seq, dim)
        multi_head_k = k.permute(0, 2, 1, 3)
        multi_head_v = v.permute(0, 2, 1, 3)
        
        x = F.scaled_dot_product_attention(
            multi_head_q, multi_head_k, multi_head_v,
            is_causal = True if self.training else False,  #^ Must be set to False for kv cache inference
        ).permute(0, 2, 1, 3).reshape(batch_dim, seq_len, self.embed_dim)
        return self.out_proj(x)


class InputEmbedding(nn.Module):
    def __init__(self, config: ComposerConfig):
        super().__init__()
        self.pitch_id_embed = nn.Embedding(88, embedding_dim=config.embed_dim // 2)
        self.rel_time_id_embed = nn.Embedding(26, embedding_dim=config.embed_dim // 2)

    def forward(self, pitch_rel_time_id_tensor: torch.Tensor) -> torch.Tensor:
        pitch_id, rel_time_id = torch.split(pitch_rel_time_id_tensor.to(torch.long), [1, 1], dim=-1)
        pitch_id_embed = self.pitch_id_embed(pitch_id.view(-1) - 1)  #? Shift pitch to 0-index
        rel_time_id_embed = self.rel_time_id_embed(rel_time_id.view(-1))
        return torch.cat([pitch_id_embed, rel_time_id_embed], dim=-1)


class DecoderBlock(nn.Module):
    def __init__(self, config: ComposerConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.embed_dim)
        self.ffn_norm = nn.RMSNorm(config.embed_dim)
        self.self_attention = FlashAttention(config)
        self.feedforward = FFN(config)

    def forward(self, x, freqs: tuple[torch.Tensor, torch.Tensor], note_index):
        x = x + self.self_attention(self.attn_norm(x), freqs, note_index)
        x = x + self.feedforward(self.ffn_norm(x))
        return x


class DecoderStack(nn.Module):
    def __init__(self, config: ComposerConfig):
        super(DecoderStack, self).__init__()
        self.config = config
        self.final_norm = nn.RMSNorm(config.embed_dim)
        self.decoder_block_stack = nn.ModuleList([copy.deepcopy(DecoderBlock(config)) for _ in range(self.config.decoder_depth)])
        self.register_buffer('freqs_cos', None)
        self.register_buffer('freqs_sin', None)
        self._precompute_freqs_sin_cos(config)
        
    def _precompute_freqs_sin_cos(self, config: ComposerConfig):
        head_dim = config.embed_dim // config.n_attn_head
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        seq_idx = torch.arange(config.training_context_len, dtype=torch.float32)
        
        # Outer product creates a matrix of shape [seq_len, head_dim/2]
        freqs = torch.outer(seq_idx, theta)
        freqs_cos = torch.cos(freqs)  # [seq_len, head_dim/2]
        freqs_sin = torch.sin(freqs)  # [seq_len, head_dim/2]
        
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)
    
    def forward(self, x: torch.Tensor, note_index):
        x = x.unsqueeze(0)  # add batch dim for attention layer
        for decoder_block in self.decoder_block_stack:
            x = decoder_block(x, (self.freqs_cos, self.freqs_sin), note_index)
        return self.final_norm(x.squeeze(0))


class OutputBlock(nn.Module):
    def __init__(self, config: ComposerConfig):
        super(OutputBlock, self).__init__()
        self.pred_pitch_logits = nn.Linear(config.embed_dim, 89)
        self.pred_rel_time_logits = nn.Linear(config.embed_dim, 26)

    def forward(self, x: torch.Tensor):
        return self.pred_pitch_logits(x), self.pred_rel_time_logits(x)


class Composer(nn.Module):
    def __init__(self, config: ComposerConfig):
        super(Composer, self).__init__()
        self.input_embedding = InputEmbedding(config)
        self.decoder_stack = DecoderStack(config)
        self.output_block = OutputBlock(config)

    def forward(self, token_tensor: torch.Tensor, note_index: torch.Tensor = torch.zeros((1))):
        #? note_index is only for inference for the kv_cache_tensor retrival and update; inputs need to be tensor for executorch compatibility
        x = self.input_embedding(token_tensor)
        x = self.decoder_stack(x, note_index.long().item())
        return self.output_block(x)

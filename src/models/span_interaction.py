import torch
import torch.nn as nn
import torch.nn.functional as F

class SpanInteraction(nn.Module):
    """
    Generates (B, L, T) span masks [start - k, end + k], ANDed with token and frame masks.
    Pools visual and acoustic frames per word, concatenates them with text, projects, 
    pools over the sequence (L), and outputs a zero-initialized vector S.
    """
    def __init__(
        self,d_model: int = 128,k_padding: int = 2,use_span_qkv: bool = False,use_attentive_pool: bool = True):
        super().__init__()
        self.d_model = d_model
        self.k = k_padding
        self.use_span_qkv = use_span_qkv
        self.use_attentive_pool = use_attentive_pool

        # Optional QKV cross-attention over spans
        if self.use_span_qkv:
            self.v_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            self.a_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        # H_t (D) + V_word (D) + A_word (D) + span_len_v (1) + span_len_a (1) + valid_flag (1)
        concat_dim = (3 * d_model) + 3

        self.proj = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        if self.use_attentive_pool:
            self.word_attn_scorer = nn.Linear(d_model, 1)

        # Zero-initialized output projection ensures Z = P + 0 * S at initialization
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _get_span_mask(
        self, 
        alignment: torch.Tensor, 
        t_max: int, 
        text_mask: torch.Tensor, 
        frame_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Creates (B, L, T) boolean span mask.
        """
        B, L, _ = alignment.shape
        t_grid = torch.arange(t_max, device=alignment.device).view(1, 1, t_max) # (1, 1, T)

        # Apply k-padding and clamp boundaries
        starts = (alignment[:, :, 0:1] - self.k).clamp(min=0)       # (B, L, 1)
        ends = (alignment[:, :, 1:2] + self.k).clamp(max=t_max - 1) # (B, L, 1)

        # Broadcast comparison -> (B, L, T)
        in_span = (t_grid >= starts) & (t_grid <= ends)

        # AND with availability masks
        # text_mask: (B, L) -> (B, L, 1)
        # frame_mask: (B, T) -> (B, 1, T)
        span_mask = in_span & text_mask.unsqueeze(-1) & frame_mask.unsqueeze(1)
        return span_mask

    def _pool_spans(
        self, 
        H_m: torch.Tensor, 
        H_t: torch.Tensor, 
        span_mask: torch.Tensor, 
        attn_layer: nn.Module = None
    ):
        """
        Extracts modality frames aligned to words.
        Returns pooled word representation (B, L, D) and span lengths (B, L, 1).
        """
        B, L, T = span_mask.shape
        span_len = span_mask.sum(dim=-1, keepdim=True) # (B, L, 1)
        safe_len = torch.clamp(span_len, min=1.0)
        span_mask_f = span_mask.unsqueeze(-1).to(H_m.dtype) # (B, L, T, 1)

        if self.use_span_qkv and attn_layer is not None:
            # Mask preparation for nn.MultiheadAttention (True = ignored/masked)
            key_padding_mask = ~span_mask.view(B * L, T)
            
            # Reshape for independent sequence attention per word
            # Queries: (B*L, 1, D)
            q = H_t.view(B * L, 1, self.d_model)
            # Keys/Values: (B, T, D) -> broadcast to (B*L, T, D)
            kv = H_m.unsqueeze(1).expand(B, L, T, self.d_model).reshape(B * L, T, self.d_model)

            # Check if entire sequence is masked to prevent NaN in MultiheadAttention
            is_empty = key_padding_mask.all(dim=-1) # (B*L)
            key_padding_mask[is_empty, 0] = False   # unmask first step temporarily 
            
            out, _ = attn_layer(q, kv, kv, key_padding_mask=key_padding_mask)
            
            # Re-zero the empty rows
            out = out.view(B, L, self.d_model)
            has_valid_frames = (span_len > 0).to(H_m.dtype)
            M_word = out * has_valid_frames
        else:
            # Default: span_avg
            # (B, 1, T, D) * (B, L, T, 1) -> sum over T -> (B, L, D)
            H_m_exp = H_m.unsqueeze(1)
            M_word = (H_m_exp * span_mask_f).sum(dim=2) / safe_len

        return M_word, span_len.float()

    def forward(
        self,
        H_v: torch.Tensor,
        H_a: torch.Tensor,
        H_t: torch.Tensor,
        alignment_indices: torch.Tensor,
        visual_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = H_t.shape
        T = H_v.shape[1]

        # Enforce Block 4 text dropout on the token mask immediately
        t_active_1d = active_mask[:, 2]                          # (B,)
        effective_text_mask = text_mask & t_active_1d.unsqueeze(1)  # (B, L)

        # Generate 3D span masks using effective_text_mask
        span_mask_v = self._get_span_mask(alignment_indices, T, effective_text_mask, visual_mask)
        span_mask_a = self._get_span_mask(alignment_indices, T, effective_text_mask, audio_mask)

        # Pool visual and acoustic features per word
        V_word, len_v = self._pool_spans(H_v, H_t, span_mask_v, getattr(self, "v_attn", None))
        A_word, len_a = self._pool_spans(H_a, H_t, span_mask_a, getattr(self, "a_attn", None))

        # Apply Block 4 modality dropout for V and A + normalize span lengths to [0, 1]
        v_active = active_mask[:, 0].view(B, 1, 1).to(H_t.dtype)
        a_active = active_mask[:, 1].view(B, 1, 1).to(H_t.dtype)
        t_scale = max(1, T)

        V_word = V_word * v_active
        A_word = A_word * a_active
        norm_len_v = (len_v / t_scale) * v_active
        norm_len_a = (len_a / t_scale) * a_active

        # Valid alignment flag (1 if token has valid visual OR audio frames)
        valid_flag = ((norm_len_v > 0) | (norm_len_a > 0)).to(H_t.dtype)

        # Concatenate and project: (B, L, 3D + 3) -> (B, L, D)
        combined = torch.cat([H_t, V_word, A_word, norm_len_v, norm_len_a, valid_flag], dim=-1)
        S_words = self.proj(combined) * effective_text_mask.unsqueeze(-1).to(H_t.dtype)

        # Sequence pooling over L -> (B, D)
        has_text = effective_text_mask.any(dim=-1, keepdim=True)  # (B, 1) bool

        if self.use_attentive_pool:
            logits = self.word_attn_scorer(S_words).squeeze(-1)  # (B, L)
            logits = logits.masked_fill(~effective_text_mask, float("-inf"))

            safe_logits = torch.where(has_text, logits, torch.zeros_like(logits))
            attn_weights = F.softmax(safe_logits, dim=-1) * effective_text_mask.to(H_t.dtype)
            S_pooled = (S_words * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            neg_inf = torch.finfo(S_words.dtype).min / 2.0
            masked_for_max = S_words.masked_fill(~effective_text_mask.unsqueeze(-1), neg_inf)
            S_pooled = masked_for_max.max(dim=1).values * has_text.to(H_t.dtype)

        # Zero-init output projection + hard mask on Text dropout (eliminates bias leakage)
        t_active_out = t_active_1d.view(B, 1).to(H_t.dtype)
        S_out = self.out_proj(S_pooled) * t_active_out

        return S_out
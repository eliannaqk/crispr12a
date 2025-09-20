import math
from typing import Optional

import torch
from torch import nn
from einops import rearrange, repeat


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000**(torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len),
                                inv_freq).to(x.device).float()
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(
        -2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[None, offset:x.shape[1] + offset, None, :].
        repeat_interleave(2, 3), sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


def apply_rope(query, key, rotary_dim=None, offset=0):
    seq_len = key.shape[1]
    if rotary_dim is not None:
        k_rot = key[:, :, :, :rotary_dim]
        k_pass = key[:, :, :, rotary_dim:]

        q_rot = query[:, :, :, :rotary_dim]
        q_pass = query[:, :, :, rotary_dim:]

        sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
        k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
        q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

        key = torch.cat([k_rot, k_pass], dim=-1)
        query = torch.cat([q_rot, q_pass], dim=-1)
    else:
        sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
        key = apply_rotary_pos_emb(key, sincos, offset=offset)
        query = apply_rotary_pos_emb(query, sincos, offset=offset)

    return query, key


class Transition(nn.Module):

    def __init__(self, d):
        super(Transition, self).__init__()

        self.d = d

        self.linear_1 = nn.Linear(self.d, self.d)
        self.linear_2 = nn.Linear(self.d, self.d)
        self.linear_3 = nn.Linear(self.d, self.d)

        # truncated normal initialization
        nn.init.normal_(self.linear_1.weight, std=2)
        nn.init.normal_(self.linear_2.weight, std=2)
        # fill weights of last layer with zeros
        nn.init.zeros_(self.linear_3.weight)

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class AttentionLayer(nn.Module):

    def __init__(
        self,
        d_q: int,
        d_kv: int,
        d_hidden: int,
        d_out: int,
        n_heads: int,
        causal: bool = False,
        use_rope: bool = False,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            d_hidden:
                Hidden channel dimension
            n_heads:
                Number of attention heads
        """
        super(AttentionLayer, self).__init__()

        self.d_q = d_q
        self.d_kv = d_kv
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.n_heads = n_heads
        self.causal = causal
        self.use_rope = use_rope
        self.inf = 1e6

        self.linear_q = nn.Linear(d_q, d_hidden * n_heads)
        self.linear_kv = nn.Linear(d_kv, 2 * d_hidden * n_heads)
        self.linear_out = nn.Linear(d_hidden * n_heads, d_out)

        self.softmax = nn.Softmax(dim=-1)

        # fill weights of last layer with zeros
        nn.init.zeros_(self.linear_out.weight)

    def forward(
        self,
        s_q: torch.Tensor,
        s_kv: torch.Tensor,
        mask_q: torch.Tensor,
        mask_kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * d_hidden]
        q = self.linear_q(s_q)
        kv = self.linear_kv(s_kv)

        # [*, N_res, H, d_hidden]
        q = q.view(q.shape[:-1] + (self.n_heads, -1))

        # [*, N_res, H, 2 * d_hidden]
        kv = kv.view(kv.shape[:-1] + (self.n_heads, -1))

        # [*, N_res, H, d_hidden]
        k, v = torch.split(kv, self.d_hidden, dim=-1)

        if self.use_rope:
            rotary_dim = self.d_hidden // 2
            q, k = apply_rope(q, k, rotary_dim=rotary_dim)

        ##########################
        # Compute attention scores
        ##########################
        # [*, H, N_res, N_res]
        a = torch.matmul(
            q.permute(0, 2, 1, 3),  # [*, H, N_res, C_hidden]
            k.permute(0, 2, 3, 1),  # [*, H, C_hidden, N_res]
        )

        if self.causal:
            # [N_res, N_res]
            causal_mask = torch.triu(
                torch.ones(a.shape[-2:], dtype=a.dtype, device=a.device),
                diagonal=1,
            )
            # [*, N_res, N_res]
            causal_mask = causal_mask.unsqueeze(0).expand(a.shape[:-2] +
                                                          (-1, -1))
            a = a - (causal_mask * self.inf)

        a *= math.sqrt(1.0 / self.d_hidden)

        square_mask = mask_q.unsqueeze(-1) * mask_kv.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        if self.causal:
            _ = 0

        ################
        # Compute output
        ################
        # [*, N_res, H, d_hidden]
        o = torch.matmul(a,
                         v.transpose(-2,
                                     -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * d_hidden]
        # o = flatten_final_dims(o, 2)
        o = rearrange(o, "b n h c -> b n (h c)", h=self.n_heads)

        # [*, N_res, C_s]
        s = self.linear_out(o)

        return s


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_s: int,
        d_hidden: int,
        n_heads: int,
        dropout_rate: float = 0.1,
        use_rope: bool = False,
    ) -> None:
        super(EncoderLayer, self).__init__()

        self.attention = AttentionLayer(
            d_q=d_s,
            d_kv=d_s,
            d_hidden=d_hidden,
            d_out=d_s,
            n_heads=n_heads,
            use_rope=use_rope,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_s)

        self.transition = Transition(d=d_s)

    def forward(
        self,
        s: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        upd = self.attention(
            s_q=s,
            s_kv=s,
            mask_q=mask,
            mask_kv=mask,
        )
        s = s + self.dropout(upd)
        s = self.layer_norm(s)

        s = self.transition(s)

        return s


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_s: int,
        d_hidden: int,
        n_heads: int,
        dropout_rate: float = 0.1,
        use_rope: bool = False,
    ) -> None:
        super(DecoderLayer, self).__init__()

        self.attention = AttentionLayer(
            d_q=d_s,
            d_kv=d_s,
            d_hidden=d_hidden,
            d_out=d_s,
            n_heads=n_heads,
            causal=True,
            use_rope=use_rope,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_s)

        self.transition = Transition(d=d_s)

    def forward(
        self,
        s: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        upd = self.attention(
            s_q=s,
            s_kv=s,
            mask_q=mask,
            mask_kv=mask,
        )
        s = s + self.dropout(upd)
        s = self.layer_norm(s)

        s = self.transition(s)

        return s


class CrossDecoderLayer(nn.Module):

    def __init__(
        self,
        d_s_self: int,
        d_s_cross: int,
        d_hidden: int,
        n_heads: int,
        dropout_rate: float = 0.1,
        use_self_rope: bool = False,
        use_cross_rope: bool = False,
    ) -> None:
        super(CrossDecoderLayer, self).__init__()

        self.self_attention = AttentionLayer(
            d_q=d_s_self,
            d_kv=d_s_self,
            d_hidden=d_hidden,
            d_out=d_s_self,
            n_heads=n_heads,
            causal=True,
            use_rope=use_self_rope,
        )
        self.self_dropout = nn.Dropout(dropout_rate)
        self.self_layer_norm = nn.LayerNorm(d_s_self)

        self.cross_attention = AttentionLayer(
            d_q=d_s_self,
            d_kv=d_s_cross,
            d_hidden=d_hidden,
            d_out=d_s_self,
            n_heads=n_heads,
            use_rope=use_cross_rope,
        )
        self.cross_dropout = nn.Dropout(dropout_rate)
        self.cross_layer_norm = nn.LayerNorm(d_s_self)

        self.transition = Transition(d=d_s_self)

    def forward(
        self,
        s_self: torch.Tensor,
        s_cross: torch.Tensor,
        mask_self: Optional[torch.Tensor] = None,
        mask_cross: Optional[torch.Tensor] = None,
    ):
        upd = self.self_attention(
            s_q=s_self,
            s_kv=s_self,
            mask_q=mask_self,
            mask_kv=mask_self,
        )
        s_self = s_self + self.self_dropout(upd)
        s_self = self.self_layer_norm(s_self)

        upd = self.cross_attention(
            s_q=s_self,
            s_kv=s_cross,
            mask_q=mask_self,
            mask_kv=mask_cross,
        )
        s_self = s_self + self.cross_dropout(upd)
        s_self = self.cross_layer_norm(s_self)

        s_self = self.transition(s_self)

        return s_self

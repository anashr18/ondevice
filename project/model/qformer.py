from __future__ import annotations
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


@dataclass
class QFormerConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    cross_attention_frequency: int = 2  # cross-attn at even layers
    encoder_hidden_size: int = 1024     # InternViT-300M output dim — DO NOT CHANGE
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0


class QFormerEmbeddings(nn.Module):
    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.word_embeddings     = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm           = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout             = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x    = self.word_embeddings(input_ids) + self.position_embeddings(pos)
        return self.dropout(self.LayerNorm(x))


class QFormerAttention(nn.Module):
    def __init__(self, config: QFormerConfig, is_cross_attention: bool = False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim  = config.hidden_size // config.num_attention_heads
        self.scale     = self.head_dim ** -0.5
        self.is_cross  = is_cross_attention

        kv_in = config.encoder_hidden_size if is_cross_attention else config.hidden_size
        self.q_proj   = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj   = nn.Linear(kv_in, config.hidden_size)
        self.v_proj   = nn.Linear(kv_in, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv_src = encoder_hidden_states if self.is_cross else hidden_states
        Q = self._split_heads(self.q_proj(hidden_states))
        K = self._split_heads(self.k_proj(kv_src))
        V = self._split_heads(self.v_proj(kv_src))

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = self.attn_drop(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, V)
        B, H, S, D = out.shape
        out = out.transpose(1, 2).contiguous().view(B, S, H * D)
        return self.out_proj(out)


class QFormerLayer(nn.Module):
    def __init__(self, config: QFormerConfig, layer_idx: int):
        super().__init__()
        self.attention  = QFormerAttention(config, is_cross_attention=False)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.has_cross_attention = (layer_idx % config.cross_attention_frequency == 0)
        if self.has_cross_attention:
            self.crossattention  = QFormerAttention(config, is_cross_attention=True)
            self.layernorm_cross = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        act = F.gelu
        H, I = config.hidden_size, config.intermediate_size

        self.ffn_q_intermediate = nn.Linear(H, I)
        self.ffn_q_output       = nn.Linear(I, H)
        self.layernorm_q        = nn.LayerNorm(H, eps=config.layer_norm_eps)

        self.ffn_t_intermediate = nn.Linear(H, I)
        self.ffn_t_output       = nn.Linear(I, H)
        self.layernorm_t        = nn.LayerNorm(H, eps=config.layer_norm_eps)

        self.act = act

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        query_length: int,
    ) -> torch.Tensor:
        # Step 1: self-attention over all tokens
        sa_out      = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = self.layernorm1(hidden_states + sa_out)

        # Step 2: cross-attention on query tokens only
        if self.has_cross_attention and encoder_hidden_states is not None:
            query_part = hidden_states[:, :query_length, :]
            ca_out     = self.crossattention(
                query_part,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            query_part    = self.layernorm_cross(query_part + ca_out)
            hidden_states = torch.cat([query_part, hidden_states[:, query_length:, :]], dim=1)

        # Step 3: split FFN for query vs text tokens
        q = hidden_states[:, :query_length, :]
        t = hidden_states[:, query_length:, :]
        q = self.layernorm_q(q + self.ffn_q_output(self.act(self.ffn_q_intermediate(q))))
        if t.shape[1] > 0:
            t = self.layernorm_t(t + self.ffn_t_output(self.act(self.ffn_t_intermediate(t))))

        return torch.cat([q, t], dim=1)


class QFormerEncoder(nn.Module):
    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.layers = nn.ModuleList([QFormerLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        query_length: int,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask,
                encoder_hidden_states, encoder_attention_mask,
                query_length,
            )
        return hidden_states


class QFormerModel(nn.Module):
    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.config     = config
        self.embeddings = QFormerEmbeddings(config)
        self.encoder    = QFormerEncoder(config)

    @classmethod
    def from_bert_pretrained(cls, config: QFormerConfig) -> "QFormerModel":
        model = cls(config)
        bert  = AutoModel.from_pretrained("bert-base-uncased")
        sd    = bert.state_dict()

        # Embeddings
        model.embeddings.word_embeddings.weight.data.copy_(sd["embeddings.word_embeddings.weight"])
        model.embeddings.position_embeddings.weight.data.copy_(sd["embeddings.position_embeddings.weight"])
        model.embeddings.LayerNorm.weight.data.copy_(sd["embeddings.LayerNorm.weight"])
        model.embeddings.LayerNorm.bias.data.copy_(sd["embeddings.LayerNorm.bias"])

        for i, layer in enumerate(model.encoder.layers):
            p = f"encoder.layer.{i}"

            # Self-attention weights
            for proj, bert_name in [("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value")]:
                getattr(layer.attention, proj).weight.data.copy_(sd[f"{p}.attention.self.{bert_name}.weight"])
                getattr(layer.attention, proj).bias.data.copy_(sd[f"{p}.attention.self.{bert_name}.bias"])
            layer.attention.out_proj.weight.data.copy_(sd[f"{p}.attention.output.dense.weight"])
            layer.attention.out_proj.bias.data.copy_(sd[f"{p}.attention.output.dense.bias"])
            layer.layernorm1.weight.data.copy_(sd[f"{p}.attention.output.LayerNorm.weight"])
            layer.layernorm1.bias.data.copy_(sd[f"{p}.attention.output.LayerNorm.bias"])

            # FFN — copy into both query and text FFNs
            inter_w = sd[f"{p}.intermediate.dense.weight"]
            inter_b = sd[f"{p}.intermediate.dense.bias"]
            out_w   = sd[f"{p}.output.dense.weight"]
            out_b   = sd[f"{p}.output.dense.bias"]
            ln_w    = sd[f"{p}.output.LayerNorm.weight"]
            ln_b    = sd[f"{p}.output.LayerNorm.bias"]

            for prefix in ("q", "t"):
                getattr(layer, f"ffn_{prefix}_intermediate").weight.data.copy_(inter_w)
                getattr(layer, f"ffn_{prefix}_intermediate").bias.data.copy_(inter_b)
                getattr(layer, f"ffn_{prefix}_output").weight.data.copy_(out_w)
                getattr(layer, f"ffn_{prefix}_output").bias.data.copy_(out_b)
                getattr(layer, f"layernorm_{prefix}").weight.data.copy_(ln_w)
                getattr(layer, f"layernorm_{prefix}").bias.data.copy_(ln_b)

        del bert
        return model

    def forward(
        self,
        query_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_length = query_embeds.shape[1]

        if input_ids is not None:
            text_embeds   = self.embeddings(input_ids)
            hidden_states = torch.cat([query_embeds, text_embeds], dim=1)
        else:
            hidden_states = query_embeds

        return self.encoder(
            hidden_states, attention_mask,
            encoder_hidden_states, encoder_attention_mask,
            query_length,
        )

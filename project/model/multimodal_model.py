import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .intern_vit import InternViTEncoder
from .qformer import QFormerConfig, QFormerModel
from .projector import LanguageProjection


class InternViTQFormerLFM(nn.Module):
    def __init__(
        self,
        intern_vit_path: str = "OpenGVLab/InternViT-300M-448px",
        lfm_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        num_query_tokens: int = 32,
        encoder_hidden_size: int = 1024,
        lfm_hidden_size: int = 896,
        lfm_top_n_unfreeze: int = 0,
    ):
        super().__init__()

        self.intern_vit   = InternViTEncoder(intern_vit_path)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, 768))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        qformer_config = QFormerConfig(encoder_hidden_size=encoder_hidden_size)
        self.qformer   = QFormerModel.from_bert_pretrained(qformer_config)

        self.language_projection = LanguageProjection(768, lfm_hidden_size)

        self.lfm = AutoModelForCausalLM.from_pretrained(
            lfm_path,
            torch_dtype=torch.bfloat16,
        )
        for param in self.lfm.parameters():
            param.requires_grad = False

        if lfm_top_n_unfreeze > 0:
            layers = self.lfm.model.layers
            for layer in layers[-lfm_top_n_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.num_query_tokens = num_query_tokens

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        n_tiles_list: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pixel_values: [B, max_n_tiles, 3, 448, 448]
        B = pixel_values.shape[0]

        # Gather real tiles, skip padding
        flat_tiles = []
        for b in range(B):
            n = n_tiles_list[b]
            flat_tiles.append(pixel_values[b, :n])  # [n, 3, 448, 448]
        flat_tiles = torch.cat(flat_tiles, dim=0).to(device, dtype=torch.bfloat16)

        # Forward through frozen ViT
        all_patch_feats = self.intern_vit(flat_tiles)  # [sum(n_tiles), 1024, 1024]

        # Split and flatten per image
        image_feats = []
        offset = 0
        for b in range(B):
            n = n_tiles_list[b]
            feats = all_patch_feats[offset:offset + n]  # [n, 1024, 1024]
            image_feats.append(feats.reshape(n * 1024, 1024))  # [n*1024, 1024]
            offset += n

        # Pad to N_max
        N_max = max(f.shape[0] for f in image_feats)
        img_feats_padded = torch.zeros(B, N_max, 1024, device=device, dtype=torch.bfloat16)
        img_attn_mask    = torch.zeros(B, 1, 1, N_max, device=device, dtype=torch.float32)
        for b, feats in enumerate(image_feats):
            N = feats.shape[0]
            img_feats_padded[b, :N] = feats
            img_attn_mask[b, 0, 0, :N] = 1.0

        # Convert to additive mask: 1->0 (attend), 0->-10000 (ignore)
        img_attn_mask = (1.0 - img_attn_mask) * -10000.0

        return img_feats_padded, img_attn_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        n_tiles_list: list[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        B = pixel_values.shape[0]

        img_feats, img_mask = self.encode_images(pixel_values, n_tiles_list, device)

        # Q-Former
        query_tokens = self.query_tokens.expand(B, -1, -1).to(device, dtype=torch.float32)
        img_feats    = img_feats.float()
        qformer_out  = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=img_feats,
            encoder_attention_mask=img_mask,
            input_ids=input_ids,
            attention_mask=None,
        )
        query_out     = qformer_out[:, :self.num_query_tokens, :]  # [B, 32, 768]
        visual_tokens = self.language_projection(query_out)        # [B, 32, lfm_hidden]
        visual_tokens = visual_tokens.to(dtype=torch.bfloat16)

        # Text embeddings from LFM
        text_embeds = self.lfm.model.embed_tokens(input_ids.to(device))  # [B, L, 896]

        full_input = torch.cat([visual_tokens, text_embeds], dim=1)       # [B, 32+L, 896]
        full_attn  = torch.cat([
            torch.ones(B, self.num_query_tokens, device=device, dtype=attention_mask.dtype),
            attention_mask.to(device),
        ], dim=1)
        full_labels = torch.cat([
            torch.full((B, self.num_query_tokens), -100, device=device, dtype=labels.dtype),
            labels.to(device),
        ], dim=1)

        outputs = self.lfm(
            inputs_embeds=full_input,
            attention_mask=full_attn,
            labels=full_labels,
        )
        return outputs.loss

    def generate(
        self,
        pixel_values: torch.Tensor,
        n_tiles_list: list[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        B = pixel_values.shape[0]

        img_feats, img_mask = self.encode_images(pixel_values, n_tiles_list, device)

        query_tokens = self.query_tokens.expand(B, -1, -1).to(device, dtype=torch.float32)
        img_feats    = img_feats.float()
        qformer_out  = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=img_feats,
            encoder_attention_mask=img_mask,
            input_ids=input_ids,
            attention_mask=None,
        )
        query_out     = qformer_out[:, :self.num_query_tokens, :]
        visual_tokens = self.language_projection(query_out).to(dtype=torch.bfloat16)

        text_embeds = self.lfm.model.embed_tokens(input_ids.to(device))
        full_input  = torch.cat([visual_tokens, text_embeds], dim=1)
        full_attn   = torch.cat([
            torch.ones(B, self.num_query_tokens, device=device, dtype=attention_mask.dtype),
            attention_mask.to(device),
        ], dim=1)

        return self.lfm.generate(
            inputs_embeds=full_input,
            attention_mask=full_attn,
            max_new_tokens=max_new_tokens,
        )

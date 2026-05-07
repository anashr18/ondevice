import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .intern_vit import InternViTEncoder
from .qformer import QFormerConfig, QFormerModel
from .projector import LanguageProjection


class InternViTQFormerLFM(nn.Module):
    """
    Dtype contract (BLIP-2 style — manual mixed precision, no autocast):
      - Frozen modules:    bfloat16  (InternViT, LFM)         <- saves memory
      - Trainable modules: float32   (Q-Former, projector,    <- numerical stability
                                      query_tokens)              for AdamW
    Forward casts at the boundaries:
      ViT(bf16) -> upcast to fp32 -> Q-Former(fp32) -> downcast to bf16 -> LFM(bf16)
    """

    def __init__(
        self,
        intern_vit_path: str = "OpenGVLab/InternViT-300M-448px",
        lfm_path: str = "LiquidAI/LFM2.5-1.2B-Instruct",
        num_query_tokens: int = 32,
        encoder_hidden_size: int = 1024,
        lfm_hidden_size: int = 2048,
        lfm_top_n_unfreeze: int = 0,
        frozen_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.frozen_dtype = frozen_dtype
        self.num_query_tokens = num_query_tokens
        self.encoder_hidden_size = encoder_hidden_size
        self.qformer_hidden_size = 768
        self.lfm_hidden_size = lfm_hidden_size

        # Frozen vision encoder (bf16 inside InternViTEncoder)
        self.intern_vit = InternViTEncoder(intern_vit_path)

        # Trainable: Q-Former (fp32)
        qformer_config = QFormerConfig(encoder_hidden_size=encoder_hidden_size)
        self.qformer = QFormerModel.from_bert_pretrained(qformer_config)

        # Trainable: query tokens (fp32)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, self.qformer_hidden_size)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Trainable: language projection (fp32)
        self.language_projection = LanguageProjection(
            self.qformer_hidden_size,
            self.lfm_hidden_size,
        )

        # Frozen LFM (bf16)
        self.lfm = AutoModelForCausalLM.from_pretrained(lfm_path, torch_dtype=frozen_dtype)
        for p in self.lfm.parameters():
            p.requires_grad = False

        if lfm_top_n_unfreeze > 0:
            inner = getattr(self.lfm, "model", self.lfm)
            layers = getattr(inner, "layers", None) or getattr(inner, "blocks", [])
            for layer in list(layers)[-lfm_top_n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self._refresh_input_embeddings()

    @property
    def device(self) -> torch.device:
        return self.query_tokens.device

    def _refresh_input_embeddings(self) -> None:
        embed_tokens = None
        if hasattr(self.lfm, "get_input_embeddings"):
            embed_tokens = self.lfm.get_input_embeddings()

        if embed_tokens is None:
            inner = getattr(self.lfm, "model", self.lfm)
            embed_tokens = (
                getattr(inner, "embed_tokens", None)
                or getattr(inner, "tok_embeddings", None)
                or getattr(inner, "token_embedding", None)
            )

        if embed_tokens is None:
            raise AttributeError(
                f"Cannot find input embeddings in {type(self.lfm).__name__}"
            )

        self._embed_tokens = embed_tokens

    def _input_vocab_size(self) -> int:
        return self._embed_tokens.num_embeddings

    def _output_vocab_size(self) -> int:
        output_embeddings = self.lfm.get_output_embeddings()
        if output_embeddings is not None:
            if hasattr(output_embeddings, "out_features"):
                return int(output_embeddings.out_features)
            if hasattr(output_embeddings, "num_embeddings"):
                return int(output_embeddings.num_embeddings)
            if hasattr(output_embeddings, "weight"):
                return int(output_embeddings.weight.shape[0])
        return int(getattr(self.lfm.config, "vocab_size"))

    def ensure_tokenizer_compatibility(self, tokenizer) -> dict[str, int | bool]:
        tokenizer_vocab_size = len(tokenizer)
        original_model_vocab_size = self._input_vocab_size()
        original_output_vocab_size = self._output_vocab_size()
        model_vocab_size = original_model_vocab_size
        output_vocab_size = original_output_vocab_size
        resized = False

        if tokenizer_vocab_size > max(model_vocab_size, output_vocab_size):
            self.lfm.resize_token_embeddings(tokenizer_vocab_size)
            self._refresh_input_embeddings()
            for param in self._embed_tokens.parameters():
                param.requires_grad = False

            output_embeddings = self.lfm.get_output_embeddings()
            if output_embeddings is not None:
                for param in output_embeddings.parameters():
                    param.requires_grad = False

            model_vocab_size = self._input_vocab_size()
            output_vocab_size = self._output_vocab_size()
            resized = True

        return {
            "original_model_vocab_size": original_model_vocab_size,
            "original_output_vocab_size": original_output_vocab_size,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "model_vocab_size": model_vocab_size,
            "output_vocab_size": output_vocab_size,
            "resized": resized,
        }

    def trainable_lfm_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.lfm.parameters() if p.requires_grad]

    def _validate_text_token_ids(self, input_ids: torch.Tensor) -> None:
        if input_ids.numel() == 0:
            return

        min_token_id = int(input_ids.min().item())
        max_token_id = int(input_ids.max().item())
        vocab_size = self._input_vocab_size()
        if min_token_id < 0:
            raise ValueError(
                f"Encountered negative token id {min_token_id}. Token ids must be in "
                f"[0, {vocab_size - 1}] for the LFM input embeddings."
            )
        if max_token_id >= vocab_size:
            raise ValueError(
                f"Encountered token id {max_token_id}, but the LFM embedding table "
                f"has only {vocab_size} rows. Check tokenizer/model compatibility."
            )

    def _validate_labels(self, labels: torch.Tensor) -> None:
        valid_labels = labels[labels != -100]
        if valid_labels.numel() == 0:
            return

        min_label = int(valid_labels.min().item())
        max_label = int(valid_labels.max().item())
        vocab_size = self._output_vocab_size()
        if min_label < 0:
            raise ValueError(
                f"Encountered negative label id {min_label}. Labels must be -100 or in "
                f"[0, {vocab_size - 1}] for the LFM loss."
            )
        if max_label >= vocab_size:
            raise ValueError(
                f"Encountered label id {max_label}, but the LFM output head supports "
                f"only {vocab_size} classes. Check tokenizer/model compatibility."
            )

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        n_tiles_list: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        B = pixel_values.shape[0]

        # Gather real tiles (skip padding) -> bf16 for ViT
        flat_tiles = torch.cat(
            [pixel_values[b, :n_tiles_list[b]] for b in range(B)], dim=0
        ).to(dtype=self.frozen_dtype)

        all_patch_feats = self.intern_vit(flat_tiles)
        n_patches_per_tile = all_patch_feats.shape[1]
        hidden_size = all_patch_feats.shape[2]

        # Per-image flatten
        image_feats = []
        offset = 0
        for b in range(B):
            n = n_tiles_list[b]
            feats = all_patch_feats[offset:offset + n]
            image_feats.append(feats.reshape(n * n_patches_per_tile, hidden_size))
            offset += n

        # Pad to N_max — UPCAST to fp32 here for Q-Former
        N_max = max(f.shape[0] for f in image_feats)
        img_feats_padded = torch.zeros(
            B,
            N_max,
            hidden_size,
            device=device,
            dtype=torch.float32,
        )
        img_attn_mask = torch.zeros(B, 1, 1, N_max, device=device, dtype=torch.float32)
        for b, feats in enumerate(image_feats):
            N = feats.shape[0]
            img_feats_padded[b, :N] = feats.float()
            img_attn_mask[b, 0, 0, :N] = 1.0

        # Additive mask
        img_attn_mask = (1.0 - img_attn_mask) * -10000.0
        return img_feats_padded, img_attn_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        n_tiles_list: list[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        device = self.device
        B = pixel_values.shape[0]

        # ViT (bf16) -> img_feats (fp32)
        img_feats, img_mask = self.encode_images(pixel_values, n_tiles_list)

        # Q-Former (fp32)
        query_tokens = self.query_tokens.expand(B, -1, -1)
        qformer_out  = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=img_feats,
            encoder_attention_mask=img_mask,
            input_ids=None,
            attention_mask=None,
        )
        query_out = qformer_out[:, :self.num_query_tokens, :]

        # Projection (fp32) -> DOWNCAST to bf16 for LFM
        visual_tokens = self.language_projection(query_out).to(dtype=self.frozen_dtype)

        # LFM (bf16)
        self._validate_text_token_ids(input_ids)
        self._validate_labels(labels)
        text_embeds = self._embed_tokens(input_ids)

        full_input  = torch.cat([visual_tokens, text_embeds], dim=1)
        full_attn   = torch.cat([
            torch.ones(B, self.num_query_tokens, device=device, dtype=attention_mask.dtype),
            attention_mask,
        ], dim=1)
        full_labels = torch.cat([
            torch.full((B, self.num_query_tokens), -100, device=device, dtype=labels.dtype),
            labels,
        ], dim=1)

        outputs = self.lfm(
            inputs_embeds=full_input,
            attention_mask=full_attn,
            labels=full_labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        n_tiles_list: list[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        **_,
    ) -> torch.Tensor:
        device = self.device
        B = pixel_values.shape[0]

        img_feats, img_mask = self.encode_images(pixel_values, n_tiles_list)

        query_tokens = self.query_tokens.expand(B, -1, -1)
        qformer_out  = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=img_feats,
            encoder_attention_mask=img_mask,
            # Q-Former is initialized from bert-base-uncased (vocab_size=30522),
            # while `input_ids` come from the LFM tokenizer. Feeding LFM token ids
            # into the Q-Former embeddings can raise "index out of range in self".
            # We only need visual query tokens here, so keep Q-Former text-free.
            input_ids=None,
            attention_mask=None,
        )
        query_out     = qformer_out[:, :self.num_query_tokens, :]
        visual_tokens = self.language_projection(query_out).to(dtype=self.frozen_dtype)

        self._validate_text_token_ids(input_ids)
        text_embeds = self._embed_tokens(input_ids)
        full_input  = torch.cat([visual_tokens, text_embeds], dim=1)
        full_attn   = torch.cat([
            torch.ones(B, self.num_query_tokens, device=device, dtype=attention_mask.dtype),
            attention_mask,
        ], dim=1)

        return self.lfm.generate(
            inputs_embeds=full_input,
            attention_mask=full_attn,
            max_new_tokens=max_new_tokens,
        )

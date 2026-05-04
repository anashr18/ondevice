# Build Context: InternViT-300M + Q-Former + Qwen2.5-0.5B

## Model Choices (Smallest Viable Setup)

| Component       | Model                                   | Params | Key Dim        | Frozen? |
|-----------------|-----------------------------------------|--------|----------------|---------|
| Vision Encoder  | `OpenGVLab/InternViT-300M-448px`        | 300M   | output = 1024  | Yes     |
| Bridge          | Q-Former (from `bert-base-uncased`)     | 188M   | hidden = 768   | No      |
| Language Model  | `Qwen/Qwen2.5-0.5B-Instruct`           | 0.5B   | hidden = 896   | Yes*    |

*Unfreeze top 4 LFM layers in stage 3 fine-tuning optionally.

Total trainable params in stage 2: ~188M (Q-Former) + Linear(768->896) ~ 189M
Total GPU memory needed: ~4-6GB in bfloat16 — runs on a single consumer GPU.

---

## Architecture Overview

```
Image (any size)
  -> dynamic tiling: 1-6 tiles + thumbnail, each 448x448
  -> [n_tiles, 3, 448, 448]
  -> InternViT-300M (frozen)
  -> [n_tiles, 1024, 1024]       <- 1024 patches/tile, 1024-dim features
  -> flatten tile+patch dims
  -> [n_tiles x 1024, 1024]      = [N, 1024]  where N varies 1024..7168
  -> pad + mask across batch
  -> Q-Former cross-attention     cross K/V proj: Linear(1024, 768)
  -> [B, 32, 768]                <- always fixed, regardless of N
  -> Linear(768, 896)             language_projection
  -> [B, 32, 896]
  -> prepend to text embeddings   [B, L, 896]
  -> [B, 32+L, 896]
  -> Qwen2.5-0.5B (frozen)
  -> generated text
```

---

## Q-Former Config (Key Numbers)

```python
QFormerConfig(
    vocab_size                  = 30522,   # bert-base-uncased vocab
    hidden_size                 = 768,     # BERT-base hidden dim
    num_hidden_layers           = 12,      # 12 transformer layers
    num_attention_heads         = 12,      # 12 heads x 64 dim = 768
    intermediate_size           = 3072,    # FFN inner dim (4 x 768)
    cross_attention_frequency   = 2,       # cross-attn at layers 0,2,4,6,8,10
    encoder_hidden_size         = 1024,    # <- InternViT-300M output dim (NOT 3200!)
    hidden_act                  = "gelu",
    layer_norm_eps              = 1e-12,
    max_position_embeddings     = 512,
)

# Linear projection
language_projection = nn.Linear(768, 896)   # 768 Q-Former -> 896 Qwen2.5-0.5B
```

THE SINGLE MOST CRITICAL NUMBER: encoder_hidden_size = 1024
This controls the K/V projection inside every cross-attention layer:
Linear(1024, 768) for K and V projections.
Get this wrong and the model silently produces garbage.

---

## Tensor Shape Reference (Concrete)

Given: batch_size=2, image0 has 5 tiles, image1 has 3 tiles, text_len=20

```
pixel_values (collated):     [2, 5, 3, 448, 448]
pixel_mask:                  [2, 5]              # [[T,T,T,T,T], [T,T,T,F,F]]

after gathering real tiles (strip padding before ViT):
  flat_tiles:                [8, 3, 448, 448]   # 5+3=8 real tiles

after InternViT-300M forward:
  all_patch_feats:           [8, 1024, 1024]    # 8 tiles x 1024 patches x 1024 dim

after split+flatten per image:
  image0_feats:              [5120, 1024]        # 5 x 1024 patches
  image1_feats:              [3072, 1024]        # 3 x 1024 patches

after padding to N_max:
  img_feats_padded:          [2, 5120, 1024]
  img_attn_mask:             [2, 1, 1, 5120]   # 1=real, 0=padding

query_tokens expanded:       [2, 32, 768]
qformer_output:              [2, 52, 768]       # 32 queries + 20 text tokens
query_out (sliced):          [2, 32, 768]       # [:, :32, :]

after language_projection:   [2, 32, 896]       # Linear(768, 896)

text_embeds:                 [2, 20, 896]
full_input:                  [2, 52, 896]       # [visual_tokens | text_embeds]

labels:                      [2, 52]
  visual positions [0:32]:   -100 (ignored)
  question positions:        -100 (ignored)
  answer positions:          real token ids     <- loss here only

LFM logits:                  [2, 52, 151936]   # Qwen2.5 vocab size
```

---

## Cross-Attention Shape Trace (Inside One Q-Former Layer)

```
# Layer idx=0 (even -> has cross-attention):

query_part:     [B, 32, 768]    # sliced from hidden_states
image_feats:    [B, N, 1024]    # N up to 5120 for 5-tile image

Q = Linear(768,  768)(query_part)   -> [B, 32, 768]
K = Linear(1024, 768)(image_feats)  -> [B, N,  768]   <- 1024->768 projection
V = Linear(1024, 768)(image_feats)  -> [B, N,  768]

reshape to multi-head (12 heads, 64 dim each):
  Q -> [B, 12, 32, 64]
  K -> [B, 12, N,  64]
  V -> [B, 12, N,  64]

attention scores:
  Q x K^T  -> [B, 12, 32, N]     # 32 queries attend to N patches
  softmax(dim=-1) -> [B, 12, 32, N]

weighted values:
  scores x V -> [B, 12, 32, 64]  # N collapsed here
  merge heads -> [B, 32, 768]
  out_proj    -> [B, 32, 768]

# N can be 1024, 2048, 3072, 4096, 5120 -- output ALWAYS [B, 32, 768]
```

---

## File Structure

```
project/
├── model/
│   ├── __init__.py
│   ├── intern_vit.py          # InternViT-300M wrapper
│   ├── qformer.py             # Full Q-Former implementation
│   ├── projector.py           # Linear(768, 896)
│   └── multimodal_model.py    # InternViTQFormerLFM top-level nn.Module
├── data/
│   ├── __init__.py
│   ├── tiling.py              # dynamic_tile_image, find_best_grid
│   ├── dataset.py             # VisionLMDataset
│   ├── sampler.py             # TileBucketSampler
│   └── collate.py             # collate_variable_tiles
├── training/
│   ├── __init__.py
│   └── trainer.py             # train_one_epoch, evaluate
├── configs/
│   ├── stage2_pretrain.yaml
│   └── stage3_finetune.yaml
├── scripts/
│   ├── train.py
│   └── infer.py
└── requirements.txt
```

---

## Detailed Module Specs

### model/intern_vit.py

```python
from transformers import AutoModel
import torch, torch.nn as nn

class InternViTEncoder(nn.Module):
    def __init__(self, model_name="OpenGVLab/InternViT-300M-448px"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        # pixel_values: [N_tiles, 3, 448, 448]  bfloat16
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        feats = outputs.last_hidden_state   # [N_tiles, 1024 or 1025, 1024]
        # Strip CLS token if present (shape[1]==1025)
        if feats.shape[1] == 1025:
            feats = feats[:, 1:, :]
        return feats   # [N_tiles, 1024, 1024]
```

### model/qformer.py

Build these classes in order:

1. QFormerConfig (dataclass with fields listed in config section above)

2. QFormerEmbeddings:
   - word_embeddings: Embedding(30522, 768)
   - position_embeddings: Embedding(512, 768)
   - LayerNorm(768, eps=1e-12) + dropout
   - forward(input_ids) -> [B, L, 768]

3. QFormerAttention(is_cross_attention=False/True):
   - if is_cross_attention:
       query  = Linear(768,  768)
       key    = Linear(1024, 768)   # encoder_hidden_size -> hidden
       value  = Linear(1024, 768)
   - else (self-attention):
       query = key = value = Linear(768, 768)
   - out_proj = Linear(768, 768)
   - num_heads=12, head_dim=64
   - forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
       Q from hidden_states always
       K, V from encoder_hidden_states if cross_attention, else from hidden_states
       scaled dot-product attention
       additive mask: 0=attend, -10000=ignore
       returns [B, seq_len, 768]

4. QFormerLayer(config, layer_idx):
   ALWAYS has:
     self.attention   = QFormerAttention(is_cross_attention=False)
     self.layernorm1  = LayerNorm(768)
   ONLY if layer_idx % 2 == 0:
     self.crossattention = QFormerAttention(is_cross_attention=True)
     self.layernorm_cross = LayerNorm(768)
     self.has_cross_attention = True
   ALWAYS has TWO separate FFNs:
     # for query tokens:
     self.ffn_q_intermediate = Linear(768, 3072)
     self.ffn_q_output       = Linear(3072, 768)
     self.layernorm_q        = LayerNorm(768)
     # for text tokens:
     self.ffn_t_intermediate = Linear(768, 3072)
     self.ffn_t_output       = Linear(3072, 768)
     self.layernorm_t        = LayerNorm(768)

   forward(hidden_states, attention_mask, encoder_hidden_states,
           encoder_attention_mask, query_length=32):

     STEP 1 self-attention over ALL tokens (queries + text together):
       sa_out = self.attention(hidden_states, attention_mask=attention_mask)
       hidden_states = self.layernorm1(hidden_states + sa_out)

     STEP 2 cross-attention on QUERY TOKENS ONLY (if has_cross_attention):
       query_part = hidden_states[:, :query_length, :]   # [B, 32, 768]
       ca_out = self.crossattention(
           query_part,
           encoder_hidden_states=encoder_hidden_states,
           attention_mask=encoder_attention_mask,
       )
       query_part = self.layernorm_cross(query_part + ca_out)
       hidden_states = cat([query_part, hidden_states[:, query_length:, :]], dim=1)

     STEP 3 split FFN by query vs text:
       q = hidden_states[:, :query_length, :]
       t = hidden_states[:, query_length:, :]
       q = layernorm_q(q + ffn_q_output(gelu(ffn_q_intermediate(q))))
       if t.shape[1] > 0:
           t = layernorm_t(t + ffn_t_output(gelu(ffn_t_intermediate(t))))
       return cat([q, t], dim=1)

5. QFormerEncoder:
   self.layers = ModuleList([QFormerLayer(config, i) for i in range(12)])
   forward: iterate through layers passing all args, return final hidden_states

6. QFormerModel:
   self.embeddings = QFormerEmbeddings(config)
   self.encoder    = QFormerEncoder(config)

   classmethod from_bert_pretrained(config):
     loads bert-base-uncased
     copies weights for:
       - embeddings (word, position, LayerNorm)
       - per layer: self-attention Q/K/V/out weights+biases, layernorm1
       - per layer: FFN intermediate+output into BOTH ffn_q_* and ffn_t_*
       - per layer: layernorm_q and layernorm_t
     does NOT copy cross-attention (randomly init is correct)
     del bert after copying

   forward(query_embeds, encoder_hidden_states, encoder_attention_mask,
           input_ids=None, attention_mask=None):
     query_length = query_embeds.shape[1]  # 32
     if input_ids is not None:
         text_embeds = self.embeddings(input_ids)
         hidden_states = cat([query_embeds, text_embeds], dim=1)
     else:
         hidden_states = query_embeds
     hidden_states = self.encoder(
         hidden_states, attention_mask,
         encoder_hidden_states, encoder_attention_mask, query_length
     )
     return hidden_states


### model/multimodal_model.py

class InternViTQFormerLFM(nn.Module):
    components:
      self.intern_vit           = InternViTEncoder("OpenGVLab/InternViT-300M-448px")
      self.query_tokens         = nn.Parameter(zeros(1, 32, 768))  # trunc_normal init std=0.02
      self.qformer              = QFormerModel.from_bert_pretrained(qformer_config)
      self.language_projection  = nn.Linear(768, 896)
      self.lfm                  = AutoModelForCausalLM.from_pretrained(
                                      "Qwen/Qwen2.5-0.5B-Instruct",
                                      torch_dtype=torch.bfloat16
                                  )  # all params frozen

    encode_images(self, pixel_values, n_tiles_list, device):
      # pixel_values: [B, max_n_tiles, 3, 448, 448]
      # 1. gather real tiles per image (don't pass padding into ViT)
      # 2. cat into flat_tiles [sum(n_tiles), 3, 448, 448]
      # 3. intern_vit(flat_tiles) -> [sum(n_tiles), 1024, 1024]
      # 4. split + flatten per image: image_i_feats = [n_i * 1024, 1024]
      # 5. pad to N_max, build mask [B, 1, 1, N_max]
      # returns img_feats_padded [B, N_max, 1024], img_attn_mask [B,1,1,N_max]

    forward(self, pixel_values, n_tiles_list, input_ids,
            attention_mask, labels, device):
      # 1. encode_images -> img_feats [B, N_max, 1024], img_mask [B,1,1,N_max]
      # 2. qformer(query_tokens, img_feats, img_mask) -> [B, 32+L, 768]
      # 3. slice [:, :32, :] -> query_out [B, 32, 768]
      # 4. language_projection(query_out) -> visual_tokens [B, 32, 896]
      # 5. lfm.model.embed_tokens(input_ids) -> text_embeds [B, L, 896]
      # 6. cat([visual_tokens, text_embeds], dim=1) -> full_input [B, 32+L, 896]
      # 7. extend attention_mask: prepend ones(B, 32) -> full_attn [B, 32+L]
      # 8. extend labels: prepend full(-100, (B,32)) -> full_labels [B, 32+L]
      # 9. lfm(inputs_embeds=full_input, attention_mask=full_attn, labels=full_labels)
      # 10. return outputs.loss

    generate(self, pixel_values, n_tiles_list, input_ids,
             attention_mask, device, max_new_tokens=128):
      # same steps 1-8 as forward but no labels
      # call lfm.generate(inputs_embeds=full_input, attention_mask=full_attn, ...)
      # return generated token ids


### data/tiling.py

find_best_grid(W, H, max_tiles) -> (cols, rows):
  try all (cols, rows) pairs where cols*rows <= max_tiles
  pick pair minimising abs(log(tile_aspect_ratio / image_aspect_ratio))
  tile_aspect_ratio = (W/cols) / (H/rows)

dynamic_tile_image(pil_image, max_tiles=6) -> list[Tensor]:
  convert to RGB
  find best grid
  crop + resize each tile to 448x448
  apply ImageNet normalisation (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  append thumbnail (full image resized to 448x448, same normalisation)
  return list of Tensors [3, 448, 448]


### data/dataset.py

class VisionLMDataset(Dataset):
  __init__(jsonl_path, tokenizer, max_tiles=6, max_text_len=256):
    load all samples from jsonl
    pre-compute self.tile_counts by briefly opening each image (no pixel load)
    store tokenizer

  __getitem__(idx):
    load image, tile it, stack -> pixel_values [n_tiles, 3, 448, 448]
    tokenize question -> q_ids [Lq]
    tokenize answer + eos_token -> a_ids [La]
    input_ids = cat([q_ids, a_ids])            [Lq+La]
    labels    = cat([full(-100,Lq), a_ids])    [Lq+La]  <- question masked
    return dict(pixel_values, n_tiles, input_ids, labels)


### data/sampler.py

class TileBucketSampler(Sampler):
  __init__(tile_counts, base_batch_size=16, max_patches_per_batch=20000,
           shuffle=True, seed=42):
    bucket samples by tile count: {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
    (key = min(n_tiles, 7))

  __iter__():
    for each bucket:
      dynamic_bs = min(base_batch_size, max_patches_per_batch // (n_tiles * 1024))
      dynamic_bs = max(dynamic_bs, 1)
      chunk indices into batches of dynamic_bs
    shuffle all batches
    yield each batch (list of indices)


### data/collate.py

make_collate_fn(pad_token_id) -> collate_fn:
  collate_variable_tiles(batch):
    pad pixel_values to [B, max_n_tiles, 3, 448, 448]
    build pixel_mask [B, max_n_tiles] bool
    pad input_ids to [B, max_text_len] using pad_token_id
    pad labels to [B, max_text_len] using -100
    build attention_mask [B, max_text_len]
    return dict of all tensors + n_tiles list


### training/trainer.py

train_one_epoch(model, loader, optimizer, scheduler, scaler,
                accum_steps=4, device="cuda", epoch=0):
  model.train()
  for step, batch in enumerate(loader):
    move tensors to device
    with autocast(bfloat16):
      loss = model.forward(...)
    n_tokens = (labels != -100).sum()
    scaler.scale(loss / accum_steps).backward()
    if (step+1) % accum_steps == 0:
      unscale, clip_grad_norm(1.0), step, update, scheduler.step, zero_grad
      log every 20 opt steps: loss, lr
  return avg loss per token

evaluate(model, loader, device):
  model.eval(), no_grad, same forward, return avg loss


### scripts/train.py

load yaml config
build tokenizer (AutoTokenizer from lfm_path)
build dataset + sampler + dataloader
build model (InternViTQFormerLFM)
build optimizer:
  param_groups:
    qformer params: lr from config
    query_tokens:   lr from config
    language_projection: lr * 0.1
  AdamW, weight_decay=0.01
build cosine schedule with warmup
build GradScaler
training loop:
  for epoch in range(max_epochs):
    train_loss = train_one_epoch(...)
    val_loss   = evaluate(...)
    if val_loss < best: save checkpoint
    early stop if no improvement for 3 epochs


### scripts/infer.py

load checkpoint
load one image + question from command line args
run model.generate(...)
decode and print answer

---

## configs/stage2_pretrain.yaml

```yaml
model:
  intern_vit_path:     "OpenGVLab/InternViT-300M-448px"
  lfm_path:            "Qwen/Qwen2.5-0.5B-Instruct"
  lfm_hidden_size:     896
  num_query_tokens:    32
  encoder_hidden_size: 1024

data:
  train_jsonl:  "data/train.jsonl"
  val_jsonl:    "data/val.jsonl"
  max_tiles:    6
  max_text_len: 256

training:
  base_batch_size:       16
  max_patches_per_batch: 20000
  accum_steps:           4
  lr:                    1.0e-4
  weight_decay:          0.01
  warmup_steps:          200
  total_steps:           50000
  save_every:            500
  output_dir:            "checkpoints/stage2"

frozen:
  intern_vit: true
  lfm:        true
```

## configs/stage3_finetune.yaml

```yaml
model:
  resume_from: "checkpoints/stage2/best.pt"

data:
  train_jsonl:  "data/finetune_train.jsonl"
  val_jsonl:    "data/finetune_val.jsonl"
  max_tiles:    6
  max_text_len: 512

training:
  base_batch_size:       8
  max_patches_per_batch: 15000
  accum_steps:           4
  lr:                    2.0e-5
  warmup_steps:          50
  total_steps:           10000
  save_every:            200
  output_dir:            "checkpoints/stage3"

frozen:
  intern_vit:          true
  lfm_top_n_unfreeze:  4
```

## requirements.txt

```
torch>=2.1.0
transformers>=4.40.0
Pillow>=10.0.0
pyyaml>=6.0
tqdm>=4.65.0
einops>=0.7.0
timm>=0.9.0
sentencepiece>=0.1.99
```

---

## JSONL Data Format

```json
{"image_path": "images/cat.jpg", "question": "What animal is in the image?", "answer": "A cat."}
{"image_path": "images/invoice.png", "question": "What is the total amount?", "answer": "The total is $142.50."}
```

For KV extraction:
```json
{"image_path": "images/form.png", "question": "Extract as JSON: {\"name\": \"string\", \"date\": \"string\", \"amount\": \"number\"}", "answer": "{\"name\": \"John Doe\", \"date\": \"2024-01-15\", \"amount\": 142.50}"}
```

---

## Build Order

1. data/tiling.py — test: dynamic_tile_image returns list of [3,448,448] tensors
2. model/intern_vit.py — test: [4,3,448,448] -> [4,1024,1024]
3. model/qformer.py — test: query[2,32,768] + image[2,512,1024] -> output[2,32,768]
4. model/projector.py — trivial Linear(768,896)
5. model/multimodal_model.py — test full forward, loss is scalar
6. data/dataset.py + sampler.py + collate.py — test one batch, print shapes
7. training/trainer.py — test 5 steps
8. scripts/train.py
9. scripts/infer.py
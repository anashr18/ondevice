# Export Commands

## FP32 baseline
```bash
optimum-cli export executorch \
  --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --task "text-generation" \
  --recipe "xnnpack" \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --output_dir baseline_fp32
```

## 8da4w linears only (Python-compatible)
```bash
optimum-cli export executorch \
  --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --task "text-generation" \
  --recipe "xnnpack" \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --qlinear 8da4w \
  --output_dir quantized_8da4w_no_emb
```

## 8da4w linears + 8w embeddings (device-only)
```bash
optimum-cli export executorch \
  --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --task "text-generation" \
  --recipe "xnnpack" \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --qlinear 8da4w \
  --qembedding 8w \
  --output_dir quantized_8da4w
```
> Note: `quantized_8da4w` requires the native ExecuTorch runtime (Android/iOS) — `quantized_decomposed::embedding_byte.out` is not registered in the Python runtime.

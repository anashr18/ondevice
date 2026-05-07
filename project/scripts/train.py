import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset  import VisionLMDataset
from data.sampler  import TileBucketSampler
from data.collate  import make_collate_fn
from model.multimodal_model import InternViTQFormerLFM
from training.trainer       import evaluate


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def format_dtype(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "unknown"
    return str(dtype).replace("torch.", "")


def module_dtype(module) -> torch.dtype | None:
    if module is None:
        return None
    try:
        return next(module.parameters()).dtype
    except (AttributeError, StopIteration):
        return None


def describe_runtime(accelerator: Accelerator, model=None) -> list[str]:
    lines = [
        f"accelerator.device={accelerator.device}",
        f"accelerator.distributed_type={accelerator.distributed_type}",
        f"accelerator.num_processes={accelerator.num_processes}",
        f"accelerator.process_index={accelerator.process_index}",
        f"accelerator.local_process_index={accelerator.local_process_index}",
        f"accelerator.mixed_precision={accelerator.mixed_precision}",
        "model_precision_policy=frozen_modules:bf16 trainable_modules:fp32",
        f"torch.cuda.is_available={torch.cuda.is_available()}",
        f"torch.cuda.device_count={torch.cuda.device_count()}",
    ]

    if torch.cuda.is_available():
        current_idx = torch.cuda.current_device()
        lines.extend([
            f"torch.cuda.current_device={current_idx}",
            f"torch.cuda.device_name={torch.cuda.get_device_name(current_idx)}",
        ])

    if model is not None:
        unwrapped = accelerator.unwrap_model(model)
        try:
            lines.extend([
                f"model.device={next(unwrapped.parameters()).device}",
                f"vision_dtype={format_dtype(module_dtype(getattr(unwrapped.intern_vit, 'model', None)))}",
                f"lfm_dtype={format_dtype(module_dtype(unwrapped.lfm))}",
                f"qformer_dtype={format_dtype(next(unwrapped.qformer.parameters()).dtype)}",
                f"projector_dtype={format_dtype(next(unwrapped.language_projection.parameters()).dtype)}",
                f"query_tokens_dtype={format_dtype(unwrapped.query_tokens.dtype)}",
            ])
        except StopIteration:
            pass

    return lines


def build_optimizer(model: InternViTQFormerLFM, train_cfg: dict) -> AdamW:
    param_groups = [
        {"params": list(model.qformer.parameters()), "lr": train_cfg["lr"]},
        {"params": [model.query_tokens], "lr": train_cfg["lr"]},
        {
            "params": list(model.language_projection.parameters()),
            "lr": train_cfg.get("projector_lr", train_cfg["lr"] * 0.1),
        },
    ]

    lfm_trainable_params = model.trainable_lfm_parameters()
    if lfm_trainable_params:
        param_groups.append({
            "params": lfm_trainable_params,
            "lr": train_cfg.get("lfm_lr", train_cfg["lr"] * 0.1),
        })

    return AdamW(param_groups, weight_decay=train_cfg["weight_decay"])


def load_training_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    with open(state_path) as f:
        return json.load(f)


def save_training_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def resolve_existing_path(path_str: str, config_path: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        config_path.parent / path,
        config_path.parent.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--max_steps",  type=int, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m_cfg, d_cfg, t_cfg = cfg["model"], cfg["data"], cfg["training"]
    fr_cfg              = cfg.get("frozen", {})
    ac_cfg              = cfg.get("accelerate", {})
    output_dir          = t_cfg["output_dir"]
    best_ckpt_dir       = Path(output_dir) / "best"
    best_state_path     = best_ckpt_dir / "trainer_state.json"

    # Mixed precision is handled INSIDE the model (frozen=bf16, trainable=fp32).
    # Keep accelerator's mixed_precision="no" so it doesn't add an autocast that
    # conflicts with our explicit dtype boundaries.
    accelerator = Accelerator(
        mixed_precision=ac_cfg.get("mixed_precision", "no"),
        gradient_accumulation_steps=t_cfg["accum_steps"],
        project_config=ProjectConfiguration(
            project_dir=output_dir,
            logging_dir=os.path.join(output_dir, "logs"),
        ),
        log_with=ac_cfg.get("log_with"),  # None = no tracker; don't pass [] here
    )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    for line in describe_runtime(accelerator):
        accelerator.print(f"[init] {line}")

    tokenizer = AutoTokenizer.from_pretrained(m_cfg["lfm_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_jsonl = resolve_existing_path(d_cfg["train_jsonl"], config_path)
    val_jsonl   = resolve_existing_path(d_cfg["val_jsonl"],   config_path)
    if not train_jsonl.exists():
        raise FileNotFoundError(
            f"Train JSONL not found: {train_jsonl}. "
            "Pass a config with the correct path or run the dataset prep script first."
        )
    if not val_jsonl.exists():
        raise FileNotFoundError(
            f"Val JSONL not found: {val_jsonl}. "
            "Pass a config with the correct path or run the dataset prep script first."
        )

    train_ds = VisionLMDataset(str(train_jsonl), tokenizer, d_cfg["max_tiles"], d_cfg["max_text_len"])
    val_ds   = VisionLMDataset(str(val_jsonl),   tokenizer, d_cfg["max_tiles"], d_cfg["max_text_len"])
    if len(train_ds) == 0:
        raise ValueError(f"Training dataset is empty: {train_jsonl}")
    if len(val_ds) == 0:
        raise ValueError(f"Validation dataset is empty: {val_jsonl}")

    train_sampler = TileBucketSampler(
        train_ds.tile_counts,
        base_batch_size=t_cfg["base_batch_size"],
        max_patches_per_batch=t_cfg["max_patches_per_batch"],
        shuffle=True,
    )
    collate_fn   = make_collate_fn(tokenizer.pad_token_id)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=2, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=pin_memory)

    model = InternViTQFormerLFM(
        intern_vit_path=m_cfg["intern_vit_path"],
        lfm_path=m_cfg["lfm_path"],
        num_query_tokens=m_cfg["num_query_tokens"],
        encoder_hidden_size=m_cfg["encoder_hidden_size"],
        lfm_hidden_size=m_cfg["lfm_hidden_size"],
        lfm_top_n_unfreeze=fr_cfg.get("lfm_top_n_unfreeze", 0),
    )
    vocab_info = model.ensure_tokenizer_compatibility(tokenizer)
    if vocab_info["resized"]:
        accelerator.print(
            "[init] Resized LFM input embeddings from "
            f"{vocab_info['original_model_vocab_size']} to match tokenizer size "
            f"{vocab_info['tokenizer_vocab_size']}"
        )
    accelerator.print(
        "[init] vocab sizes: "
        f"tokenizer={vocab_info['tokenizer_vocab_size']} "
        f"input_embeddings={vocab_info['model_vocab_size']} "
        f"lm_head={vocab_info['output_vocab_size']}"
    )

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    accelerator.print(f"[init] params: trainable={n_train/1e6:.1f}M  total={n_total/1e9:.2f}B")

    total_steps = args.max_steps or t_cfg["total_steps"]
    eval_every = t_cfg.get("eval_every", 500)
    log_every = t_cfg.get("log_every", 20)
    early_stopping_patience = t_cfg.get("early_stopping_patience", 3)

    optimizer = build_optimizer(model, t_cfg)
    scheduler = cosine_with_warmup(optimizer, t_cfg["warmup_steps"], total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    for line in describe_runtime(accelerator, model):
        accelerator.print(f"[runtime] {line}")

    # Capture trainable params AFTER prepare (DDP wraps modules)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    best_val = float("inf")
    no_improve = 0
    global_step = 0
    epoch = 0

    if args.resume_dir:
        accelerator.load_state(args.resume_dir)
        resume_state = load_training_state(Path(args.resume_dir) / "trainer_state.json")
        if resume_state:
            best_val = float(resume_state.get("best_val", best_val))
            no_improve = int(resume_state.get("no_improve", no_improve))
            global_step = int(resume_state.get("global_step", global_step))
            epoch = int(resume_state.get("epoch", epoch))
        accelerator.print(
            f"[resume] dir={args.resume_dir} step={global_step} "
            f"epoch={epoch} best_val={best_val:.4f}"
        )

    model.train()

    while global_step < total_steps:
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= total_steps:
                break

            with accelerator.accumulate(model):
                loss = model(
                    pixel_values=batch["pixel_values"],
                    n_tiles_list=batch["n_tiles_list"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Counters / logging / eval only fire on the *real* step
            if accelerator.sync_gradients:
                global_step += 1

                if global_step % log_every == 0:
                    reduced_loss = accelerator.reduce(
                        loss.detach().float(),
                        reduction="mean",
                    ).item()
                    accelerator.print(
                        f"step={global_step:>5d}  loss={reduced_loss:.4f}  "
                        f"lr={current_lr(optimizer):.2e}"
                    )

                if global_step % eval_every == 0 or global_step == total_steps:
                    val_loss = evaluate(accelerator, model, val_loader)
                    accelerator.print(f"step={global_step:>5d}  val_loss={val_loss:.4f}")

                    if val_loss < best_val:
                        best_val = val_loss
                        no_improve = 0
                        accelerator.save_state(str(best_ckpt_dir))
                        if accelerator.is_main_process:
                            save_training_state(
                                best_state_path,
                                {
                                    "best_val": best_val,
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "no_improve": no_improve,
                                },
                            )
                            accelerator.save(
                                accelerator.get_state_dict(model),
                                os.path.join(output_dir, "best_model.pt"),
                            )
                        accelerator.wait_for_everyone()
                    else:
                        no_improve += 1

                    if no_improve >= early_stopping_patience:
                        accelerator.print("Early stopping.")
                        global_step = total_steps
                        model.train()
                        break

                    model.train()

        epoch += 1

    accelerator.wait_for_everyone()
    accelerator.print(f"Training done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()

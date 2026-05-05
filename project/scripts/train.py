import argparse
import math
import os
import sys

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import VisionLMDataset
from data.sampler import TileBucketSampler
from data.collate import make_collate_fn
from model.multimodal_model import InternViTQFormerLFM
from training.trainer import evaluate


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--max_steps",  type=int, default=None, help="Override total_steps (smoke test)")
    parser.add_argument("--resume_dir", type=str, default=None, help="Path to accelerate save_state dir to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    m_cfg  = cfg["model"]
    d_cfg  = cfg["data"]
    t_cfg  = cfg["training"]
    fr_cfg = cfg.get("frozen", {})
    ac_cfg = cfg.get("accelerate", {})

    output_dir = t_cfg["output_dir"]

    # mixed_precision=None means "defer to accelerate launch --config_file or env"
    # Set it explicitly in finetune.yaml accelerate.mixed_precision to override
    accelerator = Accelerator(
        mixed_precision=ac_cfg.get("mixed_precision"),
        gradient_accumulation_steps=t_cfg["accum_steps"],
        project_config=ProjectConfiguration(
            project_dir=output_dir,
            logging_dir=os.path.join(output_dir, "logs"),
        ),
        log_with=ac_cfg.get("log_with") or [],
    )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Mixed precision: {accelerator.mixed_precision} | "
              f"Num processes: {accelerator.num_processes} | "
              f"Device: {accelerator.device}")

    tokenizer = AutoTokenizer.from_pretrained(m_cfg["lfm_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds = VisionLMDataset(d_cfg["train_jsonl"], tokenizer, d_cfg["max_tiles"], d_cfg["max_text_len"])
    val_ds   = VisionLMDataset(d_cfg["val_jsonl"],   tokenizer, d_cfg["max_tiles"], d_cfg["max_text_len"])

    train_sampler = TileBucketSampler(
        train_ds.tile_counts,
        base_batch_size=t_cfg["base_batch_size"],
        max_patches_per_batch=t_cfg["max_patches_per_batch"],
        shuffle=True,
    )
    collate_fn   = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,  collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = InternViTQFormerLFM(
        intern_vit_path=m_cfg["intern_vit_path"],
        lfm_path=m_cfg["lfm_path"],
        num_query_tokens=m_cfg["num_query_tokens"],
        encoder_hidden_size=m_cfg["encoder_hidden_size"],
        lfm_hidden_size=m_cfg["lfm_hidden_size"],
        lfm_top_n_unfreeze=fr_cfg.get("lfm_top_n_unfreeze", 0),
    )

    total_steps = args.max_steps or t_cfg["total_steps"]

    optimizer = AdamW([
        {"params": list(model.qformer.parameters()),             "lr": t_cfg["lr"]},
        {"params": [model.query_tokens],                         "lr": t_cfg["lr"]},
        {"params": list(model.language_projection.parameters()), "lr": t_cfg["lr"] * 0.1},
    ], weight_decay=t_cfg["weight_decay"])

    scheduler = cosine_with_warmup(optimizer, t_cfg["warmup_steps"], total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if args.resume_dir:
        accelerator.load_state(args.resume_dir)

    best_val    = float("inf")
    no_improve  = 0
    global_step = 0
    log_every   = t_cfg.get("log_every", 20)
    eval_every  = t_cfg.get("eval_every", 500)
    epoch       = 0

    model.train()

    while global_step < total_steps:
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

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
                    device=accelerator.device,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % log_every == 0 and accelerator.is_main_process:
                        print(f"step={global_step} loss={loss.detach().float().item():.4f} "
                              f"lr={current_lr(optimizer):.2e}")

                    if global_step % eval_every == 0:
                        val_loss = evaluate(accelerator, model, val_loader)
                        model.train()

                        if accelerator.is_main_process:
                            print(f"step={global_step} val_loss={val_loss:.4f}")

                        if val_loss < best_val:
                            best_val   = val_loss
                            no_improve = 0
                            accelerator.save_state(os.path.join(output_dir, "best"))
                            if accelerator.is_main_process:
                                accelerator.save(
                                    accelerator.unwrap_model(model).state_dict(),
                                    os.path.join(output_dir, "best_model.pt"),
                                )
                        else:
                            no_improve += 1

                        # All ranks agree on early stopping
                        stop_flag = accelerator.reduce(
                            torch.tensor(int(no_improve >= 3), device=accelerator.device),
                            reduction="sum",
                        )
                        if stop_flag.item() > 0:
                            if accelerator.is_main_process:
                                print("Early stopping.")
                            global_step = total_steps  # signal outer while to exit
                            break

        epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Training done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()

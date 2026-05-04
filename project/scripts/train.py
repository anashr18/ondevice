import argparse
import math
import os
import sys

import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import VisionLMDataset
from data.sampler import TileBucketSampler
from data.collate import make_collate_fn
from model.multimodal_model import InternViTQFormerLFM
from training.trainer import train_one_epoch, evaluate


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=None, help="Override total_steps (for smoke test)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    m_cfg   = cfg["model"]
    d_cfg   = cfg["data"]
    t_cfg   = cfg["training"]
    fr_cfg  = cfg.get("frozen", {})

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
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,  collate_fn=collate_fn, num_workers=2)

    model = InternViTQFormerLFM(
        intern_vit_path=m_cfg["intern_vit_path"],
        lfm_path=m_cfg["lfm_path"],
        num_query_tokens=m_cfg["num_query_tokens"],
        encoder_hidden_size=m_cfg["encoder_hidden_size"],
        lfm_hidden_size=m_cfg["lfm_hidden_size"],
        lfm_top_n_unfreeze=fr_cfg.get("lfm_top_n_unfreeze", 0),
    ).to(device)

    qformer_params     = list(model.qformer.parameters())
    query_params       = [model.query_tokens]
    proj_params        = list(model.language_projection.parameters())

    optimizer = AdamW([
        {"params": qformer_params, "lr": t_cfg["lr"]},
        {"params": query_params,   "lr": t_cfg["lr"]},
        {"params": proj_params,    "lr": t_cfg["lr"] * 0.1},
    ], weight_decay=t_cfg["weight_decay"])

    total_steps  = args.max_steps if args.max_steps else t_cfg["total_steps"]
    scheduler    = cosine_with_warmup(optimizer, t_cfg["warmup_steps"], total_steps)
    scaler       = GradScaler()

    output_dir = t_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    best_val  = float("inf")
    no_improve = 0
    global_step = 0

    for epoch in range(9999):
        if global_step >= total_steps:
            break
        train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            accum_steps=t_cfg["accum_steps"], device=device, epoch=epoch,
        )
        val_loss = evaluate(model, val_loader, device=device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       os.path.join(output_dir, "best.pt"))
        else:
            no_improve += 1
            if no_improve >= 3:
                print("Early stopping.")
                break

        global_step += len(train_loader) // t_cfg["accum_steps"]

        if (epoch + 1) % (t_cfg["save_every"] // max(1, len(train_loader) // t_cfg["accum_steps"])) == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       os.path.join(output_dir, f"ckpt_epoch{epoch}.pt"))


if __name__ == "__main__":
    main()

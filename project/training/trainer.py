import torch
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    accum_steps: int = 4,
    device: str = "cuda",
    epoch: int = 0,
) -> float:
    model.train()
    total_loss   = 0.0
    total_tokens = 0
    opt_step     = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        pixel_values   = batch["pixel_values"].to(device)
        n_tiles_list   = batch["n_tiles_list"]
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with autocast(dtype=torch.bfloat16):
            loss = model(
                pixel_values=pixel_values,
                n_tiles_list=n_tiles_list,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                device=device,
            )

        n_tokens     = (labels != -100).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

        scaler.scale(loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1

            if opt_step % 20 == 0:
                avg = total_loss / max(total_tokens, 1)
                lr  = scheduler.get_last_lr()[0]
                print(f"epoch={epoch} opt_step={opt_step} loss={avg:.4f} lr={lr:.2e}")

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, loader, device: str = "cuda") -> float:
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for batch in loader:
        pixel_values   = batch["pixel_values"].to(device)
        n_tiles_list   = batch["n_tiles_list"]
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with autocast(dtype=torch.bfloat16):
            loss = model(
                pixel_values=pixel_values,
                n_tiles_list=n_tiles_list,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                device=device,
            )

        n_tokens     = (labels != -100).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)

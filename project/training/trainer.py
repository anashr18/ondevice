import torch
from accelerate import Accelerator


def train_one_epoch(
    accelerator: Accelerator,
    model,
    loader,
    optimizer,
    scheduler,
    epoch: int = 0,
) -> float:
    model.train()
    total_loss   = 0.0
    total_tokens = 0
    opt_step     = 0

    for step, batch in enumerate(loader):
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

        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss   += loss.detach().float().item() * n_tokens
        total_tokens += n_tokens

        if accelerator.sync_gradients:
            opt_step += 1
            if opt_step % 20 == 0 and accelerator.is_main_process:
                avg = total_loss / max(total_tokens, 1)
                lr  = scheduler.get_last_lr()[0]
                print(f"epoch={epoch} opt_step={opt_step} loss={avg:.4f} lr={lr:.2e}")

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(accelerator: Accelerator, model, loader) -> float:
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for batch in loader:
        loss = model(
            pixel_values=batch["pixel_values"],
            n_tiles_list=batch["n_tiles_list"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            device=accelerator.device,
        )
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss   += loss.detach().float().item() * n_tokens
        total_tokens += n_tokens

    # Gather across GPUs
    total_loss_t   = torch.tensor(total_loss,   device=accelerator.device)
    total_tokens_t = torch.tensor(total_tokens, device=accelerator.device)
    total_loss_t   = accelerator.gather(total_loss_t).sum().item()
    total_tokens_t = accelerator.gather(total_tokens_t).sum().item()

    return total_loss_t / max(total_tokens_t, 1)

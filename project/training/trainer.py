import torch
from accelerate import Accelerator


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
        n_tokens      = (batch["labels"] != -100).sum().item()
        total_loss   += loss.detach().float().item() * n_tokens
        total_tokens += n_tokens

    total_loss_t   = torch.tensor(total_loss,   device=accelerator.device)
    total_tokens_t = torch.tensor(total_tokens, device=accelerator.device)
    total_loss_t   = accelerator.gather(total_loss_t).sum().item()
    total_tokens_t = accelerator.gather(total_tokens_t).sum().item()

    return total_loss_t / max(total_tokens_t, 1)

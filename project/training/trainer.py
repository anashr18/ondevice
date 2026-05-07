import torch
from accelerate import Accelerator


@torch.no_grad()
def evaluate(accelerator: Accelerator, model, loader) -> float:
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=accelerator.device, dtype=torch.float64)

    for batch in loader:
        loss = model(
            pixel_values=batch["pixel_values"],
            n_tiles_list=batch["n_tiles_list"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        n_tokens = (batch["labels"] != -100).sum().to(torch.float64)
        total_loss += loss.detach().to(torch.float64) * n_tokens
        total_tokens += n_tokens

    total_loss = accelerator.reduce(total_loss, reduction="sum")
    total_tokens = accelerator.reduce(total_tokens, reduction="sum")

    if was_training:
        model.train()

    return (total_loss / total_tokens.clamp_min(1.0)).item()

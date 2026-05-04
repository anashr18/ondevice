import torch


def make_collate_fn(pad_token_id: int):
    def collate_variable_tiles(batch: list[dict]) -> dict:
        max_n_tiles  = max(s["pixel_values"].shape[0] for s in batch)
        max_text_len = max(s["input_ids"].shape[0]    for s in batch)
        B = len(batch)

        pixel_values = torch.zeros(B, max_n_tiles, 3, 448, 448)
        pixel_mask   = torch.zeros(B, max_n_tiles, dtype=torch.bool)
        input_ids    = torch.full((B, max_text_len), pad_token_id, dtype=torch.long)
        labels       = torch.full((B, max_text_len), -100,         dtype=torch.long)
        attn_mask    = torch.zeros(B, max_text_len, dtype=torch.long)
        n_tiles_list = []

        for b, s in enumerate(batch):
            n = s["pixel_values"].shape[0]
            L = s["input_ids"].shape[0]
            pixel_values[b, :n] = s["pixel_values"]
            pixel_mask[b, :n]   = True
            input_ids[b, :L]    = s["input_ids"]
            labels[b, :L]       = s["labels"]
            attn_mask[b, :L]    = 1
            n_tiles_list.append(n)

        return {
            "pixel_values":   pixel_values,
            "pixel_mask":     pixel_mask,
            "n_tiles_list":   n_tiles_list,
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attn_mask,
        }

    return collate_variable_tiles

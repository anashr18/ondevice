import json

import torch
from PIL import Image
from torch.utils.data import Dataset

from .tiling import dynamic_tile_image


class VisionLMDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_tiles: int = 6, max_text_len: int = 512):
        self.samples      = []
        self.tokenizer    = tokenizer
        self.max_tiles    = max_tiles
        self.max_text_len = max_text_len

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        # Pre-compute tile counts without loading pixel data
        self.tile_counts = []
        from .tiling import find_best_grid
        for s in self.samples:
            try:
                with Image.open(s["image_path"]) as img:
                    W, H = img.size
                cols, rows = find_best_grid(W, H, max_tiles)
                self.tile_counts.append(cols * rows + 1)  # +1 for thumbnail
            except Exception:
                self.tile_counts.append(1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        with Image.open(sample["image_path"]) as img:
            tiles = dynamic_tile_image(img, self.max_tiles)
        pixel_values = torch.stack(tiles)  # [n_tiles, 3, 448, 448]
        n_tiles      = pixel_values.shape[0]

        tok = self.tokenizer
        q_ids = tok(sample["question"], add_special_tokens=False).input_ids
        a_ids = tok(sample["answer"],   add_special_tokens=False).input_ids
        if tok.eos_token_id is not None:
            a_ids = a_ids + [tok.eos_token_id]

        # Truncate to max_text_len total
        max_a = self.max_text_len - len(q_ids)
        if max_a < 1:
            q_ids = q_ids[:self.max_text_len - 1]
            max_a = 1
        a_ids = a_ids[:max_a]

        input_ids = torch.tensor(q_ids + a_ids, dtype=torch.long)
        labels    = torch.tensor([-100] * len(q_ids) + a_ids, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "n_tiles":      n_tiles,
            "input_ids":    input_ids,
            "labels":       labels,
        }

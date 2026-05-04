import argparse
import os
import sys

import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tiling import dynamic_tile_image
from model.multimodal_model import InternViTQFormerLFM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image",      required=True)
    parser.add_argument("--question",   required=True)
    parser.add_argument("--lfm_path",   default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.lfm_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = InternViTQFormerLFM(lfm_path=args.lfm_path).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    img   = Image.open(args.image)
    tiles = dynamic_tile_image(img)
    pixel_values = torch.stack(tiles).unsqueeze(0)  # [1, n_tiles, 3, 448, 448]
    n_tiles_list = [pixel_values.shape[1]]

    input_ids = torch.tensor(
        tokenizer(args.question, add_special_tokens=False).input_ids,
        dtype=torch.long,
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_ids = model.generate(
            pixel_values=pixel_values,
            n_tiles_list=n_tiles_list,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

import math
from PIL import Image
import torch
import torchvision.transforms as T

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def find_best_grid(W: int, H: int, max_tiles: int) -> tuple[int, int]:
    image_ar = W / H
    best = (1, 1)
    best_cost = float("inf")
    for cols in range(1, max_tiles + 1):
        for rows in range(1, max_tiles + 1):
            if cols * rows > max_tiles:
                continue
            tile_ar = (W / cols) / (H / rows)
            cost = abs(math.log(tile_ar / image_ar))
            if cost < best_cost:
                best_cost = cost
                best = (cols, rows)
    return best


def dynamic_tile_image(pil_image: Image.Image, max_tiles: int = 6) -> list[torch.Tensor]:
    img = pil_image.convert("RGB")
    W, H = img.size
    cols, rows = find_best_grid(W, H, max_tiles)

    tile_w = W / cols
    tile_h = H / rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            left   = int(round(c * tile_w))
            upper  = int(round(r * tile_h))
            right  = int(round((c + 1) * tile_w))
            lower  = int(round((r + 1) * tile_h))
            tile   = img.crop((left, upper, right, lower)).resize((448, 448), Image.BICUBIC)
            tiles.append(_transform(tile))

    thumbnail = img.resize((448, 448), Image.BICUBIC)
    tiles.append(_transform(thumbnail))

    return tiles  # each [3, 448, 448]

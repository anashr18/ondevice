import math
from PIL import Image
import torch

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _to_tensor_normalized(pil_image: Image.Image) -> torch.Tensor:
    tensor = torch.tensor(bytearray(pil_image.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(pil_image.size[1], pil_image.size[0], 3).permute(2, 0, 1).float()
    tensor = tensor.div_(255.0)
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def find_best_grid(W: int, H: int, max_tiles: int) -> tuple[int, int]:
    image_ar = W / H
    best = (1, 1)
    best_cost = float("inf")
    for cols in range(1, max_tiles + 1):
        for rows in range(1, max_tiles + 1):
            if cols * rows > max_tiles:
                continue
            # We resize every crop to a square, so we want each tile itself to be
            # as close to square as possible: (W / cols) / (H / rows) ~= 1.
            tile_ar = (W * rows) / (H * cols)
            cost = abs(math.log(tile_ar))
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
            tiles.append(_to_tensor_normalized(tile))

    thumbnail = img.resize((448, 448), Image.BICUBIC)
    tiles.append(_to_tensor_normalized(thumbnail))

    return tiles  # each [3, 448, 448]

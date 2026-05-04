import random
from torch.utils.data import Sampler


class TileBucketSampler(Sampler):
    def __init__(
        self,
        tile_counts: list[int],
        base_batch_size: int = 16,
        max_patches_per_batch: int = 20000,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.tile_counts           = tile_counts
        self.base_batch_size       = base_batch_size
        self.max_patches_per_batch = max_patches_per_batch
        self.shuffle               = shuffle
        self.seed                  = seed
        self._epoch                = 0

        buckets: dict[int, list[int]] = {}
        for idx, n in enumerate(tile_counts):
            key = min(n, 7)
            buckets.setdefault(key, []).append(idx)
        self.buckets = buckets

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _make_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        all_batches = []
        for n_tiles, indices in self.buckets.items():
            dynamic_bs = min(
                self.base_batch_size,
                max(1, self.max_patches_per_batch // (n_tiles * 1024)),
            )
            idxs = indices[:]
            if self.shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), dynamic_bs):
                batch = idxs[start:start + dynamic_bs]
                if batch:
                    all_batches.append(batch)
        if self.shuffle:
            rng.shuffle(all_batches)
        return all_batches

    def __iter__(self):
        for batch in self._make_batches():
            yield batch

    def __len__(self) -> int:
        return len(self._make_batches())

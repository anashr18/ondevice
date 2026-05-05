
import argparse
import json
import random
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm


COCO_TRAIN2017_URL = "http://images.cocodataset.org/zips/train2017.zip"
LLAVA_REPO_ID = "liuhaotian/LLaVA-Instruct-150K"
LLAVA_FILENAME = "llava_instruct_150k.json"


def _download_with_progress(url: str, dest: Path) -> None:
    # aria2c uses 16 parallel connections — much faster than a single urllib stream
    if shutil.which("aria2c"):
        print(f"Downloading with aria2c (16 connections): {dest.name}")
        subprocess.run(
            [
                "aria2c",
                "-x", "16",   # 16 parallel connections to same server
                "-s", "16",   # 16 segments
                "-k", "10M",  # 10MB chunk size
                "--file-allocation=none",
                "-d", str(dest.parent),
                "-o", dest.name,
                url,
            ],
            check=True,
        )
    else:
        print(f"aria2c not found — falling back to single-connection download.")
        print("  Install with: sudo apt install aria2   (or brew install aria2 on Mac)")

        class _Bar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with _Bar(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=dest.name) as bar:
            urllib.request.urlretrieve(url, dest, reporthook=bar.update_to)


def download_coco(data_dir: Path) -> Path:
    images_dir = data_dir / "coco" / "train2017"
    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"COCO images already found at {images_dir}, skipping download.")
        return images_dir

    zip_path = data_dir / "coco" / "train2017.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Downloading COCO train2017 (~18 GB) to {zip_path} ...")
        _download_with_progress(COCO_TRAIN2017_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(zip_path.parent)

    zip_path.unlink()  # free disk space
    return images_dir


def download_llava_json(cache_dir: Path) -> Path:
    print(f"Downloading {LLAVA_FILENAME} from HuggingFace ...")
    local_path = hf_hub_download(
        repo_id=LLAVA_REPO_ID,
        filename=LLAVA_FILENAME,
        repo_type="dataset",
        cache_dir=str(cache_dir / "hf_cache"),
    )
    return Path(local_path)


def strip_image_token(text: str) -> str:
    return re.sub(r"<image>\s*", "", text).strip()


def convert(llava_json: Path, coco_dir: Path) -> list[dict]:
    with open(llava_json) as f:
        data = json.load(f)

    records = []
    skipped_no_image = 0
    skipped_missing_file = 0

    for entry in tqdm(data, desc="Converting"):
        image_file = entry.get("image", "")
        if not image_file:
            skipped_no_image += 1
            continue

        image_path = (coco_dir / image_file).resolve()
        if not image_path.exists():
            skipped_missing_file += 1
            continue

        convs = entry.get("conversations", [])
        human_turns = [c for c in convs if c.get("from") == "human"]
        gpt_turns = [c for c in convs if c.get("from") == "gpt"]

        if not human_turns or not gpt_turns:
            skipped_no_image += 1
            continue

        question = strip_image_token(human_turns[0]["value"])
        answer = gpt_turns[0]["value"].strip()

        if not question or not answer:
            skipped_no_image += 1
            continue

        records.append({
            "image_path": str(image_path),
            "question": question,
            "answer": answer,
        })

    print(f"  Converted: {len(records)}")
    print(f"  Skipped (no image field / bad conv): {skipped_no_image}")
    print(f"  Skipped (image file missing on disk): {skipped_missing_file}")
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Root dir for JSONL output and downloads")
    parser.add_argument("--coco_dir", default=None, help="Path to existing COCO train2017/ folder (skips download)")
    parser.add_argument("--val_frac", type=float, default=0.05, help="Fraction for validation split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_coco_download", action="store_true", help="Skip COCO image download")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: resolve COCO image directory
    if args.coco_dir:
        coco_dir = Path(args.coco_dir).resolve()
        if not coco_dir.exists():
            raise FileNotFoundError(f"--coco_dir not found: {coco_dir}")
        print(f"Using existing COCO images at {coco_dir}")
    elif args.skip_coco_download:
        coco_dir = data_dir / "coco" / "train2017"
        print(f"--skip_coco_download set; expecting images at {coco_dir}")
    else:
        coco_dir = download_coco(data_dir)

    # Step 2: download LLaVA JSON
    llava_json = download_llava_json(data_dir)

    # Step 3: convert
    records = convert(llava_json, coco_dir)

    # Step 4: split
    random.seed(args.seed)
    random.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    val_records = records[:n_val]
    train_records = records[n_val:]

    # Step 5: write
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    print(f"\nDone.")
    print(f"  Train: {len(train_records)} samples → {train_path}")
    print(f"  Val:   {len(val_records)} samples → {val_path}")


if __name__ == "__main__":
    main()

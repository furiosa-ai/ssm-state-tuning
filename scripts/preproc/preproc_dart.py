import argparse

from dataset.dart import DartDataset
from modules import _load_mamba_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    tokenizer = _load_mamba_tokenizer()
    for split in ("test", "val", "train"):
        mode = "lm" if split == "train" else "gen"
        data = DartDataset(tokenizer, split=split, mode=mode, num_parallel_workers=args.workers)


if __name__ == "__main__":
    main()
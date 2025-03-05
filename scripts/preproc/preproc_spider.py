
import argparse

from dataset.spider import SpiderDataset
from modules import _load_mamba_tokenizer, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--split", nargs="+", default=("val", "train"))
    args = parser.parse_args()

    tokenizer = _load_mamba_tokenizer()

    # for split in args.split:
    #     data = SpiderDataset(tokenizer, split, hardness=["hard", "extra"], num_parallel_workers=args.workers)
    #     print(len(data))

    for split in args.split:
        data = SpiderDataset(tokenizer, split, num_parallel_workers=args.workers, has_test_split=False)


if __name__ == "__main__":
    main()

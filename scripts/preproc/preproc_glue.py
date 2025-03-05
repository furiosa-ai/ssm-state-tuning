
import argparse

from dataset.glue import GlueDataset, task_to_keys
from modules import _load_mamba_tokenizer, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", nargs="+", default=["all"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--split", nargs="+", default=["train", "val"])  # , "test"
    args = parser.parse_args()

    if args.name[0] == "all":
        names = ["rte", "mrpc", "cola", "sst2", "qnli", "qqp", "mnli"]
        # names = ["mnli"]
    else:
        names = args.name

    for name in names:
        tokenizer = _load_mamba_tokenizer()
        for split in args.split:
            # try:
            data = GlueDataset(tokenizer, name, split, num_parallel_workers=args.workers, subset_size=args.subset_size, has_test_split=False)
            # except Exception as e:
            #     print(name, split, "failed")
            #     print(e)


if __name__ == "__main__":
    main()

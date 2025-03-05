

import numpy as np
from dataset.dart import DartDataModule
from dataset.glue import GlueDataModule
from dataset.random_data import RandomDataModule
from dataset.samsum import SamSumDataModule
from dataset.spider import SpiderDataModule


def load_dataset(data, tokenizer, split, return_module=False, **kwargs):
    if data.startswith("glue"):
        glue, name, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = GlueDataModule(
            tokenizer=tokenizer,
            name=name,
            split=split,
            subset_size=subset_size,
            has_test_split=glue.endswith("-tvt"),
            **kwargs
        )
    elif data.startswith("dart"):
        alpaca, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = DartDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )   
    elif data.startswith("random"):
        _, seqlen, *size = data.split("_")
        seqlen = int(seqlen)

        if len(size) > 0:
            size = int(size[0])
        else:
            size = None

        data_module = RandomDataModule(
            tokenizer=tokenizer,
            split=split,
            seqlen=seqlen,
            size=size,
            **kwargs
        )
    elif data.startswith("spider"):
        hardness = None
        if data.endswith("_hard_extra"):
            data = data[:-len("_hard_extra")]
            hardness = ["hard", "extra"]

        spider, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = SpiderDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            hardness=hardness,
            has_test_split=spider.endswith("-tvt"),
            **kwargs
        )
    elif data.startswith("samsum"):
        samsum, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split in ("val", "test"):
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = SamSumDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )
    else:
        raise Exception(data)
    
    if not return_module:
        data_module = data_module.dataset

    return data_module

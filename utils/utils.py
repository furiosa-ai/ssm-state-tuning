import os
from pathlib import Path
import numpy as np
from torch import nn
import torch


def flatten_dict(dic, sep="_", prefix=""):
    out = {}

    for k, v in dic.items():
        if isinstance(v, dict):
            out.update(flatten_dict(v, sep, prefix + k + sep))
        else:
            out[prefix + k] = v

    return out


def find_module_parent(model: nn.Module, child: nn.Module, n_ancestor=1):
    if n_ancestor > 1:
        return find_module_parent(model, find_module_parent(model, child, n_ancestor-1))

    for c in model.children():
        if c is child:
            return model
        else:
            res = find_module_parent(c, child)

            if res is not None:
                return res

    return None


def dump_mask(mask, name):
    from PIL import Image
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    filename = Path(f"out/{os.getpid()}/{name}.png")
    filename.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray(mask).save(filename)

def get_tokenizer_cache_prefix(tokenizer):
    return ""
    

def create_non_existent_file(filename):
    filename = Path(filename)

    for i in range(100):
        filename_out = filename.parent / (filename.stem + f"_{i}" + filename.suffix)
        if not os.path.exists(filename_out):
            Path(filename_out).touch()
            return filename_out

    return filename_out
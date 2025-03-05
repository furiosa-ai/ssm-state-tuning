
import os
from mamba_ssm.modules.mamba_simple import Mamba

from .mamba_peft import MambaPeft
from .mixer_seq_simple import MambaLMHeadModelPeft

import torch
import json

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from peft import PeftModelForSeq2SeqLM

from peft import get_peft_model, PeftConfig, PeftType


def get_checkpoints(path, return_dict=False, local_only=False):
    def _get_it(file):
        try:
            return int(Path(file).stem.split("-")[1])
        except ValueError:
            return 0

    if not Path(path).exists():
        checkpoints = [path]
    else:
        path = Path(path)
        checkpoints = list(path.glob("checkpoint-*"))

        if len(checkpoints) > 0:
            checkpoints = sorted(checkpoints, key=_get_it)
        else:
            checkpoints = [path]

    if local_only:
        assert all(((c / "model.pt").is_file() or (c / "peft.pt").is_file()) for c in checkpoints)

    if return_dict:
        checkpoints = {_get_it(c): str(c) for c in checkpoints}

    return checkpoints


def apply_mamba_fixes(model):
    from torch import nn
    from modules.mamba_peft_utils import ParameterProcessor
    from modules.mamba_peft import MultiLinearLayer
    
    dtype = torch.bfloat16

    for name, module in model.named_modules():
        if isinstance(module, MambaPeft):
            if not hasattr(module, "parameter_processors"):
                module.parameter_processors = nn.ModuleDict({
                    "A_log": ParameterProcessor(None, None, None, None),
                    "B": ParameterProcessor(None, None, None, None),
                    "C": ParameterProcessor(None, None, None, None),
                    "D": ParameterProcessor(None, None, None, None),
                    "dt": ParameterProcessor(None, None, None, None),
                })
            
            if "A" not in module.parameter_processors:
                module.parameter_processors["A"] = ParameterProcessor(None, None, None, None)

            if "x_after_conv" not in module.parameter_processors:
                module.parameter_processors["x_after_conv"] = ParameterProcessor(None, None, None, None)
        elif isinstance(module, MultiLinearLayer):
            if not hasattr(module, "cat_output"):
                module.cat_output = True

    return model


def load_mamba_full(pretrained, fuse_peft=False, apply_fixes=True, cls=MambaLMHeadModel, force_4bit=False, **kwargs):
    pretrained = get_checkpoints(pretrained)[-1]

    model_kwargs = kwargs

    trainable_params = 1

    if (Path(pretrained) / "model.pt").exists():
        model = torch.load(Path(pretrained) / "model.pt")

        if isinstance(model, dict):
            model = model["model"]

        dtype = next(iter(model.parameters())).dtype

        if dtype != model_kwargs.get("dtype", dtype):
            print(f'Moving model to {model_kwargs["dtype"]}')
            model = model.to(model_kwargs["dtype"])
            assert next(iter(model.parameters())).dtype == model_kwargs["dtype"]

        if hasattr(model, "get_nb_trainable_parameters"):
            trainable, all_params = model.get_nb_trainable_parameters()
            trainable_params = trainable / all_params
        else:
            trainable_params = 1

        if fuse_peft:
            if isinstance(model, PeftModelForSeq2SeqLM):
                model = model.merge_and_unload()

            if isinstance(model, MambaLMHeadModelPeft):
                try:
                    model.combine_layers()
                except AttributeError:
                    print("no method combine_layers")

        if apply_fixes:
            model = apply_mamba_fixes(model)

        tokenizer = _load_mamba_tokenizer()
    else:
        # if (pretrained / "pytorch_model.bin").exists():
        model = cls.from_pretrained(str(pretrained), **model_kwargs)
        tokenizer = _load_mamba_tokenizer()

        model.model_args = {
            "pretrained": pretrained,
            "cls": cls,
            **kwargs,
        }

    info = {
        "trainable_params": trainable_params
    }

    return {
        "model": model, 
        "tokenizer": tokenizer,
        "info": info
    }


def load_mamba_peft(path):
    path = Path(path)

    ckpt = torch.load(path / "peft.pt")
    train_state_dict = ckpt["state_dict"]
    model_args = ckpt["model_args"]
    peft_args = ckpt["peft_args"]

    model_tokenizer = load_mamba_full(**model_args, apply_fixes=False)
    model = model_tokenizer["model"]
    tokenizer = model_tokenizer["tokenizer"]
    model, _ = get_mamba_peft_model(model, return_peft_cfg=True, no_print=True, **peft_args)
    missing_keys, unexpected_keys = model.load_state_dict(train_state_dict, strict=False)

    buffers = set(n for n, _ in model.named_buffers())

    assert len(unexpected_keys) == 0
    missing_keys = set(missing_keys)
    for n, p in model.named_parameters():
        if p.requires_grad or n in buffers:
            assert n not in missing_keys
        else:
            assert n in missing_keys

    return {
        "model": model, 
        "tokenizer": tokenizer,
    }


def save_mamba_peft(model, path):
    trainable_param_names = set(n for n, p in model.named_parameters() if p.requires_grad)
    buffers = set(n for n, _ in model.named_buffers())

    def _is_save_param(name, param):
        if name in trainable_param_names or name in buffers:
            return True
        
        return False

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    state_dict_peft = {n: p for n, p in state_dict.items() if _is_save_param(n, p)}
    torch.save({
        "model_args": model.model_args,
        "peft_args": model.peft_args,
        "state_dict": state_dict_peft,
    }, path / "peft.pt.temp")
    os.rename(path / "peft.pt.temp", path / "peft.pt")


def save_mamba(model, path):
    if hasattr(model, "peft_args"):
        save_mamba_peft(model, path)
    else:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(model, path / "model.pt")

def load_mamba(path, **kwargs):
    if (Path(path) / "peft.pt").exists():
        return load_mamba_peft(path)
    else:
        return load_mamba_full(path, **kwargs)


def load_tokenizer(pretrained):
    tokenizer = _load_mamba_tokenizer()
    return tokenizer


def _load_mamba_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = "###"
    tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template
    return tokenizer


def print_trainable_parameter_names(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def get_mamba_peft_model(model, peft, return_peft_cfg=False, train_embedding=False, no_print=False):
    model_args = getattr(model, "model_args", {})
    peft_args = peft

    if hasattr(model, "split_layers"):
        model.split_layers()
    else:
        print("no split_layers")

    if isinstance(peft, (str, Path)):
        with open(peft, "r") as f:
            peft = json.load(f)

    if isinstance(peft, list):
        peft = {

            "peft_type": "MULTI_PEFT",
            "configs": peft
        }

    if isinstance(peft, dict):
        peft = PeftConfig.from_peft_type(**peft)

    model = get_peft_model(model, peft)

    if train_embedding:
        model.model.word_embeddings.weight.requires_grad = True

    if not no_print:
        print_trainable_parameter_names(model)

    model.model_args = model_args
    model.peft_args = {
        "peft": peft
    }

    if return_peft_cfg:
        return model, peft
    return model


def get_trainable_parameters_ratio(model):
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, all_params = model.get_nb_trainable_parameters()
        trainable_params = trainable / all_params
    else:
        trainable_params = 1

    return trainable_params

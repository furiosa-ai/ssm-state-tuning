from dataclasses import dataclass, field
from functools import partial
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

import wandb
from mamba_ssm.modules.mamba_simple import Mamba
import torch
import argparse
import numpy as np
from torch import nn

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments, GPTNeoXTokenizerFast
import yaml
from peft import PeftConfig, PeftModelForSeq2SeqLM, LoraModel
from modules import MambaPeft, MambaLMHeadModelPeft, get_mamba_peft_model, get_trainable_parameters_ratio, load_mamba, print_trainable_parameter_names
from modules.generation import create_generator
from modules.mamba_peft_utils import set_peft_params_trainable
from dataset import load_dataset
from trainer.mamba_trainer import MambaTrainer, MambaTrainingArguments

import tensorboard

from utils.utils import create_non_existent_file




def init_embedding(model, tokenizer):
    model.backbone.embedding = nn.Embedding(
        tokenizer.vocab_size, 
        model.backbone.embedding.embedding_dim, 
        device=model.backbone.embedding.weight.device,
        dtype=model.backbone.embedding.weight.dtype,
    )
    model.lm_head = nn.Linear(
        model.backbone.embedding.embedding_dim, 
        tokenizer.vocab_size, 
        device=model.backbone.embedding.weight.device,
        dtype=model.backbone.embedding.weight.dtype,
        bias=False
    )
    model.tie_weights()


def run_train(args):
    if args.overwrite and args.sdt:
        assert Path(args.output_dir).exists()

    if not args.overwrite:
        if (Path(args.output_dir) / "cfg.yaml").exists():
            if args.resume:
                resume_from_checkpoint = True
            else:
                assert False, str(Path(args.output_dir) / "cfg.yaml") + " exists!"
                resume_from_checkpoint = False
        else:
            resume_from_checkpoint = False
    else:
        resume_from_checkpoint = False

    assert args.data.startswith("glue_") or args.data in ("glue_rte", "glue_mrpc", "glue_cola", "spider_1000")  or not (args.no_save and args.num_epochs > 1), "don't train for more than one epoch without saving ckpts!"

    is_custom_tokenizer = False
    # is_custom_tokenizer = args.tokenizer != "EleutherAI/gpt-neox-20b"
    # tokenizer = get_tokenizer(args.tokenizer)

    model_kwargs = dict(
        dtype={"bf16": torch.bfloat16, "fp16": torch.bfloat16, "fp32": torch.float32}[args.prec], 
        device="cuda",
        use_fast_path=False,
        mamba_cls=Mamba if args.peft is None else MambaPeft,
        backend=args.backend,
    )

    # model = load_mamba(args.model, **model_kwargs)
    model_tokenizer = load_mamba(
        args.model, 
        cls=MambaLMHeadModelPeft,
        **model_kwargs
    )
    model, tokenizer = model_tokenizer["model"], model_tokenizer["tokenizer"]

    if args.from_scratch:
        model = MambaLMHeadModelPeft(model.config, **model_kwargs)

    if is_custom_tokenizer:
        print(f"Resizing, randomly initializing and unfreezing embedding layer for custom tokenizer")
        init_embedding(model, tokenizer)

    if args.peft is not None:
        model, peft_cfg = get_mamba_peft_model(model, args.peft, return_peft_cfg=True, train_embedding=is_custom_tokenizer, no_print=True)
    else:
        peft_cfg = None

    if args.train_all_peft:
        lora = model.base_model.model.base_model
        assert isinstance(lora, LoraModel)
        set_peft_params_trainable(model, lora.prefix, enable_train=True, disable_train=False)

    print_trainable_parameter_names(model)

    print("Loaded model")

    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # tokenizer.eos_token = "<|endoftext|>"
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

    train_data_module = load_dataset(args.data, tokenizer, "train", return_module=True)

    # save_steps = its_per_epoch
    dataloader_num_workers = args.num_data_workers

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(args.output_dir) / "cfg.yaml", "w") as f:
        yaml.dump(vars(args), f)

    if args.eval_gen is not None:
        eval_generator = create_generator(tokenizer, **args.eval_gen)
    else:
        eval_generator = None
    
    val_data_module = load_dataset(
        args.val_data if args.val_data is not None else args.data, 
        tokenizer, 
        args.val_data_split, 
        mode="lm" if args.eval_gen is None else "gen",
        return_module=True)

    compute_metrics = val_data_module.dataset.compute_metrics

    if args.debug:
        eval_type = val_data_module.dataset.eval_type
        eval_do_concat_batches = val_data_module.dataset.eval_do_concat_batches
        preprocess_logits_for_metrics = val_data_module.dataset.preprocess_logits_for_metrics

        train_data_module.dataset = torch.utils.data.Subset(train_data_module.dataset, range(8))
        val_data_module.dataset = torch.utils.data.Subset(val_data_module.dataset, range(2))
        val_data_module.dataset.eval_type = eval_type
        val_data_module.dataset.eval_do_concat_batches = eval_do_concat_batches
        val_data_module.dataset.preprocess_logits_for_metrics = preprocess_logits_for_metrics

        args.num_epochs = 1

    its_per_epoch = int(np.ceil(len(train_data_module.dataset) / args.batch_size))
    logging_steps = min(50, its_per_epoch)

    run_name = str(args.output_dir).replace("weights/", "")
    # wandb.init()  # project='my_research'
    # wandb.run.name = run_name

    print("Dropping last batch")
    trainer = MambaTrainer(
        model=model,
        train_dataset=train_data_module.dataset,
        tokenizer=tokenizer,
        args=MambaTrainingArguments(
            learning_rate=args.learning_rate,
            # num_train_epochs=args.num_epochs,
            max_steps=int(args.num_epochs * its_per_epoch),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output_dir,
            logging_steps=logging_steps,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=2,
            eval_accumulation_steps=128,
            info={
                "trainable_params": get_trainable_parameters_ratio(model),
                # "peft_cfg": peft_cfg.to_dict() if peft_cfg is not None else None,
                "cfg_path": args.cfg_path,
                "device": os.environ.get("CUDA_VISIBLE_DEVICES", None),
            },
            save_strategy="steps" if not args.no_save else "no",
            evaluation_strategy="steps" if not args.skip_eval else "no",
            save_steps=int(args.eval_epochs * its_per_epoch),
            eval_steps=int(args.eval_epochs * its_per_epoch),
            dataloader_drop_last=val_data_module.dataset.eval_type != "log_likelihood", # only ll works for batch
            seed=args.seed,
            run_name=run_name,
            report_to="wandb",  # disable tensorboard
        ),
        compute_metrics=compute_metrics,
        data_collator=train_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        eval_generator=eval_generator,
        min_eval_metric_after_epoch=args.min_eval_metric_after_epoch,
        skip_metrics=args.skip_metrics,
        log_speed=args.log_speed
    )

    trainer.preprocess_logits_for_metrics = val_data_module.dataset.preprocess_logits_for_metrics

    # resume_from_checkpoint is bugged
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # trainer.evaluate()

    if args.log_speed:
        with open(create_non_existent_file(Path(args.output_dir) / "train_timestamps.yaml"), "w") as f:
            yaml.safe_dump(trainer.train_timestamps, f)


def get_output_path_for_cfg(cfg_path):
    output_dir = str(Path(cfg_path).parent / Path(cfg_path).stem)
    output_dir = output_dir.replace("cfg/exps/", "")
    output_dir = Path("weights", output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sdt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--prec")
    parser.add_argument("--device")
    parser.add_argument("--log_speed", action="store_true")
    args = parser.parse_args()

    if args.device is not None:
        os.environ["VISIBLE_DEVICES"] = args.device

    dft_cfg = {
        "tokenizer": "EleutherAI/gpt-neox-20b",
        "learning_rate": 5e-5,
        "batch_size": 4,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optim": "adamw_torch",
        "peft": None,
        "from_scratch": False,
        "skip_eval": False,
        "eval_epochs": 1,
        "val_data": None,
        "val_data_split": "val",
        "no_save": False,
        "backend": "cuda",
        "num_data_workers": 8,
        "model_transform": None,
        "repeat": None,
        "eval_gen": None,
        "min_eval_metric_after_epoch": None,
        "train_all_peft": False,
        "skip_metrics": False,
        "seed": 42,
        "log_speed": False,
    }

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = get_output_path_for_cfg(args.cfg)
    args_dict = vars(args)

    if args_dict["model"] is None:
        del args_dict["model"]

    if args_dict["prec"] is None:
        del args_dict["prec"]

    args = {**dft_cfg, **cfg, **args_dict, "output_dir": str(output_dir), "cfg_path": args.cfg}

    if args["repeat"] is None:
        run_train(SimpleNamespace(**args))
    else:
        for i in range(args["repeat"]):
            print(f"Starting run {i}")
            args["output_dir"] = str(Path(output_dir) / f"{i:03d}")
            run_train(SimpleNamespace(**args))


if __name__ == "__main__":
    main()

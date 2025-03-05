from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from transformers.trainer import logger
from transformers.trainer_utils import denumpify_detensorize
import torch
import numpy as np
from torch import nn
import os
from tqdm import tqdm
import wandb
import yaml
from yaml import CSafeLoader
from peft import PeftModel

from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from modules import load_mamba, save_mamba
from trainer.loss import CrossEntropy, Accuracy

import torch.nn.functional as F



class MambaEvalPrediction:
    def __init__(self, tokenizer=None, input_ids=None, pred_ids=None, label_ids=None, save_file=None, remove_eos=False):
        self.tokenizer = tokenizer

        self.inputs = tokenizer.batch_decode(self.remove_pad_token_id(input_ids) if remove_eos else input_ids) if input_ids is not None else None
        self.preds = tokenizer.batch_decode(self.remove_eos_token_id(pred_ids) if remove_eos else pred_ids) if pred_ids is not None else None
        self.labels = tokenizer.batch_decode(self.remove_eos_token_id(label_ids) if remove_eos else label_ids) if label_ids is not None else None

        self.input_ids = [t.cpu().numpy() for t in input_ids] if input_ids is not None else None
        self.pred_ids = [t.cpu().numpy() for t in pred_ids] if pred_ids is not None else None
        self.label_ids = [t.cpu().numpy() for t in label_ids] if label_ids is not None else None

        self.save_file = save_file

    def remove_pad_token_id(self, ids):
        ids_no_eos = [(id if id[-1] != self.tokenizer.pad_token_id else id[:-1])  for id in ids]
        return ids_no_eos

    def remove_eos_token_id(self, ids):
        eos_token_id = self.tokenizer.eos_token_id

        ids_no_eos = [(id if id[-1] != eos_token_id else id[:-1])  for id in ids]
        return ids_no_eos

    @staticmethod
    def from_file(path):
        p = MambaEvalPrediction()
        p.load(path)
        return p

    def load(self, path):
        with open(path, "r") as f:
            state = yaml.load(f, Loader=CSafeLoader)

        self.inputs = state["inputs"]
        self.preds = state["preds"]
        self.labels = state["labels"]
        self.input_ids = [np.array(x) for x in state["input_ids"]]
        self.pred_ids = [np.array(x) for x in state["pred_ids"]]
        self.label_ids = [np.array(x) for x in state["label_ids"]]
        self.save_file = path

    def save(self, path=None):
        if path is None:
            path = self.save_file

        out_dict = dict(
            inputs=self.inputs,
            preds=self.preds,
            labels=self.labels,
            input_ids=[t.astype(int).tolist() for t in self.input_ids],
            pred_ids=[t.astype(int).tolist() for t in self.pred_ids],
            label_ids=[t.astype(int).tolist() for t in self.label_ids],
        )

        Path(path).parent.mkdir(exist_ok=True, parents=True)

        with open(path, "w") as f:
            yaml.safe_dump(out_dict, f, sort_keys=False)


class MambaLogLikelihoodPrediction:
    def __init__(self, input_ids, pred_lls, label_ids) -> None:
        self.input_ids = input_ids
        self.pred_lls = pred_lls
        self.label_ids = label_ids


class TrainLossEarlyStop:
    def __init__(self) -> None:
        self.nan_limit = 10
        self.consec_nans = 0
        self.should_stop = False

    def __call__(self, control, train_loss) -> Any:
        train_loss = train_loss.item()

        if np.isnan(train_loss) or train_loss <= 1.e-6:
            self.consec_nans += 1

            if self.consec_nans >= self.nan_limit:
                print(f"Stopping after {self.consec_nans} 0/nan losses")
                self.should_stop = True
                control.should_training_stop = True
        else:
            self.consec_nans = 0


class BadEvalEarlyStop:
    def __init__(self, eval_after_epochs, metric=None):
        self.eval_after_epochs = eval_after_epochs
        self.metric = metric

    def __call__(self, control, metrics) -> Any:
        epoch = int(metrics["epoch"])

        if epoch in self.eval_after_epochs:
            metric = self.metric if self.metric is not None else next(iter(metrics.keys()))
            min_val = self.eval_after_epochs[epoch]
            val = metrics[metric]

            if val < min_val:
                control.should_training_stop = True


@dataclass
class MambaTrainingArguments(TrainingArguments):
    info: Dict[str, Any] = field(default=None)


class MambaTrainer(Trainer):
    def __init__(self, 
                 model: PreTrainedModel | Module = None, 
                 args: TrainingArguments = None, 
                 data_collator: Any | None = None, 
                 train_dataset: Dataset | None = None, 
                 eval_dataset: Dataset | Dict[str, Dataset] | None = None, 
                 tokenizer: PreTrainedTokenizerBase | None = None, 
                 model_init: Callable[[], PreTrainedModel] | None = None, 
                 compute_metrics: Callable[[EvalPrediction], Dict] | None = None, 
                 callbacks: List[TrainerCallback] | None = None, 
                 optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                 eval_generator=None,
                 min_eval_metric_after_epoch=None,
                 log_speed=False,
                 skip_metrics=False):
        # args.include_inputs_for_metrics = True
        if callbacks is None:
            callbacks = []

        args.eval_do_concat_batches = eval_dataset.eval_do_concat_batches if eval_dataset is not None else None

        """
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        """

        super_extra_kwargs = {}

        if "processing_class" in Trainer.__init__.__code__.co_names:  
            # for new transformers versions  
            super_extra_kwargs["processing_class"] = tokenizer
        else:
            super_extra_kwargs["tokenizer"] = tokenizer

        super().__init__(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset, 
                         eval_dataset=eval_dataset, 
                         model_init=model_init, compute_metrics=compute_metrics, callbacks=callbacks, 
                         optimizers=optimizers, preprocess_logits_for_metrics=preprocess_logits_for_metrics, **super_extra_kwargs)
        
        self.train_crit = CrossEntropy()
        self.val_crits = [Accuracy()]
        self.train_loss_early_stop = TrainLossEarlyStop()
        self.eval_generator = eval_generator
        self.min_eval_metric_after_epoch_early_stop = BadEvalEarlyStop(min_eval_metric_after_epoch) if min_eval_metric_after_epoch is not None else None
        self.skip_metrics = skip_metrics
        self.run_name = args.run_name
        self.wandb_init = False
        self.train_timestamps = [] if log_speed else None

        if hasattr(model, "load_config"):
            model.load_config(self.args.output_dir)

    def log(self, *args, **kwargs):
        super().log(*args, **kwargs)

        if not self.wandb_init and not self.args.report_to == []:
            wandb.run.name = self.run_name
            self.wandb_init = True

    def log_train_seq(self, input_ids, label_ids, lm_logits, idx=0):
        input_ids, label_ids, lm_logits = input_ids[idx], label_ids[idx], lm_logits[idx]

        output_ids = lm_logits.argmax(-1)

        valid_ids = label_ids != -100

        input_txt = self.tokenizer.decode(input_ids)
        input_txt_valid = self.tokenizer.decode(input_ids[valid_ids])
        label_txt_valid = self.tokenizer.decode(label_ids[valid_ids])
        output_txt_valid = self.tokenizer.decode(output_ids[valid_ids])

        print(input_txt)
        print(input_txt_valid, "->", label_txt_valid)
        print(output_txt_valid, "==", label_txt_valid)

    def _forward(self, model, inputs):
        if not self.wandb_init and wandb.run is not None:
            wandb.run.name = self.run_name
            self.wandb_init = True

        input_ids = inputs["input_ids"]
        label_ids = inputs["label_ids"]

        add_inputs = {}

        if isinstance(model, PeftModel):
            base = model.base_model

            # if "label_ids" in base.forward.__code__.co_varnames:
            #     add_inputs["label_ids"] = label_ids

        lm_logits = model(input_ids, **add_inputs).logits

        return input_ids, label_ids, lm_logits

    def log_iter(self, metrics, interval):
        if (self.state.global_step + 1) % interval == 0:
            self.log(metrics)

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        input_ids, label_ids, lm_logits = self._forward(model, inputs)
        lm_loss = self.train_crit(lm_logits, label_ids)

        if hasattr(model, "compute_reg_loss"):
            reg_loss, reg_loss_dict = model.compute_reg_loss()
            if len(reg_loss_dict) > 0:
                self.log_iter({"loss": lm_loss.item(), "reg_loss": reg_loss.item(), **reg_loss_dict}, 10)
        else:
            reg_loss, reg_loss_dict = 0, {}

        lm_loss = lm_loss + reg_loss

        # if getattr(model, "should_reset_optimizer", False):
        #     self.reset_optimizer()

        if getattr(model, "should_training_stop", False):
            if hasattr(model, "save_config"):
                model.save_config(self.args.output_dir)
                self.control.should_training_stop = True

        # from modules.sdt import SDTReg
        # lm_loss = lm_loss + SDTReg()(model)

        if False:
            self.log_train_seq(input_ids, label_ids, lm_logits)

        self.train_loss_early_stop(self.control, lm_loss)

        if self.train_timestamps is not None:
            self.train_timestamps.append({
                "time": time.time(),
                "mem": torch.cuda.memory_allocated(),
            })
            # if len(self.train_timestamps) > 1:
            #     print(f"delta: {self.train_timestamps[-1] - self.train_timestamps[-2]}")

        return lm_loss
    
    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        input_ids, label_ids, lm_logits = self._forward(model, inputs)
        lm_loss = self.train_crit(lm_logits, label_ids)

        logits_valid = []
        label_ids_valid = []
        for i, (logits_sample, label_ids_sample) in enumerate(zip(lm_logits, label_ids)):
            valid_pos = label_ids_sample != self.train_crit.ignore_index

            logits_sample_valid = logits_sample[valid_pos]  # .argmax(-1)
            label_ids_sample_valid = label_ids_sample[valid_pos]

            logits_valid.append(logits_sample_valid)
            label_ids_valid.append(label_ids_sample_valid)

        return (lm_loss, logits_valid, label_ids_valid)
        
    def generation_step(self, generator, model, inputs):
        input_ids, label_ids = inputs["input_ids"], inputs["label_ids"]
        out_seq = generator(model, input_ids)
        output_ids = out_seq
        return (output_ids, label_ids)

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # torch.save(self.model, f"{output_dir}/model.pt")


        # try:
        #     torch.save(self.model, f"{output_dir}/model.pt")
        # except Exception as e:
        #     print(f"Failed saving model", e)
        save_mamba(self.model, output_dir)

        # try:
        #     save_mamba(self.model, output_dir)
        # except Exception as e:
        #     print(f"Failed saving peft", e)
        #     torch.save(self.model, f"{output_dir}/model.pt")

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.train_loss_early_stop.should_stop:
            self.control.should_evaluate = False

        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
    
    def load_model(self, path):
        self.model = load_mamba(path)["model"]

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        assert model is None

        self.model.load_state_dict(load_mamba(resume_from_checkpoint)["model"].state_dict())
        logger.info(f"Loading model from {resume_from_checkpoint}.")

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ):
        return data_collator

    def reset_optimizer(self):
        print("Resetting optimzer")
        self.optimizer = None
        self.lr_scheduler = None
        self.create_optimizer_and_scheduler(self.args.max_steps - self.state.global_step)

    def create_optimizer(self):
        if hasattr(self.model, "create_optimizer"):
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = self.model.create_optimizer(self.model, optimizer_cls, optimizer_kwargs)
            if self.optimizer is not None:
                return self.optimizer
            else:
                return super().create_optimizer()
        else:
            return super().create_optimizer()

    def _evaluate_default(self, eval_dataset, ignore_keys, metric_key_prefix):
        data = self.eval_dataset if eval_dataset is None else eval_dataset
        if data is not None:
            data.save_pred_file = str(Path(self.args.output_dir) / f"predictions-{self.state.global_step}.dat")
        
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if data is not None:
            data.save_pred_file = None
        
        return metrics

    def evaluate(self, eval_dataset: Dataset | Dict[str, Dataset] | None = None, ignore_keys: List[str] | None = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        if self.eval_generator is not None:
            metrics = self.evaluate_generation(self.eval_generator, metric_key_prefix=metric_key_prefix)
        elif self.get_eval_dataloader().dataset.eval_type == "log_likelihood":
            metrics = self.evaluate_log_likelihood()
        else:
            metrics = self._evaluate_default(eval_dataset, ignore_keys, metric_key_prefix)

        if self.min_eval_metric_after_epoch_early_stop is not None:
            self.min_eval_metric_after_epoch_early_stop(self.control, metrics)

        return metrics
    
    @torch.no_grad()
    def evaluate_log_likelihood(self, metric_key_prefix="eval"):
        dataloader = self.get_eval_dataloader()

        model = self.model
        model.eval()

        input_ids_all = []
        pred_lls_all = []
        label_ids_all = []

        for step, inputs in enumerate(tqdm(dataloader, desc="Evaluate")):
            batch = self._forward(model, inputs)
            for input_ids, label_ids, lm_logits in zip(*batch):
                mask = label_ids != dataloader.dataset.ignore_index

                # assert lm_logits.ndim == 2
                label_ids = label_ids[mask]
                ll_all = F.log_softmax(lm_logits[mask], 1)

                # Obtain log-probs at the corresponding continuation token indices
                # pred_lls = ll_all[range(label_ids.shape[0]), label_ids]  # select gt tokens
                pred_lls = torch.gather(ll_all, 1, label_ids.unsqueeze(-1)).squeeze(-1)  # select gt tokens

                input_ids_all.append(input_ids.cpu())
                pred_lls_all.append(pred_lls.cpu())
                label_ids_all.append(label_ids.cpu())

        eval_pred = MambaLogLikelihoodPrediction(input_ids_all, pred_lls_all, label_ids_all)
        metrics = self.compute_metrics(eval_pred)

        if metric_key_prefix != "":
            metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics

    @torch.no_grad()
    def evaluate_generation(self, generator, use_cache=True, skip_metrics=None, metric_key_prefix="eval", pred_out_file=None):
        if skip_metrics is None:
            skip_metrics = self.skip_metrics

        if pred_out_file is not None:
            eval_pred_file = pred_out_file
        else:
            eval_pred_file = Path(self.args.output_dir) / f"predictions-{self.state.global_step}.yaml"

        if not use_cache or not eval_pred_file.is_file():
            model = self.model
            model.eval()

            dataloader = self.get_eval_dataloader()

            input_ids_all = []
            pred_ids_all = []
            label_ids_all = []

            for step, inputs in enumerate(tqdm(dataloader, desc="Evaluate")):
                pred_ids, label_ids = self.generation_step(generator, model, inputs)
                input_ids_all += [*inputs["input_ids"]]
                pred_ids_all += [*pred_ids]
                label_ids_all += [*label_ids]

            eval_pred = MambaEvalPrediction(generator.tokenizer, input_ids_all, pred_ids_all, label_ids_all, 
                                            save_file=eval_pred_file, remove_eos=True)
            eval_pred.save(pred_out_file)
        else:
            if not skip_metrics:
                print(f"Loading prediction {eval_pred_file}")

        if not skip_metrics:
            eval_pred = MambaEvalPrediction.from_file(eval_pred_file)
            metrics = self.compute_metrics(eval_pred)

            if metric_key_prefix != "":
                metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

            return metrics
        else:
            return None

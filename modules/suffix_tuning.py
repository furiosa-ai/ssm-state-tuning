
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import enum
import math
from typing import Dict, List, Optional, Union
from peft.config import PeftConfig
import torch
import torch.nn.functional as F
from torch import nn
from einops import einsum, repeat, rearrange

from peft.tuners.tuners_utils import BaseTunerLayer

from modules.mamba_peft_utils import MambaPeftType, register_peft_config, register_peft_tuner
from modules.mamba_tuner_utils import MambaBaseTuner



class SuffixTuningBiasType(str, enum.Enum):
    SILU_Z = "SILU_Z"
    SILU_Z_C = "SILU_Z_C"


class SuffixTuningBiasInit(str, enum.Enum):
    ZERO = "ZERO"
    RANDOM = "RANDOM"




class LoraParam(nn.Module):
    def __init__(self, d1, d2, r, dropout, ratio, device, dtype):
        super().__init__()

        if dropout is None:
            dropout = 0

        if ratio is not None:
            assert d1 % ratio == 0
            d1 = d1 // ratio
            d2 = d2 * ratio

        self.ratio = ratio
        self.lora_A = nn.Parameter(torch.zeros(r, d2, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d1, r, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self):
        out = self.lora_B @ self.lora_A

        if hasattr(self, "ratio") and self.ratio is not None:
            d1, d2 = out.shape
            out = out.reshape(d1 * self.ratio, d2 // self.ratio)

        return out


@register_peft_config(MambaPeftType.SUFFIX_TUNING)
@dataclass
class SuffixTuningConfig(PeftConfig):
    bias_init: str = field(default=None)
    bias_type: str = field(default=None)
    finetune_parameters: List[str] = field(default=None)
    C_scale_shape: str = field(default=None)
    C_bias_shape: str = field(default=None)
    r: int = field(default=None, metadata={"help": "Lora attention dimension"})
    r_ratio: int = field(default=None)
    dropout: float = field(default=None)

    def __post_init__(self):
        self.peft_type = MambaPeftType.SUFFIX_TUNING
        self.bias_init = SuffixTuningBiasInit(self.bias_init)
        self.bias_type = SuffixTuningBiasType(self.bias_type)


@register_peft_tuner(MambaPeftType.SUFFIX_TUNING)
class SuffixTuningModel(MambaBaseTuner):
    prefix: str = "suffixtuning_"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        super().__init__(model, peft_config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        peft_config.target_modules = ["x_after_ssm"]

        if peft_config.C_scale_shape is not None or peft_config.C_bias_shape is not None:
            peft_config.target_modules += ["C"]

        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        finetune_parameters = self.peft_config[self.active_adapter].finetune_parameters

        if finetune_parameters is None:
            finetune_parameters = []

        for n, p in model.named_parameters():
            if self.prefix in n or any(n.endswith("." + fp) for fp in finetune_parameters):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _create_new_module(self, peft_config, adapter_name, target, target_name):
        new_module = None

        if target_name == "x_after_ssm":
            new_module = SuffixTuningBiasProcessor(
                target, adapter_name,
                bias_type=peft_config.bias_type, 
                bias_init=peft_config.bias_init, 
                r=peft_config.r, dropout=peft_config.dropout, r_ratio=peft_config.r_ratio
            )

        return new_module


class SuffixTuningBiasProcessor(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer, adapter_name, r=None, **kwargs) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer

        self.suffixtuning_bias = nn.ParameterDict({}) if r is None else nn.ModuleDict({})
        self.suffixtuning_type = {}

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r=r,
            **kwargs
        )

    def _create_param(self, shape, init_type, dtype, device, r=None, dropout=None, r_ratio=None):
        if r is not None:
            d, n, = shape
            return LoraParam(d, n, r, dropout, ratio=r_ratio, dtype=dtype, device=device)
        else:
            match init_type:
                case SuffixTuningBiasInit.RANDOM:
                    data = torch.randn(shape, dtype=dtype, device=device) * 0.1
                case SuffixTuningBiasInit.ZERO:
                    data = torch.zeros(shape, dtype=dtype, device=device)

            return nn.Parameter(data)

    def update_layer(self, adapter_name, bias_type, bias_init, r=None, dropout=None, r_ratio=None):
        dims = self.base_layer.all_dims

        # xor
        shape = {
            SuffixTuningBiasType.SILU_Z_C: [dims["d"], dims["n"]],
            SuffixTuningBiasType.SILU_Z: [dims["d"]],
        }[bias_type]

        param = self._create_param(shape, bias_init, self.base_layer.dtype, self.base_layer.device,
                                   r=r, dropout=dropout, r_ratio=r_ratio)

        self.suffixtuning_bias[adapter_name] = param
        self.suffixtuning_type[adapter_name] = bias_type

        self.set_adapter(self.active_adapters)

    def forward(self, x, z, A, B, C, D, dt):
        y = x

        for active_adapter in self.active_adapters:
            param = self.suffixtuning_bias[active_adapter]

            if isinstance(param, nn.Module):
                param = param()

            bias_type = self.suffixtuning_type[active_adapter]
            
            no_seqlen_dim = z.ndim == 2

            if no_seqlen_dim:
                z = z.unsqueeze(2)
                C = C.unsqueeze(2)

            match bias_type:
                case SuffixTuningBiasType.SILU_Z_C:
                    y_add = torch.einsum("bdl,bnl,dn -> bdl", F.silu(z), C, param)
                case SuffixTuningBiasType.SILU_Z:
                    y_add = torch.einsum("bdl,d -> bdl", F.silu(z), param)

            if no_seqlen_dim:
                y_add = y_add.squeeze(2)

            y = y + y_add

        return y


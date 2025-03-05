

from dataclasses import dataclass, field
from typing import Optional, Union
from peft.config import PeftConfig
import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer

from modules.mamba_peft_utils import MambaPeftType, register_peft_config, register_peft_tuner
from modules.mamba_tuner_utils import MambaBaseTuner



@register_peft_config(MambaPeftType.ADAPTER)
@dataclass
class AdapterConfig(PeftConfig):
    r: int = field(default=8)
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
    )
    act_func: str = field(default="silu")

    def __post_init__(self):
        self.peft_type = MambaPeftType.ADAPTER



@register_peft_tuner(MambaPeftType.ADAPTER)
class AdapterModel(MambaBaseTuner):
    prefix: str = "adapter_"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        super().__init__(model, peft_config, adapter_name)
    
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def _create_new_module(self, peft_config, adapter_name, target, target_name):
        new_module = None

        new_module = AdapterInjectionLayer(target, adapter_name, r=peft_config.r, act_func=peft_config.act_func)

        return new_module



class AdapterInjectionLayer(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer: nn.Linear, adapter_name, r, act_func) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer
        self.adapter_proj_down = nn.ModuleDict({})
        self.adapter_proj_up = nn.ModuleDict({})
        self.act_func_name = act_func

        self.update_layer(adapter_name, r)

    def update_layer(self, adapter_name, r):
        device = self.base_layer.device if hasattr(self.base_layer, "device") else self.base_layer.weight.device
        dtype = self.base_layer.dtype if hasattr(self.base_layer, "dtype") else self.base_layer.weight.dtype

        self.adapter_proj_down[adapter_name] = nn.Linear(in_features=self.base_layer.in_features, out_features=r, dtype=dtype, device=device)
        self.adapter_proj_up[adapter_name] = nn.Linear(in_features=r, out_features=self.base_layer.in_features, dtype=dtype, device=device)

        self.set_adapter(self.active_adapters)

    def act(self, x):
        return {"relu": F.relu, "silu": F.silu, None: (lambda y: y)}[self.act_func_name](x)

    def forward(self, x):
        y = x

        for active_adapter in self.active_adapters:
            y_skip = y
            y = self.adapter_proj_down[active_adapter](y)
            y = self.act(y)
            y = self.adapter_proj_up[active_adapter](y) + y_skip  # skip

        y = self.base_layer(y)

        return y


from dataclasses import dataclass, field
import math
from types import SimpleNamespace
from typing import Optional
from peft.config import PeftConfig
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer

from modules.mamba_peft_utils import MambaPeftType, StateMlp, register_peft_config, register_peft_tuner
from modules.mamba_tuner_utils import MambaBaseTuner

from peft.tuners.lora import Linear as LoraLinear

def _create_param(shape, dtype, device, init, hidden_dim=None, output_shape="n d", bias=None):
    # bdnl
    if hidden_dim is None:
        param = nn.Parameter(torch.zeros(
                shape, dtype=dtype, device=device))
        match init:
            case "random":
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            case "zero":
                pass
            case "one":
                nn.init.ones_(param)
            case _:
                assert False
        param._no_weight_decay = True
    else:
        n, d = shape
        param = StateMlp(n, d, hidden_dim=hidden_dim, init=init, dtype=dtype, device=device, output_shape=output_shape, bias=bias)

    return param


@register_peft_config(MambaPeftType.DIM_EXTEND)
@dataclass
class DimExtendConfig(PeftConfig):
    d_state: Optional[int] = field(default=None)
    hidden_dim: Optional[int] = field(default=None)
    hidden_dim_BC: Optional[int] = field(default=None)
    proj_lora_r: int = field(default=None)
    use_dora: bool = field(default=False)

    def __post_init__(self):
        self.peft_type = MambaPeftType.DIM_EXTEND



@register_peft_tuner(MambaPeftType.DIM_EXTEND)
class DimExtendModel(MambaBaseTuner):
    prefix: str = "dimex_"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        super().__init__(model, peft_config, adapter_name)
    
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        peft_config.target_modules = [
            "parameter_processors.A_log",
            "x_proj_B",
            "x_proj_C",
        ]

        if peft_config.proj_lora_r is not None:
            # "in_proj_x", "in_proj_z", 
            peft_config.target_modules += ["out_proj"]
        # peft_config.inference_mode = False

        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix in n or (self.peft_config["default"].proj_lora_r is not None and "lora_" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False


    def _create_new_module(self, peft_config, adapter_name, target, target_name):
        new_module = None

        if peft_config.d_state is not None:
            if target_name == "A_log":
                new_module = TensorDimExtend(target, adapter_name, dim=1, n=peft_config.d_state, hidden_dim=peft_config.hidden_dim)
            elif target_name in ("x_proj_B", "x_proj_C"):
                new_module = LinearDimExtend(target, adapter_name, target_name, n=peft_config.d_state, hidden_dim=peft_config.hidden_dim_BC)
            elif target_name in ("b_layernorm", "c_layernorm"):
                new_module = LayerNormDimExtend(target, adapter_name, target_name, n=peft_config.d_state, hidden_dim=peft_config.hidden_dim)
            elif target_name in ("in_proj_x", "in_proj_z", "out_proj") and peft_config.proj_lora_r is not None:
                new_module = LoraLinear(target, adapter_name, r=peft_config.proj_lora_r, lora_alpha=peft_config.proj_lora_r, lora_dropout=0.1, 
                                        use_dora=peft_config.use_dora)

        return new_module


class TensorDimExtend(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer, adapter_name, dim, n=1, hidden_dim=None) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer

        self.dimex_param = nn.ParameterDict({}) if hidden_dim is None else nn.ModuleDict({})  # nn.Parameter(param)
        self.dim = {}

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            dim=dim,
            n=n,
            hidden_dim=hidden_dim,
        )

    def update_layer(self, adapter_name, dim, n, hidden_dim):
        device = self.base_layer.device
        shape_extra = list(self.base_layer.shape)
        shape_extra[dim] = n

        if self.base_layer.parameter_name == "A_log":
            # see mamba A_log init
            d_inner, d_state = self.base_layer.shape

            # param_init = repeat(
            #     torch.arange(d_state + 1, d_state + 1 + n, dtype=torch.float32),
            #     "n -> d n",
            #     d=d_inner,
            # )
            # param_init = torch.log(param_init).to(self.base_layer.dtype)
            param_init = self.base_layer.param_data[:, -1:].repeat(1, n)
            dtype = self.base_layer.param_data.dtype
        
            param = _create_param([n, d_inner], device=device, dtype=dtype, hidden_dim=hidden_dim, init="zero", output_shape="d n", bias=param_init)

            if hidden_dim is None:
                param = param.T.contiguous()
                with torch.no_grad():
                    param.copy_(param_init)
            else:
                # nn.init.constant_(param.transform[2].bias, torch.log(d_state + 1))
                pass
        else:
            assert False
            param = torch.zeros(shape_extra, dtype=self.base_layer.dtype)
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))

        self.dimex_param[adapter_name] = param
        self.dim[adapter_name] = dim

        self.set_adapter(self.active_adapters)

    def forward(self, x):
        assert not x.requires_grad

        out = x

        for active_adapter in self.active_adapters:
            param, dim = self.dimex_param[active_adapter], self.dim[active_adapter]

            if isinstance(param, nn.Module):
                param = param()

            if self.training:
                assert param.requires_grad, "TensorDimensionExtend is frozen!"

            assert out.dtype == param.dtype
            out = torch.cat([out, param], dim=dim)

        return out


class LinearDimExtend(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer: nn.Linear, adapter_name, target_name, n=1, hidden_dim=None) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer
        self.dimex_weight = nn.ParameterDict({}) if hidden_dim is None else nn.ModuleDict()
        self.target_name = target_name

        self.update_layer(adapter_name, n, hidden_dim)

    @property
    def out_features(self):
        dim_ex = sum([self.dimex_weight[a].shape[0] for a in self.active_adapters])
        return self.base_layer.out_features + dim_ex

    def update_layer(self, adapter_name, n, hidden_dim):
        bias = self.base_layer.bias is not None
        assert not bias

        # weight = torch.zeros(
        #     [n, self.base_layer.in_features], 
        #     device=self.base_layer.weight.device,
        #     dtype=self.base_layer.weight.dtype
        # )

        # if self.target_name == "x_proj_B":
        #     nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        # elif self.target_name == "x_proj_C":
        #     weight.data.fill_(0)
        # else:
        #     raise Exception(self.target_name)
        weight = _create_param(
            [n, self.base_layer.in_features],
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
            init={"x_proj_B": "random", "x_proj_C": "zero"}[self.target_name],
            hidden_dim=hidden_dim,
        )

        self.dimex_weight[adapter_name] = weight  # nn.Parameter(weight)

        self.set_adapter(self.active_adapters)

    def forward(self, x):
        assert not self.base_layer.weight.requires_grad
        assert self.base_layer.bias is None or not self.base_layer.bias.requires_grad

        y1 = self.base_layer(x)

        tensors = [y1]
        for active_adapter in self.active_adapters:
            weight = self.dimex_weight[active_adapter]

            if isinstance(weight, nn.Module):
                weight = weight()

            if self.training:
                assert weight.requires_grad

            y2 = F.linear(x, weight)

            assert y1.dtype == y2.dtype
            tensors.append(y2)

        y = torch.cat(tensors, dim=-1)
        return y


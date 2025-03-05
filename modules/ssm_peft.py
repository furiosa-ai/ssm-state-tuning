
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import enum
import math
from typing import Dict, Optional, Union, Any
from peft.config import PeftConfig
import torch
import torch.nn.functional as F
from torch import nn
from einops import einsum, repeat, rearrange

from peft.tuners.tuners_utils import BaseTunerLayer

from modules.mamba_peft_utils import MambaPeftType, StateMlp, register_peft_config, register_peft_tuner
from modules.mamba_tuner_utils import MambaBaseTuner
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from utils.utils import find_module_parent

from peft.tuners.lora import Linear as LoraLinear


def _create_param(shape, dtype, device, init):
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

    return param


@register_peft_config(MambaPeftType.SSMPEFT)
@dataclass
class SsmPeftConfig(PeftConfig):
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
    )
    scan_state: Optional[Dict[str, Any]] = field(default=None)
    state_to_y: Optional[Dict[str, Any]] = field(default=None)
    conv_init_state: bool = field(default=False)
    conv_init_state_init: str = field(default="zero")
    conv_init_state_mlp_hidden_dim: Optional[int] = field(default=None)
    proj_lora_r: int = field(default=None)
    use_dora: bool = field(default=False)

    def __post_init__(self):
        self.peft_type = MambaPeftType.SSMPEFT

        if self.target_modules is None:
            self.target_modules = []

            if self.scan_state is not None or self.state_to_y is not None:
                self.target_modules += ["mixer"]

            if self.conv_init_state:
                self.target_modules += ["conv1d"]


@register_peft_tuner(MambaPeftType.SSMPEFT)
class SsmPeftModel(MambaBaseTuner):
    prefix: str = "ssmpeft_"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        super().__init__(model, peft_config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.proj_lora_r is not None:
            # "in_proj_x", "in_proj_z", 
            peft_config.target_modules += ["out_proj"]
        
        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for n, p in model.named_parameters():
            if self.prefix in n or (self.peft_config["default"].proj_lora_r is not None and "lora_" in n):
                p.requires_grad = True

    def _create_new_module(self, peft_config, adapter_name, target, target_name):
        from modules.mamba_peft import MambaPeft

        if target_name == "mixer":
            assert isinstance(target, MambaPeft)
            # target.selective_scan_fn = SelectiveScanPeft(target, adapter_name, state_to_y=peft_config.state_to_y)
            delattr(target, "selective_scan_fn")
            setattr(target, "selective_scan_fn", SelectiveScanPeft(
                target, adapter_name, scan_state=peft_config.scan_state, state_to_y=peft_config.state_to_y))
            return None
        elif target_name == "conv1d":
            mamba_block = find_module_parent(self.model, target)
            return CausalConvWithHiddenState(target, mamba_block, adapter_name, 
                                             init=peft_config.conv_init_state_init, 
                                             hidden_dim=peft_config.conv_init_state_mlp_hidden_dim)
        elif target_name in ("in_proj_x", "in_proj_z", "out_proj") and peft_config.proj_lora_r is not None:
            return LoraLinear(target, adapter_name, r=peft_config.proj_lora_r, lora_alpha=peft_config.proj_lora_r, lora_dropout=0.1, 
                                    use_dora=peft_config.use_dora)
        else:
            assert False


class StateToY(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, u, C, h):
        return torch.einsum("bnl,bndl->bdl", C, h)  # + torch.einsum("nd,bndl->bdl", param, h)


class StateToYWithTransform(nn.Module):
    def __init__(self, block, scale=False, bias=False, scale_C=False, bias_C=False, init="zero", dtype=None, device=None) -> None:
        super().__init__()
        assert any([scale, bias, scale_C, bias_C])

        self.ssmpeft_scale = _create_param(
            (block.d_state, block.d_inner),
            dtype=dtype if dtype is not None else block.x_proj.x_proj_C.weight.dtype, 
            device=device if device is not None else block.x_proj.x_proj_C.weight.device,
            init={"zero": "one", "random": "random"}[init]
        ) if scale else None

        self.ssmpeft_bias = _create_param(
            (block.d_state, block.d_inner),
            dtype=dtype if dtype is not None else block.x_proj.x_proj_C.weight.dtype, 
            device=device if device is not None else block.x_proj.x_proj_C.weight.device,
            init=init
        ) if bias else None

        self.ssmpeft_scale_C = _create_param(
            (block.d_state, block.d_inner),
            dtype=dtype if dtype is not None else block.x_proj.x_proj_C.weight.dtype, 
            device=device if device is not None else block.x_proj.x_proj_C.weight.device,
            init={"zero": "one", "random": "random"}[init]
        ) if scale_C else None

        self.ssmpeft_bias_C = _create_param(
            (block.d_state, block.d_inner),
            dtype=dtype if dtype is not None else block.x_proj.x_proj_C.weight.dtype, 
            device=device if device is not None else block.x_proj.x_proj_C.weight.device,
            init=init
        ) if bias_C else None

    def forward(self, u, C, h):
        if self.ssmpeft_scale is not None:
            h = h * rearrange(self.ssmpeft_scale, "n d -> 1 n d 1")

        if self.ssmpeft_bias is not None:
            h = h + rearrange(self.ssmpeft_bias, "n d -> 1 n d 1")

        if hasattr(self, "ssmpeft_scale_C") and (self.ssmpeft_scale_C is not None or self.ssmpeft_bias_C is not None):
            C = rearrange(C, "b n l -> b n 1 l")

            if self.ssmpeft_scale_C is not None:
                C = C * rearrange(self.ssmpeft_scale_C, "n d -> 1 n d 1")

            if self.ssmpeft_bias_C is not None:
                C = C + rearrange(self.ssmpeft_bias_C, "n d -> 1 n d 1")

            y = torch.einsum("bndl,bndl->bdl", C, h)
        else:
            y = torch.einsum("bnl,bndl->bdl", C, h)

        return y


class StateToYWithPrevState(nn.Module):
    def __init__(self, block, init="random") -> None:
        super().__init__()

        # self.ssmpeft_x_proc_C_prev = nn.Linear(
        #     block.d_inner, block.d_state, bias=False, dtype=block.out_proj.weight.dtype, device=block.out_proj.weight.device
        # ).weight

        self.ssmpeft_x_proc_C_prev = _create_param(
            (block.d_state, block.d_inner),
            dtype=block.x_proj.x_proj_C.weight.dtype, 
            device=block.x_proj.x_proj_C.weight.device,
            init=init
        )

    def forward(self, u, C, h):
        C_prev = F.linear(rearrange(u, "b d l -> b l d"), self.ssmpeft_x_proc_C_prev)
        C_prev = rearrange(C_prev, "b l dstate -> b dstate l", b=u.shape[0]).contiguous()

        h_shift = F.pad(h[..., :-1], [1, 0], value=0)
        return torch.einsum("bnl,bndl->bdl", C, h) + torch.einsum("bnl,bndl->bdl", C_prev, h_shift)


class SelectiveScanState(nn.Module):
    def forward(self, u, delta, A, B):
        b, n, l = B.shape
        ones = torch.ones((b, 1, l), device=u.device, dtype=u.dtype, requires_grad=True)

        h = []
        for i in range(n):
            h_i = selective_scan_fn(u, delta, A[:, i:i+1], B[:, i:i+1], C=ones, D=None, z=None, 
                                    delta_bias=None, delta_softplus=False, return_last_state=False)
            h.append(h_i)
        h = torch.stack(h, 1)

        return h


class SelectiveScanStateWithInitState(nn.Module):
    def __init__(self, block, init="random", hidden_dim=None, dtype=None, device=None) -> None:
        super().__init__()

        self.ssmpeft_init_state = self.create_init_state(
            (block.d_inner, block.d_state),
            dtype=dtype if dtype is not None else block.x_proj.x_proj_C.weight.dtype, 
            device=device if device is not None else block.x_proj.x_proj_C.weight.device,
            init=init,
            hidden_dim=hidden_dim
        )

        self.selective_scan_fn = selective_scan_fn

    def create_init_state(self, shape, dtype, device, init, hidden_dim):
        if hidden_dim is None:
            param = _create_param(
                shape, dtype, device, init
            )
        else:
            param = StateMlp(shape[1], shape[0], hidden_dim, init, dtype, device)

        param._no_weight_decay = True
        return param

    def get_init_state(self):
        param = self.ssmpeft_init_state

        if isinstance(param, StateMlp):
            param = param()

        return param

    def forward(self, u, delta, A, B):
        b, n, l = B.shape
        init_state = self.get_init_state()[None].repeat(b, 1, 1)
        ones = torch.ones((b, 1, l+1), device=u.device, dtype=u.dtype, requires_grad=True)

        h = []
        for i in range(n):
            u_i = torch.cat([init_state[:, :, i:i+1], u], 2)
            A_i = A[:, i:i+1]
            B_i = F.pad(B[:, i:i+1], (1, 0), value=1)
            delta_i = F.pad(delta, (1, 0), value=1)
            
            # dont use ref
            h_i = self.selective_scan_fn(u_i, delta_i, A_i, B_i, ones, D=None, z=None)
            h_i = h_i[:, :, 1:]

            h.append(h_i)  # discard initial state
            
        h = torch.stack(h, 1)
        return h


class SelectiveScanPeft(nn.Module, BaseTunerLayer):
    def __init__(self, mamba_block, adapter_name, **kwargs) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.scan_state = nn.ModuleDict({
            adapter_name: SelectiveScanState()
        })

        self.state_to_y = nn.ModuleDict({
            adapter_name: StateToY()
        })

        self._active_adapter = adapter_name
        self.update_layer(
            mamba_block, 
            adapter_name,
            **kwargs
        )

    def update_layer(self, mamba_block, adapter_name, scan_state=None, state_to_y=None):
        if state_to_y is not None:
            state_to_y = {**state_to_y}
            self.state_to_y[adapter_name] = {
                "prev_state": lambda: StateToYWithPrevState(mamba_block, **state_to_y),
                "transform": lambda: StateToYWithTransform(mamba_block, **state_to_y),
            }[state_to_y.pop("type")]()

        if scan_state is not None:
            scan_state = {**scan_state}
            self.scan_state[adapter_name] = {
                "init_state": lambda: SelectiveScanStateWithInitState(mamba_block, **scan_state)
            }[scan_state.pop("type")]()
        
        self.set_adapter(self.active_adapters)

    def forward(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        assert delta_bias is None
        assert delta_softplus
        # assert not return_last_state

        if delta_softplus:
            delta = F.softplus(delta)

        h = self.scan_state[self.active_adapter[0]](u, delta, A, B)
        y = self.state_to_y[self.active_adapter[0]](u, C, h)
        
        y = y + u * D[None, :, None]  # .to(u.dtype)
        y = y * F.silu(z)
        y = y.to(u.dtype)
        
        if return_last_state:
            return y, rearrange(h[..., -1], "b n d -> b d n")
        else:
            return y


class CausalConvWithHiddenState(nn.Module, BaseTunerLayer):
    def __init__(self, conv1d: nn.Conv1d, mamba_block, adapter_name, **kwargs) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.ssmpeft_conv_state = nn.ParameterDict({})
        self.conv1d = conv1d

        self._active_adapter = adapter_name
        self.update_layer(
            conv1d, 
            mamba_block,
            adapter_name,
            **kwargs
        )

    @property
    def weight(self):
        return self.conv1d.weight

    def update_layer(self, conv1d: nn.Conv1d, mamba_block, adapter_name, init, hidden_dim=None):
        shape = (mamba_block.d_inner, mamba_block.d_conv-1)
        device = conv1d.weight.device
        dtype = conv1d.weight.dtype

        self.ssmpeft_conv_state[adapter_name] = self.create_init_state(shape, dtype, device, init, hidden_dim)
        
        self.set_adapter(self.active_adapters)

    def create_init_state(self, shape, dtype, device, init, hidden_dim):
        if hidden_dim is None:
            param = _create_param(
                shape, dtype, device, init
            )
        else:
            param = StateMlp(shape[1], shape[0], hidden_dim, init, dtype, device)

        param._no_weight_decay = True
        return param

    def get_init_state(self):
        param = self.ssmpeft_conv_state[self.active_adapter[0]]

        if isinstance(param, StateMlp):
            param = param()

        return param

    def forward(self, x):
        if x.ndim == 3:
            param = self.get_init_state()
            param = repeat(param, "d k ->b d k", b=x.shape[0])
            x = torch.cat([param, x], 2)

        y = self.conv1d(x)

        if x.ndim == 3:
            y = y[:, :, param.shape[2]:]

        return y
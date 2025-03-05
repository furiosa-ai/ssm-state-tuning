
import enum
from einops import rearrange
from torch import nn
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
import torch


def set_peft_params_trainable(model, pattern, enable_train, disable_train):
    for n, p in model.named_parameters():
        if pattern in n:
            if enable_train:
                p.requires_grad = True
        else:
            if disable_train:
                p.requires_grad = False


class ParameterProcessor(nn.Module):
    def __init__(self, parameter_name, shape, dim_names, dtype, device=None, all_dims=None, param=None) -> None:
        super().__init__()

        self.parameter_name = parameter_name
        self.shape = shape
        self.dim_names = dim_names
        self.dtype = dtype
        self.all_dims = all_dims

        self.param_data = param.data if param is not None else None

        if device is not None:
            self.device = device

    @property
    def in_features(self):
        return self.shape[1]
    
    @property
    def out_features(self):
        return self.shape[1]

    def get_dim_size(self, dim):
        if isinstance(dim, str):
            dim = self.get_dim_index(dim)
        return self.shape[dim]

    def get_dim_index(self, dim):
        return self.dim_names.index(dim)

    def forward(self, x, **kwargs):
        return x



class StateMlp(nn.Module):
    shared_embed = {}

    def __init__(self, n, token_dim, hidden_dim, init, dtype, device, output_shape=None, bias=None):
        super().__init__()

        if isinstance(hidden_dim, float):
            hidden_dim = int(hidden_dim * token_dim)
        
        if n not in StateMlp.shared_embed:
            StateMlp.shared_embed[(n, dtype)] = nn.Embedding(n, token_dim, dtype=dtype, device=device)

        self.embed = StateMlp.shared_embed[(n, dtype)]
        self.transform = nn.Sequential(
            nn.Linear(token_dim, hidden_dim, dtype=dtype, device=device),
            nn.Tanh(),
            nn.Linear(hidden_dim, token_dim, dtype=dtype, device=device, bias=bias is None),
        )

        if init == "zero":
            nn.init.zeros_(self.transform[2].weight)
            if self.transform[2].bias is not None:
                nn.init.zeros_(self.transform[2].bias)
        elif init == "random":
            pass
        else:
            assert False

        self.prompt_tokens = torch.arange(n, device=device).long()
        self.output_shape = output_shape
        self.bias = nn.Parameter(bias).to(device).to(dtype) if bias is not None else None

    @property
    def shape(self):
        return self.embed.weight.shape

    def forward(self):
        y = self.transform(self.embed(self.prompt_tokens))

        if self.output_shape is None:
            y = y.T
        else:
            y = rearrange(y, "n d -> " + self.output_shape)

        if self.bias is not None:
            y = y + self.bias

        return y
        

class MambaPeftType(str, enum.Enum):
    DIM_EXTEND = "DIM_EXTEND"
    SUFFIX_TUNING = "SUFFIX_TUNING"
    SSMPEFT = "SSMPEFT"


def register_peft_tuner(name):
    def _wrap(cls):
        PEFT_TYPE_TO_MODEL_MAPPING[name] = cls
        return cls
    
    return _wrap


def register_peft_config(name):
    def _wrap(cls):
        PEFT_TYPE_TO_CONFIG_MAPPING[name] = cls
        return cls
    
    return _wrap


def _init():
    from .dim_extend import DimExtendModel
    from .suffix_tuning import SuffixTuningModel
    from .ssm_peft import SsmPeftModel

_init()

import functools
import sys
from pathlib import Path

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba_simple import Mamba
from modules import get_mamba_peft_model
from modules.mamba_peft import MambaPeft

from modules.mixer_seq_simple import MambaLMHeadModelPeft
from utils.debug_utils import enable_deterministic

sys.path.insert(0, str(Path(__file__).parent.parent))

from pprint import pprint
import unittest

import numpy as np
import torch
import yaml
import time
from typing import Callable, List, Dict




class TestMambaRec(unittest.TestCase):
    peft_cfgs = [
        None,
        "cfg/final/peft/mamba-130m/dart/add_scan.json",
        "cfg/final/peft/mamba-130m/dart/init_state.json",
        "cfg/final/peft/mamba-130m/dart/lora.json",
        "cfg/final/peft/mamba-130m/dart/output_tuning.json",
        "cfg/final/peft/mamba-130m/dart/state_tuning.json"
    ]

    peft_logit_means = {
        None: 63.794083,
        "cfg/final/peft/mamba-130m/dart/add_scan.json": 64.193474,
        "cfg/final/peft/mamba-130m/dart/init_state.json": 14.948129,
        "cfg/final/peft/mamba-130m/dart/lora.json": 64.225052,
        "cfg/final/peft/mamba-130m/dart/output_tuning.json": 61.770397,
        "cfg/final/peft/mamba-130m/dart/state_tuning.json": -33.536594
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = "cuda"
        cls.dtype = torch.float32
        # cls.dtype = torch.bfloat16

        cls.b = 1
        cls.test_in_len = 5
        cls.test_out_len = 3

        cls.input_ids = torch.randint(1, 100, (cls.b, cls.test_in_len), 
                                      generator=torch.Generator(device=cls.device).manual_seed(0), device=cls.device).long()

    @classmethod
    def assert_logits_close(cls, actual, expected, msg=None):
        if isinstance(actual, float):
            return np.testing.assert_allclose(float(actual), float(expected), rtol=1e-5, atol=1e-5)
        else:
            return torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3, msg=msg)
    
    @classmethod
    def assert_equal(cls, actual, expected, msg=None):
        return torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=msg)

    @classmethod
    def init_uniform(cls, param, a_min, a_max=None):
        v = torch.rand(param.data.shape, dtype=param.data.dtype, device=param.data.device, 
                       generator=torch.Generator(device=cls.device).manual_seed(0))

        if a_max is None:
            a_min, a_max = -a_min, a_min

        v = v  * (a_max - a_min) + a_min
        param.data[:] = v

    @classmethod
    def init_param(cls, name, param):
        if "actscale_param" in name:
            cls.init_uniform(param, 0.5, 1.5)
        else:
            cls.init_uniform(param, 0.1)

    @classmethod
    def random_init_trainable(cls, model):
        trainable_params = {
            name: param for name, param in model.named_parameters() if param.requires_grad
        }

        assert len(trainable_params) > 0

        for name, param in trainable_params.items():
            cls.init_param(name, param)

    @classmethod
    def generate(cls, model):
        out_seq = model.generate(
            input_ids=cls.input_ids,
            max_length=cls.test_in_len + cls.test_out_len,
            top_k=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

        return out_seq

    @classmethod
    def load_mamba(cls, peft, base=False):
        enable_deterministic()
        # tokenizer = get_tokenizer("EleutherAI/gpt-neox-20b")

        model_kwargs = dict(
            dtype=cls.dtype, 
            device=cls.device,
            use_fast_path=False,
            mamba_cls=MambaPeft if not base else Mamba
        )

        model = MambaLMHeadModelPeft.from_pretrained(
            "state-spaces/mamba-130m", 
            **model_kwargs
        )

        if peft is not None:
            if isinstance(peft, list):
                for peft_inst in peft:
                    model = get_mamba_peft_model(model, peft_inst, return_peft_cfg=False)
            else:
                model = get_mamba_peft_model(model, peft, return_peft_cfg=False)
            cls.random_init_trainable(model)

        model.eval()
        return model
    
    @classmethod
    @torch.no_grad()
    def template_test_rec_par_equal(cls, peft_cfg):
        print("recurrent")
        model = cls.load_mamba(peft_cfg)

        enable_deterministic()
        out_seq_rec = cls.generate(model)

        input_par_ids = out_seq_rec.sequences[:, :-1]
        cls.assert_equal(out_seq_rec.sequences[:, :cls.test_in_len], cls.input_ids)
        # target_par_ids = out_seq_rec.sequences[:, cls.test_in_len:]
        target_par_logits = torch.stack(out_seq_rec.scores, 1)

        # discard logits for input ids
        print("parallel")

        enable_deterministic()
        pred_par_logits = model(input_par_ids).logits[:, cls.test_in_len-1:]
        # pred_par_ids = pred_par_logits.argmax(2)
        
        # self.assert_equal(pred_par_ids, target_par_ids, "Output ids not equal")
        cls.assert_logits_close(pred_par_logits, target_par_logits)  # , "Output logits not equal"

    
    @classmethod
    @torch.no_grad()
    def template_test_logit_mean_equal(cls, peft_cfg):
        model = cls.load_mamba(peft_cfg)

        enable_deterministic()
        pred_par_logits = model(cls.input_ids).logits
        cls.assert_logits_close(pred_par_logits.mean().item(), cls.peft_logit_means[peft_cfg])  # , "Output logits not equal"
    
    @classmethod
    def generate_test_functions(cls) -> None:
        for peft_cfg in cls.peft_cfgs:
            test_name = peft_cfg

            if test_name is not None and isinstance(test_name, (tuple, list)):
                test_name = "_".join([t[len("cfg/peft/"):].split(".")[0].replace("/", "_") for t in test_name])
            else:
                test_name = test_name[len("cfg/peft/"):].split(".")[0].replace("/", "_") if test_name is not None else "no_peft"

            # setattr(cls, attr_name, cls.template_test_equal(peft_cfg))
            setattr(cls, f"test_{test_name}_equal", functools.partial(cls.template_test_rec_par_equal, peft_cfg=peft_cfg))
            setattr(cls, f"test_{test_name}_logit_equal", functools.partial(cls.template_test_logit_mean_equal, peft_cfg=peft_cfg))

    # def test_base_equal(self):
    #     self.assert_rec_par_equal(self.load_mamba(None, base=True))


if __name__ == '__main__':
    TestMambaRec.generate_test_functions()
    unittest.main()

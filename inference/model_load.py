import random
import numpy as np
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM
import torch

from SkipLayer.setup_models import setup_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelLoader:
    def __init__(self, device, dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype


    def load_tokenizer(self, model_id: str, token=""):
        return AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
    

    def basic_load(self, 
                   model_id: str, 
                   seed: int, 
                   quant: int | str | None = None, 
                   token: str = "", 
                   decoding_type: str = "basic"):
        set_seed(seed)

        tokenizer= self.load_tokenizer(model_id, token)

        model_args = {
            "pretrained_model_name_or_path": model_id,
            "torch_dtype": self.dtype,
            "use_auth_token": token
        }

        if quant == "4":
            model_args["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16
                            )
        elif quant == "8":
            model_args['quantization_config'] = BitsAndBytesConfig(
                                load_in_8bit=True,
                            )

        elif quant == "32" or (quant is None and os.environ.get('USE_FP32_MODEL', False)):
            model_args["torch_dtype"] = torch.float32

        if 'sl' in decoding_type:
            # decoding_type sl-h는 랜덤하게 layer 건너 뛴 걸 아마추어 모델로 사용. sh-d는 엔트로피 고려하여 layer 건너 뜀.
            tokenizer, model = setup_model(
                    algorithm=decoding_type,
                    model_dir=model_id,
                    prefix="당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트이다. 한국어와 영어로 생각하고, 한국어로 답한다.", # need by sl-d
                    model_cls=LlamaForCausalLM,
                    seed=seed,
                    quant=quant
                )
        else:
            model_args["trust_remote_code"]=True
            model = AutoModelForCausalLM.from_pretrained(**model_args)
            model.to(self.device)

        return tokenizer, model
    

class ModelSetting:
    def __init__(self, max_len: int, temperature: float, top_p: float):
        self.max_len = max_len
        self.temperature = temperature
        self.top_p = top_p


    def basic_setting(self, tokenizer):
        tokenizer.pad_token = tokenizer.eos_token
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.convert_tokens_to_ids("<|eot_id|>") else tokenizer.convert_tokens_to_ids("<|endoftext|>")
        ]
        return terminators
import argparse
import json
import os

from datasets import Dataset, concatenate_datasets

import torch

from transformers import set_seed, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--train_type", type=str, choices=["sft", "cpt"], help="supervised fine-tuning or continual pre-training")
g.add_argument("--model_id", type=str, required=True, help="model id")
g.add_argument("--train_file", type=str, required=True, help="train filename")
g.add_argument("--dev_file", type=str, required=True, help="dev filename")
g.add_argument('--output_dir', type=str, default="")
g.add_argument('--model_max_length', type=int, default=1024)
g.add_argument('--lora_r', type=int, default=16)
g.add_argument('--lora_alpha', type=int, default=32)
g.add_argument('--learning_rate', type=float, default=8e-4)
g.add_argument('--warmup_ratio', type=float, default=0.05)
g.add_argument('--batch_size', type=int, default=4)
g.add_argument('--gradient_accumulation_steps', type=int, default=4)
g.add_argument('--num_epochs', type=int, default=3)
g.add_argument('--seed', type=int, default=42)
# g.add_argument('--wandb_key', type=str, required=True)
# g.add_argument('--wandb_project_name', type=str, default="Korean_25_Training")
g.add_argument('--experiment_name', type=str, default="sft")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    # question_type이 "서술형"인 항목만 남기기
    return Dataset.from_list([
        item
        for item in items
        if item.get("input", {}).get("question_type") == "서술형"
    ])


def generate_prompt(example, tokenizer, type_instructions, sys_prompt):
    original_question = example['input']['question']
    q_type = example['input']['question_type']
    answer = example['output']['answer']

    category = example['input']['category']
    domain = example['input']['domain']

    persona = f"당신은 한국 {category}, {domain}에 대해 잘 아는 한국 특화 인공지능 어시스턴트이다."

    instruction = type_instructions.get(q_type, "")

    chat_parts = [instruction]

    other_info = {k: v for k, v in example['input'].items() if k not in ['question', 'question_type', 'context']}

    if other_info:
        info_list = ["[기타 정보]"]
        for key, value in other_info.items():
            info_list.append(f"- {key}: {value}")
        chat_parts.append("\n".join(info_list))

    chat_parts.append(f"[질문]\n{original_question}")

    chat = "\n\n".join(chat_parts)
    
    message = {"messages":
            [{"role": "system", "content": sys_prompt + "\n\n" + persona},
            {"role": "user", "content": chat},
            {"role": "assistant", "content": answer}]
    }
    
    # formatted_text = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    # print(formatted_text)
    return message


def main(args):
    # wandb.login(key=args.wandb_key) 

    set_seed(args.seed)

    # os.environ["WANDB_PROJECT"] = args.wandb_project_name

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"[RANK {local_rank}] Current CUDA device: {torch.cuda.current_device()}")

    experiment_dir = f"{args.output_dir}/{args.experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    sys_prompt = 'You are a helpful AI assistant.\n당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트이다.\n한국어와 영어로 생각하고, 한국어로 답하시오.\n\n[기타 정보]를 답변에 충실히 반영하시오.'

    with open("../prompt/type_instructions_basic_noex.json", "r") as f:
        type_instructions = json.load(f)

    train_items = load_json(args.train_file)
    dev_items   = load_json(args.dev_file)

    train_dataset = train_items.map(generate_prompt, 
                                    fn_kwargs={
                                            "tokenizer": tokenizer,
                                            "type_instructions": type_instructions,
                                            "sys_prompt": sys_prompt,
                                            },
                                    remove_columns=train_items.column_names)
    dev_dataset = dev_items.map(generate_prompt, 
                                  fn_kwargs={
                                            "tokenizer": tokenizer,
                                            "type_instructions": type_instructions,
                                            "sys_prompt": sys_prompt,
                                            },
                                    remove_columns=dev_items.column_names)
    print(train_dataset[0])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        # llm_int8_enable_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        # device_map="auto",
        device_map=None,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    model.train()
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
    )

    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=experiment_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no", 
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        bf16=True,
        tf32=True,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        warmup_ratio=args.warmup_ratio,
        push_to_hub=False,
        # report_to="wandb",
        save_total_limit=0 ,
        ddp_find_unused_parameters=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=dev_dataset,
        # formatting_func=lambda x: x["text"]
    )

    trainer.train()

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(f"{experiment_dir}/merged_model")
    tokenizer.save_pretrained(f"{experiment_dir}")
    
    
if __name__ == "__main__":
    main(parser.parse_args())

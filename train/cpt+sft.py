import os
import json
import argparse
import torch
import numpy as np
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from data_fix import prepare_datasets

# 원본 모델 경로
model_id = "K-intelligence/Midm-2.0-Base-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def create_model_and_tokenizer(model_id, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    return model

def train_continuous_sft(model, continuous_dataset, eval_continuous_dataset, args):
    print("\n[1단계] CPT (단답형, 선다형) 시작")
    
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
    
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.train()
    model.gradient_checkpointing_enable()

    stage_dir = f"{args.output_dir}/{args.experiment_name}_continuous_sft"
    os.makedirs(stage_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=stage_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="epoch",
        learning_rate=args.learning_rate_cp,
        bf16=True,
        max_grad_norm=1.0,
        warmup_steps=50,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        push_to_hub=False,
        # report_to="wandb",
        save_total_limit=0,
        run_name=f"{args.experiment_name}_continuous_sft",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=continuous_dataset,
        eval_dataset=eval_continuous_dataset,
        formatting_func=lambda x: x["text"]  # prepare_datasets 에서 이미 전처리됨
    )

    trainer.train()

    print(f"cpt 학습 완료")

    return trainer.model

def train_instruction_tuning(model, instruction_dataset, eval_instruction_dataset, args):
    print("\n[2단계] SFT(서술형) 시작")

    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

    model.train()
    model.gradient_checkpointing_enable()

    stage_dir = f"{args.output_dir}/{args.experiment_name}_instruction_tuning"
    os.makedirs(stage_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=stage_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="epoch",
        learning_rate=args.learning_rate_sft,
        bf16=True,
        max_grad_norm=1.0,
        warmup_steps=30,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        push_to_hub=False,
        # report_to="wandb",
        save_total_limit=0,
        run_name=f"{args.experiment_name}_instruction_tuning",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=instruction_dataset,
        eval_dataset=eval_instruction_dataset,
        formatting_func=lambda x: x["text"]
    )

    trainer.train()

    # LoRA 병합 후 저장
    final_model = trainer.model.merge_and_unload()
    final_model.save_pretrained(f"{stage_dir}/final_model")
    tokenizer.save_pretrained(f"{stage_dir}/final_model")
    print(f"\n최종 모델 저장 완료: {stage_dir}/final_model")
    return final_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default="./trained_model")
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--learning_rate_cp', type=float, required=True, help="Continuous(단답형) SFT learning rate")
    parser.add_argument('--learning_rate_sft', type=float, required=True, help="Instruction tuning learning rate")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--experiment_name', type=str, default="cpt_sft")
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--eval_data', type=str)
    args = parser.parse_args()

    # 시드 고정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # W&B 로깅
    # os.environ["WANDB_PROJECT"] = "midm_cpt_fix_sft"
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    # 데이터 준비 (이름 유지: train_continuous / train_instruction)
    train_continuous, train_instruction = prepare_datasets(args.train_data, tokenizer)
    eval_continuous, eval_instruction = prepare_datasets(args.eval_data, tokenizer)

    # 모델 로드 (4bit 양자화)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = create_model_and_tokenizer(model_id, bnb_config)
    model.train()
    model.gradient_checkpointing_enable()

    # 1단계: 단답형,선다형(continuous) SFT
    #_ = train_continuous_sft(model, train_continuous, eval_continuous, args)
    
    # 단답형 선다형 SFT → 병합 없이 LoRA 유지
    model = train_continuous_sft(model, train_continuous, eval_continuous, args)
    
    # instruction tuning → 마지막에만 병합
    final_model = train_instruction_tuning(model, train_instruction, eval_instruction, args)
    
    print("\n전체 학습 완료.")


if __name__ == "__main__":
    main()

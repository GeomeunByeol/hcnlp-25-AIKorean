MODEL_ID="K-intelligence/Midm-2.0-Base-Instruct"
TRAIN_FILE="../resource/QA/korean_culture_qa_V1.0_train+.json"
DEV_FILE="../resource/QA/korean_culture_qa_V1.0_dev+.json"
OUTPUT_DIR="./trained_model"

LORA_R=8
LORA_ALPHA=8
LEARNING_RATE=2e-5
WARMUP=0.05
BATCH=4
GRAD_ACC=2
EPOCH=2
SEED=42

CUDA_VISIBLE_DEVICES="0" python sft.py \
  --model_id "$MODEL_ID" \
  --train_file "$TRAIN_FILE" \
  --dev_file "$DEV_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --learning_rate "$LEARNING_RATE" \
  --warmup_ratio "$WARMUP" \
  --batch_size "$BATCH" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --num_epochs "$EPOCH" \
  --seed "$SEED" \
  --experiment_name "$EXP_NAME"
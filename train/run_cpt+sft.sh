TRAIN_DATA="../resource/QA/korean_culture_qa_V1.0_train+.json"
EVAL_DATA="../resource/QA/korean_culture_qa_V1.0_dev+.json"

CUDA_VISIBLE_DEVICES=0 python /home/ellt/Workspace/Korean_Culture/jungwon/cpt/cpt_basic.py \
    --lora_r 4 \
    --lora_alpha 16 \
    --learning_rate_cp 5e-6 \
    --learning_rate_sft 1e-5 \
    --num_epochs 2 \
    --batch_size 4 \
    --train_data "$TRAIN_DATA" \
    --eval_data "$EVAL_DATA"
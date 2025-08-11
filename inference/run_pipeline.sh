OUTPUT_FILE=../output/pipeline.json

CUDA_VISIBLE_DEVICES=0 python run.py \
    --type pipeline \
    --model_id Eooojin/hcnlp-mincho \
    --input ../resource/QA/korean_culture_qa_V1.0_test+.json \
    --output "$OUTPUT_FILE" \
    --device cuda:0 \
    --context_type rag \
    --decoding_type sl-d \
    --max_len 2048 \
    --temperature 0.4 \
    --top_p 0.6
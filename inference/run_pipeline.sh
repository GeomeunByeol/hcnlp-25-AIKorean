OUTPUT_FILE=../output/117.json

CUDA_VISIBLE_DEVICES=3 python run.py \
    --type pipeline \
    --input ../resource/QA/korean_culture_qa_V1.0_test+.json \
    --output "$OUTPUT_FILE" \
    --device cuda:0 \
    --context_type rag \
    --decoding_type sl-d \
    --max_len 2048 \
    --temperature 0.4 \
    --top_p 0.6
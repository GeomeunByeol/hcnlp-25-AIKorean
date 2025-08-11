OUTPUT_FILE=../output/117.json

CUDA_VISIBLE_DEVICES=0 python run.py \
    --input ../resource/QA/korean_culture_qa_V1.0_test+.json \
    --output "$OUTPUT_FILE" \
    --seon_model_id K-intelligence/Midm-2.0-Base-Instruct \
    --dan_model_id Eooojin/hcnlp-hamster \
    --seo_model_id Eooojin/hcnlp-mincho \
    --seed 42 \
    --temperature 0.4 \
    --top_p 0.6

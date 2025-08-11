# hcnlp-25-AIKorean
HCNLP팀 \
국립국어원 AI말평 [2025]한국문화 질의응답(가 유형)

## 폴더 구조
<pre>
index
output
prompt
resource / QA
├── korean_culture_qa_V1.0_dev+.json
├── korean_culture_qa_V1.0_test+.json
└── korean_culture_qa_V1.0_train+.json

train
├── cpt+sft.py
├── run_cpt+sft.sh
├── sft.sh
└── run_train.sh
  
inference
├── SkipLayer
      ├── compute_skip_layer.py
      ├── llama_contrastive_skip_model.py
      ├── model_loading.py
      ├── model_registry.py
      ├── number_token_ids_llama-3.json
      ├── setup_models.py
      └── stats.py
├── model_load.py
├── rag.py
├── run_routing.sh
└── run_train.py</pre>

## 최종 모델(라우팅) 사용법
1. 데이터 다운로드 받아 resource/QA 폴더 아래 위치
2. run_routing.sh 실행 시 허깅페이스에서 모델 다운로드 후 모델 추론 시작 (모든 모델 CPU에 로드 후 필요할 때만 GPU에 로드하므로 RTX 4090 하나에서 가동 가능)
```bash
OUTPUT_FILE=../output/117.json
CUDA_VISIBLE_DEVICES=0 python run.py \
    --input ../resource/QA/korean_culture_qa_V1.0_test+.json \
    --output "$OUTPUT_FILE" \
    --seon_model_id K-intelligence/Midm-2.0-Base-Instruct \  # 선다형 모델
    --dan_model_id Eooojin/hcnlp-hamster \  # 단답형 모델
    --seo_model_id Eooojin/hcnlp-mincho \  # 서술형 모델
    --seed 42 \
    --temperature 0.4 \
    --top_p 0.6
```
3. 완료 후 output 아래 117.json 파일 생성 (정답 파일)

## 비교 모델(하나의 파이프라인) 사용법


## 학습부터 하는 법

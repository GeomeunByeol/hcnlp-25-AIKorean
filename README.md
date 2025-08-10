# hcnlp-25-AIKorean

## 폴더 구조
<pre>index
output
prompt
resource / QA
├── korean_culture_qa_V1.0_dev+.json
├── korean_culture_qa_V1.0_test+.json
└── korean_culture_qa_V1.0_train+.json
SkipLayer
├── compute_skip_layer.py
├── llama_contrastive_skip_model.py
├── model_loading.py
├── model_registry.py
├── number_token_ids_llama-3.json
├── setup_models.py
└── stats.py
train
├── train.py
└── train.sh
inference.py
models.py
run_routing.sh </pre>

## 최종 모델 사용법
1. 데이터 다운로드 받아 resource/QA 폴더 아래 위치
2. run_routing.sh 실행 시 허깅페이스에서 모델 다운로드 후 모델 추론 시작
3. 완료 후 output 아래 117.json 파일 생성 (정답 파일)

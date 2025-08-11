import argparse
import json
import tqdm
import random
import numpy as np
import ast
import re
import os

import torch
from transformers import set_seed

from model_load import *
from rag import *

parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
# 파이프라인 선택
g.add_argument("--type", type=str, default="routing", choices=["routing", "pipeline"], help="첫 번째 트랙: 하나의 파이프라인. 두 번째 트랙: 라우팅")

# 파일 경로
g.add_argument("--input", type=str, required=True, help="input filename")
g.add_argument("--output", type=str, required=True, help="output filename")

# 모델 선택
g.add_argument("--seon_model_id", type=str, help="라우팅에서 선다형을 위한 모델")
g.add_argument("--dan_model_id", type=str, help="라우팅에서 단답형을 위한 모델")
g.add_argument("--seo_model_id", type=str, help="라우팅에서 서술형을 위한 모델")
g.add_argument("--model_id", type=str, default="K-intelligence/Midm-2.0-Base-Instruct", help="파이프라인 단일 모델")

g.add_argument("--quant", type=str, default=None, choices=["4", "8"], help="quantization(bit)")
g.add_argument("--device", type=str, default="cpu", help="device to load the model")
g.add_argument("--token", type=str, help="Hugging Face token for accessing gated models")

# 디코딩: dola, skiplayer
g.add_argument("--decoding_type", type=str, default="basic", choices=["dola", "sl-h", "sl-d"], help="디고팅 방식. sl-h(hueristic)와 sl-d(dynamic)는 SkipLayer 종류.")
g.add_argument("--dola_layer", type=str, default=None, help="dola layers. 'high', 'low', 숫자들로 이루어진 리스트 중 하나 가능.")

# RAG
g.add_argument("--context_type", type=str, default="basic", choices=["rag"], help="engine type")
g.add_argument("--embedding_path", type=str, default="./qdrant_data", help="Qdrant DB가 저장된 경로")
g.add_argument("--collection_name", type=str, default="rag_collection", help="Qdrant 컬렉션 이름")
g.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-8B", help="embedding_model")
g.add_argument("--k", type=int, default=5, help="검색할 초기 후보 문서 수(Candidate 수)")
g.add_argument("--min_threshold_desc", type=float, default=0.4, help="[서술형] 동적 임계값의 최저 한계값")
g.add_argument("--min_threshold_obj", type=float, default=0.6, help="[선다/단답형] 동적 임계값의 최저 한계값")
g.add_argument("--scaling_factor_desc", type=float, default=1.0, help="[서술형] 동적 임계값 계산 시 적용할 가중치")
g.add_argument("--scaling_factor_obj", type=float, default=2.0, help="[선다/단답형] 동적 임계값 계산 시 적용할 가중치")

# sampling
g.add_argument("--shot", type=int, default=0, help="few shot prompting")
g.add_argument("--max_len", type=int, default=1024, help="max sequence length")
g.add_argument("--temperature", type=float, default=0.6, help="temperature")
g.add_argument("--top_p", type=float, default=0.9, help="top_p")
g.add_argument("--repetition_penalty", type=float, default=1.05, help="repetition_penalty")
g.add_argument("--seed", type=int, default=42, help="seed")


# ---------------------------
# 공통 유틸
# ---------------------------
def _ensure_output_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def _load_prompts(args):
    '''프롬프트 로드'''
    def _read(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    system_prompt_brief = _read("../prompt/system_prompt_brief.txt") if os.path.exists("../prompt/system_prompt_brief.txt") else \
                          'You are a helpful AI assistant.\n당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트이다.\n한국어와 영어로 생각하고, 한국어로 답하시오.\n\n[기타 정보]를 답변에 충실히 반영하시오.'
    system_prompt_desc  = _read("../prompt/system_prompt_desc.txt") if os.path.exists("../prompt/system_prompt_desc.txt") else \
                          'You are a helpful AI assistant.\n당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트이다.\n한국어와 영어로 생각하고, 한국어로 답하시오.\n동일한 문장을 절대 반복하지 마시오.\n300 ~ 500자 사이로 답변하시오.'
    if args.context_type == "rag":
        with open("../prompt/type_instructions_rag.json", "r", encoding="utf-8") as f:
            type_instructions_basic = json.load(f)
        with open("../prompt/type_instructions_rag_noex.json", "r", encoding="utf-8") as f:
            type_instructions_basic_noex = json.load(f)
    else:
        with open("../prompt/type_instructions_basic.json", "r", encoding="utf-8") as f:
            type_instructions_basic = json.load(f)
        with open("../prompt/type_instructions_basic_noex.json", "r", encoding="utf-8") as f:
            type_instructions_basic_noex = json.load(f)
    return system_prompt_brief, system_prompt_desc, type_instructions_basic, type_instructions_basic_noex


def _load_data(input_path: str):
    '''데이터 로드'''
    with open("../resource/QA/korean_culture_qa_V1.0_dev+.json", 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open(input_path, "r", encoding="utf-8") as f:
        test_result = json.load(f)
    return dev_data, test_result


def _select_thresholds(q_type: str, args):
    '''RAG에서 임계치 설정'''
    if q_type == "서술형":
        return args.scaling_factor_desc, args.min_threshold_desc
    else:
        return args.scaling_factor_obj, args.min_threshold_obj
    

def _maybe_rag(input_data, expanded_question, args, rag_chain, q_type: str):
    '''문서 검색(RAG)'''
    if args.context_type != "rag" or rag_chain is None:
        return None
    scaling, min_th = _select_thresholds(q_type, args)
    
    docs_with_scores = rerank_retrieve_with_dynamic_threshold(
        rag_chain,
        expanded_question,
        input_data,
        k=args.k,
        min_threshold=min_th,
        scaling_factor=scaling
    )
    # print(docs_with_scores)
    context = "\n".join([f"[문서 {i + 1}]\n{document.page_content}" for i, (document, score) in enumerate(docs_with_scores)])
    return context


def _build_few_shot_messages(few_shots, type_instructions_basic, type_instructions_basic_noex, args, rag_chain):
    '''Few-Shot 구성'''
    messages = []
    for i, shot in enumerate(few_shots):
        # 첫 샷만 예시 포함 instruction, 이후는 _noex
        ti = type_instructions_basic if i == 0 else type_instructions_basic_noex
        chat, persona = pipeline_chat_prompt(few_shots, i, ti, args, rag_chain=rag_chain)
        messages.append({"role": "system", "content": persona})
        messages.append({"role": "user", "content": chat})
        messages.append({"role": "assistant", "content": f'{shot["output"]["answer"]}'})
    return messages
    

def _extract_answer(q_type: str, output_text: str):
    '''정답 추출'''
    if q_type == "서술형":
        return {"answer": output_text}

    if "답은" in output_text:
        patterns = list(re.finditer(r'답은\s*(.*)', output_text, re.IGNORECASE))
    elif "답변:" in output_text:
        patterns = list(re.finditer(r'답변:\s*(.*)', output_text, re.IGNORECASE))
    else:
        patterns = list(re.finditer(r'.*', output_text, re.IGNORECASE))

    if not patterns:
        return {"answer": output_text}

    pattern = patterns[-1]
    try:
        if q_type == "선다형":
            match = re.search(r'\b([12345])', pattern.group(1))
            if match:
                answer = match.group(1)
            else:
                answer = pattern.group(1).strip()
        else:
            answer = pattern.group(1).strip().rstrip(".").rstrip("입니다")
        return {"answer": answer}
    except IndexError:
        return {"answer": output_text}



# ---------------------------
# Prompt builders
# ---------------------------
def routing_chat_prompt(result, idx, type_instructions, args, rag_chain=None):
    '''라우팅 프롬프트 구성'''
    input_data = result[idx]['input']
    original_question = input_data['question']
    q_type = input_data['question_type']
    topic_keyword = input_data.get('topic_keyword', "")

    instruction = type_instructions.get(q_type, "")
    chat_parts = [instruction]

    # RAG
    expanded_question = f"{topic_keyword}: {original_question}" if topic_keyword else original_question
    context = _maybe_rag(input_data, expanded_question, args, rag_chain, q_type)
    if context:
        result[idx]['input']['context'] = context
        chat_parts.append(context)

    other_info = {k: v for k, v in input_data.items() if k not in ['question', 'question_type', 'context']}
    if other_info:
        info_list = ["[기타 정보]"]
        for key, value in other_info.items():
            info_list.append(f"- {key}: {value}")
        chat_parts.append("\n".join(info_list))

    chat_parts.append(f"[질문]\n{original_question}")
    chat = "\n\n".join(chat_parts)
    return chat


def pipeline_chat_prompt(result, idx, type_instructions, args, rag_chain=None):
    '''파이프라인 프롬프트 구성'''
    input_data = result[idx]['input']
    original_question = input_data['question']
    q_type = input_data['question_type']

    category = input_data.get('category', '')
    domain = input_data.get('domain', '')
    topic_keyword = input_data.get('topic_keyword', '')

    persona = f"당신은 한국 {category}, {domain}에 대해 잘 아는 한국 특화 인공지능 어시스턴트이다."

    expanded_question = f"{topic_keyword}: {original_question}" if topic_keyword else original_question

    instruction = type_instructions.get(q_type, "")
    chat_parts = [instruction]

    # RAG
    context = _maybe_rag(input_data, expanded_question, args, rag_chain, q_type)
    if context:
        result[idx]['input']['context'] = context
        chat_parts.append(context)

    other_info = {k: v for k, v in input_data.items() if k not in ['question', 'question_type', 'context']}
    if other_info:
        info_list = ["[기타 정보]"]
        for key, value in other_info.items():
            info_list.append(f"- {key}: {value}")
        chat_parts.append("\n".join(info_list))

    chat_parts.append(f"[질문]\n{original_question}")
    chat = "\n\n".join(chat_parts)
    return chat, persona


def rout_basic_infer(tokenizer, model, inp, terminators, device, max_len, temperature, top_p, repetition_penalty, decoding_type, dola_layer):
    '''라우팅(CPU->GPU->CPU)'''
    model.to("cuda:0")
    generation_args = {
        "input_ids": inp.to("cuda:0").unsqueeze(0),
        "max_new_tokens": max_len,
        "eos_token_id": terminators,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True
    }
    if decoding_type == "dola":
        if dola_layer is not None and "[" in dola_layer:
            dola_layer = ast.literal_eval(dola_layer)
        generation_args["dola_layers"] = dola_layer

    outputs = model.generate(**generation_args)
    model.to("cpu")
    torch.cuda.empty_cache()
    return tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)


def basic_infer(tokenizer, model, inp, terminators, device, max_len, temperature, top_p, repetition_penalty, decoding_type, dola_layer):
    generation_args = {
        "input_ids": inp.unsqueeze(0),
        "max_new_tokens": max_len,
        "eos_token_id": terminators,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True
    }
    if decoding_type == "dola":
        if dola_layer is not None and "[" in dola_layer:
            dola_layer = ast.literal_eval(dola_layer)
        generation_args["dola_layers"] = dola_layer

    outputs = model.generate(**generation_args)
    torch.cuda.empty_cache()
    return tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)


def routing_main(args):
    set_seed(args.seed)
    _ensure_output_dir(args.output)

    system_prompt_brief, system_prompt_desc, type_inst, type_inst_noex = _load_prompts(args)
    dev_data, result = _load_data(args.input)

    rag_chain = None
    if args.context_type == "rag":
        rag_chain = RAG_Chain(embedding_path=args.embedding_path,
                              model_name=args.embedding_model,
                              collection_name=args.collection_name)

    m = ModelLoader(device=args.device)
    s = ModelSetting(max_len=args.max_len, temperature=args.temperature, top_p=args.top_p)

    seon_tok, seon_model = m.basic_load(model_id=args.seon_model_id, seed=args.seed, quant=args.quant, token=args.token, decoding_type=args.decoding_type)
    dan_tok,  dan_model  = m.basic_load(model_id=args.dan_model_id,  seed=args.seed, quant=args.quant, token=args.token, decoding_type=args.decoding_type)
    seo_tok,  seo_model  = m.basic_load(model_id=args.seo_model_id,  seed=args.seed, quant=args.quant, token=args.token, decoding_type=args.decoding_type)

    seon_terms = s.basic_setting(seon_tok)
    dan_terms  = s.basic_setting(dan_tok)
    seo_terms  = s.basic_setting(seo_tok)

    for idx in tqdm.tqdm(range(len(result))):
        q_type = result[idx]['input']['question_type']
        system_prompt = system_prompt_desc if q_type == "서술형" else system_prompt_brief

        chat = routing_chat_prompt(result, idx, type_inst, args, rag_chain=rag_chain)
        messages = [{"role": "system", "content": system_prompt}] + [{"role": "user", "content": chat}]

        print(messages)

        if q_type == "선다형":
            tok, model, terms = seon_tok, seon_model, seon_terms
            temperature, top_p = args.temperature, args.top_p
        elif q_type == "단답형":
            tok, model, terms = dan_tok, dan_model, dan_terms
            temperature, top_p = 0.6, 0.9
        else:
            tok, model, terms = seo_tok, seo_model, seo_terms
            temperature, top_p = args.temperature, args.top_p

        chat_args = {"conversation": messages, "add_generation_prompt": True, "enable_thinking": False, "return_tensors": "pt"}
        inp = tok.apply_chat_template(**chat_args)
        output_text = rout_basic_infer(tok, model, inp[0], terms,
                                  args.device, args.max_len,
                                  temperature, top_p,
                                  args.repetition_penalty,
                                  args.decoding_type,
                                  args.dola_layer)

        result[idx]["generation"] = output_text
        print(output_text)
        result[idx]["output"] = _extract_answer(q_type, output_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


def pipeline_main(args):
    set_seed(args.seed)
    _ensure_output_dir(args.output)

    system_prompt_brief, system_prompt_desc, type_inst, type_inst_noex = _load_prompts(args)
    dev_data, result = _load_data(args.input)

    rag_chain = None
    if args.context_type == "rag":
        rag_chain = RAG_Chain(embedding_path=args.embedding_path,
                              model_name=args.embedding_model,
                              collection_name=args.collection_name)

    m = ModelLoader(device=args.device)
    s = ModelSetting(max_len=args.max_len, temperature=args.temperature, top_p=args.top_p)

    tokenizer, model = m.basic_load(model_id=args.model_id, seed=args.seed, quant=args.quant, token=args.token, decoding_type=args.decoding_type)
    terminators = s.basic_setting(tokenizer)

    for idx in tqdm.tqdm(range(len(result))):
        q_type = result[idx]['input']['question_type']
        few_shots = [item for item in dev_data if item["input"]["question_type"] == q_type][4:4 + args.shot]
        system_prompt = system_prompt_desc if q_type == "서술형" else system_prompt_brief

        fs_msgs = _build_few_shot_messages(few_shots, type_inst, type_inst_noex, args, rag_chain)
        chat, persona = pipeline_chat_prompt(result, idx, type_inst, args, rag_chain=rag_chain)
        messages = [{"role": "system", "content": system_prompt + "\n\n" + persona}] + fs_msgs + [{"role": "user", "content": chat}]
        print(messages)

        chat_args = {"conversation": messages, "add_generation_prompt": True, "enable_thinking": False, "return_tensors": "pt"}
        inp = tokenizer.apply_chat_template(**chat_args)
        output_text = basic_infer(tokenizer, model, inp[0], terminators,
                                  args.device, args.max_len,
                                  args.temperature, args.top_p,
                                  args.repetition_penalty,
                                  args.decoding_type,
                                  args.dola_layer)

        result[idx]["generation"] = output_text
        print(output_text)
        result[idx]["output"] = _extract_answer(q_type, output_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.type == "routing":
        routing_main(args)
    else:
        pipeline_main(args)
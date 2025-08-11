import os
import json
from datasets import Dataset
from transformers import AutoTokenizer
import re
from datasets import concatenate_datasets

def load_json_files(directory): 
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'): 
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                data.append(json.load(f))
    return data

def load_json_file(file_directory):
    with open(file_directory, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_dataset(data):
    dataset_dict = {
        "id": [],
        "category": [],
        'domain': [],
        "question_type": [],
        "topic_keyword": [],
        "question": [],
        "answer": []
    }
    
    for item in data:
        dataset_dict["id"].append(item["id"])
        dataset_dict["category"].append(item["input"]["category"])
        dataset_dict["domain"].append(item["input"]["domain"])
        dataset_dict["question_type"].append(item["input"]["question_type"])
        dataset_dict["topic_keyword"].append(item["input"]["topic_keyword"])
        dataset_dict["question"].append(item["input"]["question"])
        dataset_dict["answer"].append(item["output"]["answer"])
        
    return Dataset.from_dict(dataset_dict)

def create_simple_qa_format(question, answer):
    """질문 뒤에 정답을 단순하게 붙이는 형태"""
    question = question.strip()
    answer = answer.strip()
    
    if "#" in answer:
        answer = answer.replace("#", ", ")
    
    return f"{question} {answer}"

def extract_sequence_answer(question, sequence_answer):
    """순서 배열 문제 - ㄱㄴㄷㄹ을 실제 내용으로 변환 (문장 연결 없이 단순 나열)"""
    if "\\n" not in question:
        return sequence_answer
    
    choice_dict = {}
    for char in "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ":
        for pattern in [f"{char})", f"{char}."]:
            if pattern in question:
                start_idx = question.find(pattern) + len(pattern)
                remaining_text = question[start_idx:]
                next_pattern_idx = len(remaining_text)
                for next_char in "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ":
                    for next_pattern in [f"{next_char})", f"{next_char}."]:
                        if next_pattern in remaining_text and next_char != char:
                            idx = remaining_text.find(next_pattern)
                            if idx < next_pattern_idx:
                                next_pattern_idx = idx
                content = remaining_text[:next_pattern_idx].replace("\\n", " ").strip()
                content = content.rstrip(".").strip()
                choice_dict[char] = content
                break

    sequence_chars = []
    if "-" in sequence_answer:
        sequence_chars = [char.strip() for char in sequence_answer.split("-")]
    else:
        sequence_chars = [char for char in sequence_answer if char in choice_dict]

    sequence_contents = [choice_dict[char] for char in sequence_chars if char in choice_dict]

    return ", ".join(sequence_contents)


def convert_mc_to_simple_qa_tabbed(example):
    """
    선다형 문제를 단답형처럼 변환:
    보기에서 정답 번호에 해당하는 선택지만 남기고 질문 뒤에 붙임.
    """
    question = example["question"]
    answer = example["answer"].strip()

    if not answer.isdigit():
        return {"text": question}
    
    if "\\n" not in question:
        return {"text": question}
    
    question_part, choices_part = question.split("\\n", 1)
    
    pattern = r"(\d)\s*\t\s*([^\d]+?)(?=(?:\d\s*\t)|$)"
    choices = dict(re.findall(pattern, choices_part))
    
    correct_choice = choices.get(answer)
    if not correct_choice:
        return {"text": question}
    
    clean_choice = correct_choice.strip()
    final_text = f"{question_part.strip()} {clean_choice}"
    return {"text": final_text}


def convert_to_knowledge_sentence(example):
    """단답형을 간단한 QA 형태로 변환"""
    question = example["question"]
    answer = example["answer"]

    is_sequence_problem = (
        "-" in answer and
        any(char in answer for char in "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ") and
        "\\n" in question and
        any(f"{char})" in question for char in "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
    )

    if is_sequence_problem:
        if "#" in answer:
            answer = answer.split("#")[0].strip()

        final_answer = extract_sequence_answer(question, answer)
        clean_question = question.split("\\n")[0].strip()
        if not final_answer.endswith("."):
            final_answer += "."
        simple_qa = f"{clean_question} {final_answer}"
    else:
        if "따라서 답은" in answer:
            final_answer = answer.split("따라서 답은")[-1].strip().rstrip('.')
        else:
            final_answer = answer.strip()

        simple_qa = create_simple_qa_format(question, final_answer)

    return {"text": simple_qa}

def generate_prompts_enhanced(example, tokenizer):
    descriptive_prompt = '''You are a helpful AI assistant.
        당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트이다. 
        한국어와 영어로 생각하고, 한국어로 답하시오. \
        동일한 문장을 절대 반복하지 마시오.'''
    
    question_type = example["question_type"]
    question = example["question"]
    answer = example["answer"]
    
    if question_type == "선다형":
        instruction = '''[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.
[지침]
주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.'''
    else:
        instruction = '''[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오. 질문에 대한 답변을 완성된 문장으로 서술하시오.'''
    
    other_info = {k: v for k, v in example.items() if k not in ['question', 'question_type', 'answer']}
    
    chat_parts = [instruction]
    if other_info:
        info_list = ["[기타 정보]"]
        for key, value in other_info.items():
            info_list.append(f"- {key}: {value}")
        chat_parts.append("\n".join(info_list))
    
    chat_parts.append(f"[질문]\n{question}")
    user_content = "\n\n".join(chat_parts)
    
    messages = [
        {"role": "system", "content": descriptive_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer}
    ]
    
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": formatted_text}

def generate_prompts_with_labels(example, tokenizer):
    """instruction tuning용 텍스트에 대해 input_ids / labels 생성"""
    descriptive_prompt = '''You are a helpful AI assistant.
    당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트이다. 
    한국어와 영어로 생각하고, 한국어로 답하시오. 동일한 문장을 절대 반복하지 마시오.'''
    
    instruction = '''[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오. 질문에 대한 답변을 완성된 문장으로 서술하시오.'''
    chat_parts = [instruction, f"[질문]\n{example['question']}"]
    user_content = "\n\n".join(chat_parts)
    
    messages = [
        {"role": "system", "content": descriptive_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["answer"]}
    ]
    
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # prompt / answer 분리
    prompt_only = tokenizer.apply_chat_template(messages[:-1], tokenize=False)
    prompt_len = len(tokenizer(prompt_only)["input_ids"])
    full_tokenized = tokenizer(full_text)
    
    input_ids = full_tokenized["input_ids"]
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": full_tokenized["attention_mask"]
    }
    
def prepare_datasets(data_directory, tokenizer):
    if os.path.isdir(data_directory):
        raw_data = load_json_files(data_directory)
    else:
        raw_data = load_json_file(data_directory)

    dataset = create_dataset(raw_data)

    # 1. 단답형만 따로
    dan_data = dataset.filter(lambda x: x["question_type"] == "단답형")
    dan_dataset = dan_data.map(convert_to_knowledge_sentence, remove_columns=dataset.column_names)

    # 2. 선다형만 따로
    mc_data = dataset.filter(lambda x: x["question_type"] == "선다형")
    mc_dataset = mc_data.map(convert_mc_to_simple_qa_tabbed, remove_columns=dataset.column_names)

    # 3. 연속 사전학습용 통합
    continuous_dataset = concatenate_datasets([dan_dataset, mc_dataset])

    # 4. 인스트럭션 튜닝 (서술형만)
    instruction_data = dataset.filter(lambda x: x["question_type"] == "서술형")
    instruction_dataset = instruction_data.map(
        lambda x: generate_prompts_with_labels(x, tokenizer),
        remove_columns=dataset.column_names
    )

    return continuous_dataset, instruction_dataset

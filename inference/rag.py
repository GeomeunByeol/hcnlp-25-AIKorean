# 파일명: RAG.py
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_qdrant import Qdrant, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from typing import List, Dict
import torch
import numpy as np


class RAG_Builder:
    """
    JSON 데이터를 읽어 DB를 구축하고 저장하는 클래스.
    """
    def __init__(self, data_path: str, model_name: str, device: str = 'cpu'):
        self.data_path = data_path
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device, 
                          "model_kwargs": {"torch_dtype": "bfloat16"}},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _load_data(self):
        """JSON 데이터를 로드하여 LangChain Document 형태로 변환합니다."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            # 질문과 답변을 모두 지식으로 활용
            content = f"질문: {item['input']['question']}\n답변: {item['output']['answer']}"
            self.documents.append(Document(page_content=content))

    def build_vector_db(self, save_path: str = "faiss_index"):
        """Vector DB를 구축하고 로컬에 저장합니다."""
        print("1. 데이터 로드를 시작합니다...")
        self._load_data()
        print(f"데이터 로드 완료. 총 {len(self.documents)}개의 문서를 불러왔습니다.")

        print("2. 텍스트 분할을 시작합니다...")
        split_docs = self.text_splitter.split_documents(self.documents)
        print(f"텍스트 분할 완료. 총 {len(split_docs)}개의 청크로 분할되었습니다.")
        
        print("3. Vector DB 구축 및 저장을 시작합니다...")
        vector_db = FAISS.from_documents(split_docs, self.embedding_model)
        vector_db.save_local(save_path)
        print(f"'{save_path}' 경로에 Vector DB 저장이 완료되었습니다.")


class RAG_Chain:
    """
    HuggingFaceEmbeddings를 사용하여 Qdrant DB에서 문서를 검색하는 클래스
    """
    def __init__(self, embedding_path: str, model_name: str, collection_name: str, device: str = 'cpu'):
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device, 
                            "model_kwargs": {"torch_dtype": torch.bfloat16}},
            encode_kwargs={'normalize_embeddings': True}
        )
        client = QdrantClient(path=embedding_path)
        # 로컬에 저장된 Vector DB를 로드합니다.
        self.vector_db = Qdrant(
            client=client, 
            embeddings=embedding_model, 
            collection_name=collection_name
        )

        print("Qdrant 리트리버 준비 완료.")

    def retrieve(self, query: str, k: int = 3, metadata_filter: Dict = None):
        qdrant_filter = None
        if metadata_filter:
            conditions = [
                rest.FieldCondition(key=f"metadata.{key}", match=rest.MatchValue(value=value))
                for key, value in metadata_filter.items() if value
            ]
            if conditions:
                qdrant_filter = rest.Filter(must=conditions)
        return self.vector_db.similarity_search_with_score(query, k=k, filter=qdrant_filter)
    

def rerank_retrieve_with_dynamic_threshold(retriever, question, metadata, k=5, min_threshold=0.6, scaling_factor=1.0):
    candidate_docs = {}
    filter_configs = [
        {"category": metadata.get("category"), "domain": metadata.get("domain"), "topic": metadata.get("topic_keyword")},
        {"category": metadata.get("category"), "domain": metadata.get("domain")},
        {"category": metadata.get("category")}
    ]
    for f in filter_configs:
        valid_filter = {key: val for key, val in f.items() if val}
        if not valid_filter: continue
        for doc, score in retriever.retrieve(question, k=k, metadata_filter=valid_filter):
            if doc.page_content not in candidate_docs:
                candidate_docs[doc.page_content] = (doc, score)
    
    topic = metadata.get("topic_keyword")
    if topic:
        for doc, score in retriever.retrieve(question, k=k, metadata_filter={"topic": topic}):
            if doc.page_content not in candidate_docs:
                candidate_docs[doc.page_content] = (doc, score)

    if not candidate_docs:
        for doc, score in retriever.retrieve(question, k=k):
            candidate_docs[doc.page_content] = (doc, score)
            
    if not candidate_docs:
        return []

    sorted_docs_with_scores = sorted(candidate_docs.values(), key=lambda x: x[1], reverse=True)
    
    scores = [score for doc, score in sorted_docs_with_scores[:k]]
    
    if len(scores) > 1:
        score_avg = np.mean(scores)
        score_std = np.std(scores)
    elif len(scores) == 1:
        score_avg = scores[0]
        score_std = 0
    else:
        return []

    dynamic_threshold = score_avg - (score_std * scaling_factor)
    dynamic_threshold = max(dynamic_threshold, min_threshold)

    final_docs = []
    for doc, score in sorted_docs_with_scores:
        if score >= dynamic_threshold:
            final_docs.append((doc, score))

    return final_docs[:k]
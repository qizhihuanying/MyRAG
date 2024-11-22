import os
import json
import time
import numpy as np

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm import ollama_embedding, ollama_model_complete

def insert_text(rag, file_path):
    with open(file_path, mode="r", encoding="utf-8") as f:
        unique_contexts = json.load(f)

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")

datasets = ["agriculture", "cs", "legal", "mix"]

for cls in datasets:
    WORKING_DIR = f"../{cls}"

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:7b",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
    )

    file_path = f"../datasets/unique_contexts/{cls}_unique_contexts.json"
    insert_text(rag, file_path)

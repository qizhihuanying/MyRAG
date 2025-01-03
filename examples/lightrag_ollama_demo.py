import os
import sys
import logging
import time

import torch
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

total_start_time = time.time()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())

WORKING_DIR = "./results/multi_lightrag/index_results/legal"
time_records = {}  

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
httpx_logger = logging.getLogger("httpx")
httpx_logger.disabled = True # Disable httpx logger
httpx_logger.setLevel(logging.INFO) 
httpx_logger.propagate = False
httpx_logger.handlers = []
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
httpx_logger.addHandler(stream_handler)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

db_start_time = time.time()

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
            texts, embed_model="nomic-embed-text:latest", host="http://localhost:11434"
        ),
    ),
)

# with open("./book.txt", "r", encoding="utf-8") as f:
#     rag.insert(f.read())

db_end_time = time.time()
time_records['Database construction'] = db_end_time - db_start_time

def save_search_results(query, mode, filename):
    query_start_time = time.time()
    result = rag.query(query, param=QueryParam(mode=mode))
    query_end_time = time.time()
    time_records[f"Query time ('{mode}' mode)"] = query_end_time - query_start_time
    
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(result)

model_name = rag.llm_model_name.replace(":", "_")

save_search_results("How does the 'Ratio of Consolidated Indebtedness to Total Capitalization' clause impact the company’s debt strategy?", "my", f"outputs/{model_name}_local_results.md")

total_end_time = time.time()
time_records['Total time'] = total_end_time - total_start_time

for key, value in time_records.items():
    print(f"{key}: {value:.2f} seconds")

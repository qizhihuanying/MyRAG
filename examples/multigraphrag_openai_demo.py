import os
import sys
import logging
import time

import torch
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_embedding, openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '../env', '.env')
print("dotenv_path:", dotenv_path)
load_dotenv(dotenv_path)

total_start_time = time.time()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())

WORKING_DIR = "./dickens"
time_records = {}  

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
httpx_logger = logging.getLogger("httpx")
# httpx_logger.disabled = True # Disable httpx logger
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
    
embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    )

def openai_llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return openai_complete_if_cache(
        # model="gpt-3.5-turbo",
        model="gpt-4-turbo",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),  
        base_url="http://47.89.164.141:9000/v1",  
        # base_url="http://xiaoai.plus/v1",  
        **kwargs,
    )

db_start_time = time.time()

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_llm_model_func,  
    llm_model_name="gpt-4-turbo",  
    llm_model_max_async=4,  
    llm_model_max_token_size=4096, 
    llm_model_kwargs={},  
    embedding_func=embedding_func,  
)

with open("./book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

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

save_search_results("What is the top themes of this article?", "my", f"outputs/{model_name}_local_results.md")

total_end_time = time.time()
time_records['Total time'] = total_end_time - total_start_time

for key, value in time_records.items():
    print(f"{key}: {value:.2f} seconds")

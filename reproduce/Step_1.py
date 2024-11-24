import multiprocessing
import logging
import os
import json
import time
import subprocess

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm import ollama_embedding, ollama_model_complete

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.INFO)
httpx_logger.propagate = False
httpx_logger.handlers = []
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
httpx_logger.addHandler(stream_handler)

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

def start_ollama_server(port, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OLLAMA_HOST"] = f"0.0.0.0:{port}"
    process = subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process

def process_dataset(cls, port):
    time.sleep(10) 

    WORKING_DIR = f"../{cls}"

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:14b",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={"host": f"http://localhost:{port}", "options": {"num_ctx": 32768}},
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts, embed_model="nomic-embed-text", host=f"http://localhost:{port}"
            ),
        ),
    )

    file_path = f"../datasets/unique_contexts/{cls}_unique_contexts.json"
    insert_text(rag, file_path)

if __name__ == "__main__":
    datasets = ["mix", "agriculture", "cs", "legal"]
    ports = [11434, 11435, 11436, 11437]  # 每个 Ollama 服务器实例的端口
    gpu_ids = [0, 1, 2, 3]  # 分配给每个进程的 GPU ID

    servers = []
    for port, gpu_id in zip(ports, gpu_ids):
        server = start_ollama_server(port, gpu_id)
        servers.append(server)

    time.sleep(10)  # 稍微延长时间以确保服务器完全启动

    processes = []
    for cls, port in zip(datasets, ports):
        p = multiprocessing.Process(target=process_dataset, args=(cls, port))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for server in servers:
        server.terminate()
        server.wait() 

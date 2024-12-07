import multiprocessing
import logging
import os
import sys
import time
import subprocess
import json
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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

def process_dataset(cls, port, gpu_id):
    time.sleep(10)

    model = "lightrag"
    WORKING_DIR = f"./results/{model}/index_results/{cls}"

    os.makedirs(WORKING_DIR, exist_ok=True)

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

    file_path = f"./datasets/unique_contexts/{cls}_unique_contexts.json"
    insert_text(rag, file_path)

def get_free_gpu(threshold=18):
    free_gpus = set()

    nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"])
    nvidia_smi_output = nvidia_smi_output.decode("utf-8").strip().splitlines()

    for line in nvidia_smi_output:
        gpu_id, memory_free = map(str.strip, line.split(','))
        memory_free = int(memory_free.split()[0]) 
        if memory_free >= threshold * 1024:  
            free_gpus.add(int(gpu_id))

    return free_gpus

def assign_gpus(datasets, ports, gpu_ids):
    assigned_gpus = set()
    tasks_to_assign = list(zip(datasets, ports))

    while tasks_to_assign:
        free_gpus = get_free_gpu()
        free_gpus -= assigned_gpus

        if free_gpus:
            gpu_id = free_gpus.pop()
            cls, port = tasks_to_assign.pop(0)

            print(f"Assigning task for {cls} to GPU {gpu_id} on port {port}")
            assigned_gpus.add(gpu_id)

            p = multiprocessing.Process(target=process_dataset, args=(cls, port, gpu_id))
            p.start()

        else:
            print("No free GPU available. Waiting...")
            time.sleep(5)

    return tasks_to_assign  

if __name__ == "__main__":
    datasets = ["mix", "agriculture", "cs", "legal"]
    ports = [11434, 11435, 11436, 11437]
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]  

    servers = []
    for port, gpu_id in zip(ports, gpu_ids):
        server = start_ollama_server(port, gpu_id)
        servers.append(server)

    remaining_tasks = assign_gpus(datasets, ports, gpu_ids)
    for p in multiprocessing.active_children():
        p.join()

    for server in servers:
        server.terminate()
        server.wait()

    print("All tasks completed.")

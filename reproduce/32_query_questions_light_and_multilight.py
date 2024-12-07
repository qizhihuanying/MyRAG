import os
import sys
import re
import json
import asyncio
from lightrag import LightRAG, QueryParam
from tqdm import tqdm
from lightrag.llm import ollama_model_if_cache, ollama_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
import ollama

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await ollama_embedding(
        texts,
        embed_model="nomic-embed-text",
        host="http://localhost:11434",  
    )

def extract_queries(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    data = data.replace("**", "")
    queries = re.findall(r"- Question \d+: (.+)", data)
    return queries

async def process_query(query_text, rag_instance, query_param):
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}

def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    loop = always_get_an_event_loop()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(error_file), exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as result_file, open(
        error_file, "a", encoding="utf-8"
    ) as err_file:
        result_file.write("[\n")
        first_entry = True

        for query_text in tqdm(queries, desc="Processing queries", unit="query"):
            result, error = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )

            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

        result_file.write("\n]")

if __name__ == "__main__":

    classes = ["agriculture", "cs", "legal", "mix"]
    models = ["multi_lightrag", "lightrag"]

    for cls in classes:
        for model in models:
            print(f"Processing class: {cls} with mode: {model}")

            WORKING_DIR = f"./results/{model}/index_results/{cls}"
            QUERY_RESULTS_DIR = f"./results/{model}/query_results/{cls}"

            os.makedirs(WORKING_DIR, exist_ok=True)
            os.makedirs(QUERY_RESULTS_DIR, exist_ok=True)

            rag = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=ollama_model_complete,
                llm_model_name="qwen2.5:7b", 
                llm_model_max_async=4,
                llm_model_max_token_size=32768,
                llm_model_kwargs={
                    "host": "http://localhost:11434", 
                    "options": {"num_ctx": 32768},
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=768,
                    max_token_size=8192,
                    func=embedding_func,
                ),
            )

            query_param = QueryParam(mode="hybrid")

            base_dir = "./datasets/questions"
            queries = extract_queries(f"{base_dir}/{cls}_questions.txt")

            run_queries_and_save_to_json(
                queries,
                rag,
                query_param,
                f"{QUERY_RESULTS_DIR}/result.json",  
                f"{QUERY_RESULTS_DIR}/errors.json",  
            )

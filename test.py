import torch
import asyncio
from lightrag.llm import ollama_embedding
from lightrag.utils import EmbeddingFunc

# 将EmbeddingFunc定义为异步
async def async_embedding_func(texts):
    return await ollama_embedding(
        texts, embed_model="nomic-embed-text", host="http://localhost:11434"
    )

kw_1 = "ensure"
kw_2 = "INVITROGEN SAN RAFFAELE SCIENTIFIC INSTITUTE collaboration"

# 定义一个异步函数来计算余弦相似度
async def compute_similarity():
    embedding_1 = (await async_embedding_func([kw_1]))[0]
    embedding_2 = (await async_embedding_func([kw_2]))[0]

    # 计算两个关键词的余弦相似度,不用torch
    similarity = sum(a * b for a, b in zip(embedding_1, embedding_2)) / (
        (sum(a ** 2 for a in embedding_1) ** 0.5) * (sum(a ** 2 for a in embedding_2) ** 0.5)
    )
    print(f"Similarity between '{kw_1}' and '{kw_2}': {similarity}")
    

# 运行异步函数
asyncio.run(compute_similarity())

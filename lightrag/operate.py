import asyncio
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Tuple, Union
from collections import Counter, defaultdict
import warnings
import numpy as np

from numpy import dot
from torch import norm
import torch
from tqdm import tqdm
from lightrag.llm import ollama_embedding, ollama_model_complete
from lightrag.storage import NetworkXStorage
from .utils import (
    EmbeddingFunc,
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    locate_json_string_body_from_string,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: Union[str, Tuple[str, str], Tuple[str, str, str]],
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )

    # 根据传入的 entity_or_relation_name 构建 entity_name
    if isinstance(entity_or_relation_name, tuple):
        if len(entity_or_relation_name) == 3:
            src_id, tgt_id, keyword = entity_or_relation_name
            entity_name = f"relation: from {src_id} to {tgt_id}, keyword: {keyword}"
        else:
            entity_name = "UNKNOWN RELATIONSHIP"
    else:
        entity_name = entity_or_relation_name

    context_base = dict(
        entity_name=entity_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary

async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    
    edge_keywords = edge_keywords.strip().strip('"').strip("'")
    
    keywords_list = split_string_by_multi_markers(
        edge_keywords, [',']
    )
    
    edges_data = []
    for keyword in keywords_list:
        keyword = keyword.strip().strip('"').strip("'")
        if keyword:
            edge_data = dict(
                src_id=source,
                tgt_id=target,
                weight=weight,
                description=edge_description,
                keyword=keyword, 
                source_id=edge_source_id,
            )
            edges_data.append(edge_data)

    return edges_data if edges_data else None
    
    # return dict(
    #     src_id=source,
    #     tgt_id=target,
    #     weight=weight,
    #     description=edge_description,
    #     keywords=edge_keywords,
    #     source_id=edge_source_id,
    # )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    keyword: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    # already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id, keyword):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id, keyword)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        # already_keywords.extend(
        #     split_string_by_multi_markers(already_edge["keyword"], [GRAPH_FIELD_SEP])
        # )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    # keywords = GRAPH_FIELD_SEP.join(
    #     sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    # )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id, keyword), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        keyword,
        edge_data=dict(
            weight=weight,
            description=description,
            keyword=keyword,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keyword=keyword,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relations = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relations is not None:
                for if_relation in if_relations:
                    maybe_edges[(if_relation["src_id"], if_relation["tgt_id"], if_relation["keyword"])].append(
                        if_relation
                    )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        logger.info(
            f"{now_ticks} Processed {already_processed} chunks, "
            f"{already_entities} entities(duplicated), "
            f"{already_relations} relations(duplicated)\r"
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            src_id, tgt_id, keyword = k
            # 对 src_id 和 tgt_id 排序，但保留 keyword 的位置
            sorted_nodes = tuple(sorted([src_id, tgt_id]))
            new_key = (*sorted_nodes, keyword)
            maybe_edges[new_key].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], k[2], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"] + dp["keyword"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "keyword": dp["keyword"],
                "content": dp["keyword"]
                + dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    )

class MY_Cache:
    def __init__(self, embedding_func, batch_size=20000):
        self.embedding_func = embedding_func
        self.embedding_cache = {}
        self.edge_cache = {}
        self.lock = asyncio.Lock()
        self.batch_size = batch_size
        self.embedding_access_count = 0
        self.embedding_hit_count = 0
        self.edge_access_count = 0
        self.edge_hit_count = 0

    async def get_embeddings(self, keywords: List[str]) -> Dict[str, List[float]]:
        missing = [kw for kw in keywords if kw not in self.embedding_cache]
        self.embedding_access_count += len(keywords)
        self.embedding_hit_count += (len(keywords) - len(missing))

        if missing:
            for i in range(0, len(missing), self.batch_size):
                batch = missing[i:i + self.batch_size]
                embeddings = await self.embedding_func.func(batch)
                for kw, emb in zip(batch, embeddings):
                    self.embedding_cache[kw] = emb
        return {kw: self.embedding_cache[kw] for kw in keywords}
    
    async def get_similarities_batch(self, edge_pairs: set) -> Dict[tuple, float]:
        """
        Computes similarities for a batch of (relation_kw, edge_kw) pairs using vectorized operations.
        Returns a dictionary mapping (relation_kw, edge_kw) to similarity.
        """
        # Extract all unique keywords
        keywords = {str(kw) for pair in edge_pairs for kw in pair}
        
        # Get embeddings for all keywords
        embeddings = await self.get_embeddings(list(keywords))
        
        # Convert embeddings into a matrix
        keywords_list = list(keywords)
        embedding_matrix = np.array([embeddings[kw] for kw in keywords_list])
        
        # Normalize the embeddings to unit vectors
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # To avoid division by zero
        normalized_embeddings = embedding_matrix / norms
        
        # Create a map from keyword to its index in the matrix
        keyword_to_index = {kw: i for i, kw in enumerate(keywords_list)}
        
        # Compute all similarities at once using matrix multiplication
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Extract the similarities for the requested pairs
        similarities = {}
        for relation_kw, edge_kw in edge_pairs:
            idx1 = keyword_to_index[str(relation_kw)]
            idx2 = keyword_to_index[str(edge_kw)]
            similarity = similarity_matrix[idx1, idx2]
            similarities[(relation_kw, edge_kw)] = similarity

        return similarities
    
    async def get_edges(self, node: str, graph: "BaseGraphStorage") -> List[tuple]:
        self.edge_access_count += 1
        if node in self.edge_cache:
            self.edge_hit_count += 1
            return self.edge_cache[node]

        edges = await graph.get_node_edges(node)
        self.edge_cache[node] = edges
        return edges

    def embedding_cache_hit_rate(self):
        if self.embedding_access_count == 0:
            return 0.0
        return self.embedding_hit_count / self.embedding_access_count

    def edge_cache_hit_rate(self):
        if self.edge_access_count == 0:
            return 0.0
        return self.edge_hit_count / self.edge_access_count
    
    def reset_cache(self):
        """Reset both embedding and edge caches."""
        self.embedding_cache.clear()
        self.edge_cache.clear()
        self.embedding_access_count = 0
        self.embedding_hit_count = 0
        self.edge_access_count = 0
        self.edge_hit_count = 0

# 初始化全局缓存
my_cache = MY_Cache(embedding_func)
rwr_semaphore = asyncio.Semaphore(1000)
edges_scores = {}

async def precompute_embeddings_and_similarities(
    relation_initial_nodes_groups: List[Tuple[str, List[str]]],
    knowledge_graph_inst: "BaseGraphStorage",
    my_cache: MY_Cache,
    working_dir: str = "./temp"
) -> Dict[Tuple[str, str], float]:
    
    # 确保工作目录存在
    os.makedirs(working_dir, exist_ok=True)
    edge_embeddings_file = os.path.join(working_dir, "edge_embeddings.npz")
    
    # 1. 收集所有 relation_kw
    all_relation_kws = set(r_kw for r_kw, _ in relation_initial_nodes_groups)
    
    # 2. 从图中获取所有边以提取 edge_kw
    all_graph_edges = await knowledge_graph_inst.get_all_edges()
    all_edge_kws = set(edge[2] for edge in all_graph_edges)

    # 3. 加载已有的 edge embeddings
    loaded_edge_kws = []
    loaded_e_emb_matrix_norm = None
    if os.path.exists(edge_embeddings_file):
        try:
            data = np.load(edge_embeddings_file, allow_pickle=True)
            loaded_edge_kws = data["edge_kw_list"].tolist()
            loaded_e_emb_matrix_norm = data["edge_embeddings"]
            print(f"已加载 {len(loaded_edge_kws)} 条边的嵌入。")
        except Exception as e:
            print(f"加载边嵌入文件失败: {e}")
            loaded_edge_kws = []
            loaded_e_emb_matrix_norm = None

    # 4. 查找需要新计算的 edge_kw
    new_edge_kws = list(all_edge_kws - set(loaded_edge_kws))
    print(f"需要新计算的边关键词数量: {len(new_edge_kws)}")

    # 5. 计算新 edge_kw 的 embeddings 并归一化
    if new_edge_kws:
        # 获取新的边关键词嵌入
        new_edge_embeddings = await my_cache.get_embeddings([str(e_kw) for e_kw in new_edge_kws])
        new_e_emb_matrix = np.array([new_edge_embeddings[str(e_kw)] for e_kw in new_edge_kws])

        # 归一化
        new_e_norms = np.linalg.norm(new_e_emb_matrix, axis=1, keepdims=True)
        new_e_norms[new_e_norms == 0] = 1
        new_e_emb_matrix_norm = new_e_emb_matrix / new_e_norms

        # 合并已有的和新的
        if loaded_edge_kws and loaded_e_emb_matrix_norm is not None:
            final_edge_kws = loaded_edge_kws + new_edge_kws
            final_e_emb_matrix_norm = np.concatenate([loaded_e_emb_matrix_norm, new_e_emb_matrix_norm], axis=0)
        else:
            final_edge_kws = new_edge_kws
            final_e_emb_matrix_norm = new_e_emb_matrix_norm

        # 保存更新后的 edge embeddings
        try:
            np.savez(edge_embeddings_file, 
                     edge_kw_list=np.array(final_edge_kws, dtype=object),
                     edge_embeddings=final_e_emb_matrix_norm)
            print(f"已保存 {len(new_edge_kws)} 条新边的嵌入，总边数量: {len(final_edge_kws)}")
        except Exception as e:
            print(f"保存边嵌入文件失败: {e}")
    else:
        if loaded_edge_kws:
            final_edge_kws = loaded_edge_kws
            final_e_emb_matrix_norm = loaded_e_emb_matrix_norm
        else:
            final_edge_kws = []
            final_e_emb_matrix_norm = np.array([])

    # 6. 获取query中提取的relation_keyword的嵌入
    all_need_kws = list(all_relation_kws)
    all_need_kws = [str(kw) for kw in all_need_kws]
    batch_size = my_cache.batch_size
    relation_embeddings = {}

    for i in tqdm(range(0, len(all_need_kws), batch_size), desc="预计算关系关键词嵌入"):
        batch = all_need_kws[i:i + batch_size]
        batch_embeddings = await my_cache.get_embeddings(batch)
        relation_embeddings.update(batch_embeddings)

    # 7. 处理relation_keyword的嵌入并归一化
    r_kws_list = list(relation_embeddings.keys())
    r_emb_matrix = np.array([relation_embeddings[r] for r in r_kws_list])
    r_norms = np.linalg.norm(r_emb_matrix, axis=1, keepdims=True)
    r_norms[r_norms == 0] = 1
    r_emb_matrix_norm = r_emb_matrix / r_norms

    # 8. 计算相似度矩阵
    if final_e_emb_matrix_norm.size == 0:
        print("没有边关键词的嵌入可用于计算相似度。")
        return {}
    
    similarity_matrix = np.dot(r_emb_matrix_norm, final_e_emb_matrix_norm.T)

    # 9. 构建相似度字典
    similarity_dict = {}
    for i, r_kw in enumerate(r_kws_list):
        for j, e_kw in enumerate(final_edge_kws):
            similarity_dict[(r_kw, e_kw)] = float(similarity_matrix[i, j])

    print(f"已计算 {len(similarity_dict)} 对关键词的相似度。")
    return similarity_dict



async def my_query(
    query: str,
    knowledge_graph_inst: "BaseGraphStorage",
    entities_vdb: "BaseVectorStorage",
    relationships_vdb: "BaseVectorStorage",
    text_chunks_db: "BaseKVStorage[TextChunkSchema]",
    query_param: "QueryParam",
    global_config: dict,
    working_dir: str="./temp"
) -> str:
    use_model_func = global_config["llm_model_func"]


    print("Question: ", query)
    
    # 第1步：关键词提取
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        entity_keywords = keywords_data.get("entity_keywords", [])
        relation_keywords = keywords_data.get("relation_keywords", [])
    except json.JSONDecodeError:
        # 尝试另一种解析
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json.loads(result)
            entity_keywords = keywords_data.get("entity_keywords", [])
            relation_keywords = keywords_data.get("relation_keywords", [])
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]

    if not entity_keywords and not relation_keywords:
        return PROMPTS["fail_response"]
    print("entity keywords: ", entity_keywords)
    print("relation keywords: ", relation_keywords)

    # 设置总需要的Top K数量
    top_k = 60

    # 第2步：利用关键词从entities_vdb和relationships_vdb中检索初始实体和关系集合
    # 2.1 对每个entity_keyword单独查询
    entity_tasks = [
        entities_vdb.query(kw, top_k=top_k) for kw in entity_keywords
    ]
    entity_results_list = await asyncio.gather(*entity_tasks)
    entity_results = [item for sublist in entity_results_list for item in sublist]

    # 2.2 对每个relation_keyword单独查询
    relation_tasks = [
        relationships_vdb.query(kw, top_k=top_k ) for kw in relation_keywords
    ]
    relation_results_list = await asyncio.gather(*relation_tasks)

    # 2.3 聚集实体
    initial_nodes = set()
    for r in entity_results:
        initial_nodes.add(r["entity_name"])

    # 2.4 处理每个relation_keyword的关系
    relation_initial_nodes_groups = []  # 每个group对应一个relation_keyword
    for idx, relation_results in enumerate(relation_results_list):
        current_relation_kw = relation_keywords[idx]
        # 提取relations
        current_relations = [ (r["src_id"], r["tgt_id"], r["keyword"]) for r in relation_results ]
        initial_edges = current_relations

        # 将关系对应的节点加入初始节点集合
        current_initial_nodes = set(initial_nodes)  
        for (s, _, _) in initial_edges:
            current_initial_nodes.add(s)

        relation_initial_nodes_groups.append( (current_relation_kw, list(current_initial_nodes)) )
      
    # 定义一个异步锁用于同步进度条更新
    pbar_lock = asyncio.Lock()
    
    similarity_cache = await precompute_embeddings_and_similarities(
        relation_initial_nodes_groups,
        knowledge_graph_inst,
        my_cache,
        working_dir=working_dir
    )

    # 第3步：基于多重图的加权随机游走（Weighted RWR） - 对每个relation_keyword单独执行
    async def perform_weighted_rwr(
        start_nodes_name: List[str],
        graph: "BaseGraphStorage",
        relation_kw: str,
        walk_steps: int = 10,
        restart_prob: float = 0.05,
        paths_to_collect: int = 10
    ) -> List[List[tuple]]:
        paths = []
        total_paths = len(start_nodes_name) * (paths_to_collect // max(1, len(start_nodes_name)))
        paths_per_node = paths_to_collect // max(1, len(start_nodes_name))
        total_steps = paths_per_node * walk_steps * len(start_nodes_name)
        print("initiate with total nodes: ", len(start_nodes_name))
        # print("nodes name: ", start_nodes_name)

        async with rwr_semaphore:
            # 初始化进度条
            async with pbar_lock:
                pbar = tqdm(total=total_steps, desc="Performing Weighted RWR (steps)", mininterval=1.0)

            async def walk(start_node_name: str) -> List[List[tuple]]:
                local_paths = []
                visited_edges = set()

                path = []
                current_node_name = start_node_name
                
                for step in range(walk_steps):
                    edges = await my_cache.get_edges(current_node_name, graph)

                    if not edges:
                        break

                    all_edge_pairs = set()
                    for edge in edges:
                        _, target_node_name, edge_kw, _ = edge
                        edge_id = (relation_kw, edge_kw)
                        all_edge_pairs.add(edge_id)

                    if not all_edge_pairs:
                        break

                    similarities = {pair: similarity_cache.get(pair, 0.0) for pair in all_edge_pairs}
                    
                    edges_info = []
                    for edge in edges:
                        _, target_node_name, edge_kw, data = edge
                        edge_id = (relation_kw, edge_kw)
                        
                        if (current_node_name, target_node_name, edge_kw) in visited_edges:
                            continue

                        sim = similarities.get(edge_id, 0.0)
                        w = data.get("weight", 1.0)
                        score = w * sim
                        # 用sim而不是score作为选择的依据，但仍然记录score
                        if sim > 0:
                            edges_info.append((current_node_name, target_node_name, edge_kw, data, sim, score))
                            edges_scores[(current_node_name, target_node_name, edge_kw)] = score
                        else:
                            # print("similarity is 0: ", edge_id)
                            pass

                    if not edges_info:
                        break

                    # 使用sim作为随机选择的权重 (edges_info的第五个元素)
                    weights = [item[4] for item in edges_info]
                    chosen = random.choices(edges_info, weights=weights, k=1)[0]
                    _, chosen_node_name, chosen_edge_kw, chosen_edge_data, chosen_sim, chosen_score = chosen

                    visited_edges.add((current_node_name, chosen_node_name, chosen_edge_kw))
                    visited_edges.add((chosen_node_name, current_node_name, chosen_edge_kw))

                    path.append((current_node_name, chosen_edge_data))
                    current_node_name = chosen_node_name
                    
                    if random.random() < restart_prob:
                        current_node_name = start_node_name

                    async with pbar_lock:
                        pbar.update(1)

                # 添加路径最后一个节点
                path.append((current_node_name, None))
                local_paths.append(path)                
                return local_paths

            # 创建所有walk任务
            walk_tasks = [walk(node) for node in start_nodes_name]
            walk_results = await asyncio.gather(*walk_tasks)

            # 关闭进度条
            async with pbar_lock:
                pbar.close()

            # 收集所有路径
            for walk_path in walk_results:
                paths.extend(walk_path)

        return paths


    all_paths = []
    tasks = []
    for relation_kw, start_nodes_name in relation_initial_nodes_groups:
        task = perform_weighted_rwr(
            start_nodes_name=start_nodes_name,
            graph=knowledge_graph_inst,
            relation_kw=relation_kw,
            walk_steps=500,           # 调整后的步数
            restart_prob=0.15,       # 调整后的重启概率
            paths_to_collect=10      # 调整后的路径数
        )
        tasks.append(task)

    # 并行执行所有RWR任务
    all_paths_results = await asyncio.gather(*tasks)
    for paths in all_paths_results:
        all_paths.extend(paths)
    
    os.makedirs("analyze", exist_ok=True)
    save_paths_to_file(all_paths, 'analyze/paths.json')
        
    print("cached embedding hit rate: ", my_cache.embedding_cache_hit_rate())
    print("cached edge hit rate: ", my_cache.edge_cache_hit_rate())

    # 第4步：从筛选出的路径中提取节点、边以及相应文本构建上下文，并进行评分
    # 收集路径中的节点和边
    selected_nodes = set()
    selected_edges = []
    for path in all_paths:
        for i in range(len(path) - 1):
            src_node = path[i][0]
            edge_info = path[i][1]
            tgt_node = path[i + 1][0]
            if edge_info is not None:
                selected_nodes.add(src_node)
                selected_nodes.add(tgt_node)
                selected_edges.append((src_node, tgt_node, edge_info))
        # 添加路径最后一个节点
        if path:
            selected_nodes.add(path[-1][0])

    selected_nodes = list(selected_nodes)

    # 异步获取节点信息和节点度数
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(n) for n in selected_nodes]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(n) for n in selected_nodes]
    )
    node_info = {
        node: {"data": data, "degree": degree}
        for node, data, degree in zip(selected_nodes, node_datas, node_degrees)
        if data is not None
    }

    # 第5步：评分路径
    def score_path(path: List[Tuple[str, Dict[str, Any]]]) -> float:
        edges_info = [e for (_, e) in path if e is not None]
        total_edge_weight = sum(e.get("weight", 1.0) for e in edges_info)
        total_kw_match = sum(edges_scores.get((path[i][0], path[i + 1][0], e.get("keyword")), 0.0) for i, e in enumerate(edges_info[:-1]))
        node_degree_sum = sum(node_info[n]["degree"] for n, _ in path if n in node_info)
        path_length = len(path)
        alpha, beta, gamma, delta = 1.0, 2.0, 1.0, 0.5
        score = alpha * total_edge_weight + beta * total_kw_match + gamma * node_degree_sum + delta * path_length
        return score

    scored_top_k = 1000
    scored_paths = [(p, score_path(p)) for p in all_paths]
    scored_paths.sort(key=lambda x: x[1], reverse=True)
    top_paths = [p for p, s in scored_paths[:scored_top_k]] if scored_paths else []

    if not top_paths:
        return PROMPTS["fail_response"]

    selected_nodes = set()
    selected_edges = []
    for path in top_paths:
        for i in range(len(path) - 1):
            src_node = path[i][0]
            edge_info = path[i][1]
            tgt_node = path[i + 1][0]
            if edge_info is not None:
                selected_nodes.add(src_node)
                selected_nodes.add(tgt_node)
                selected_edges.append((src_node, tgt_node, edge_info))
        if path:
            selected_nodes.add(path[-1][0])

    selected_nodes = list(selected_nodes)

    # 使用已经存在的node_data和edge查询信息来构建上下文
    node_datas = []
    for n in selected_nodes:
        info = node_info[n]["data"]
        deg = node_info[n]["degree"]
        node_datas.append({**info, "entity_name": n, "rank": deg})
        
    # 获取文本信息
    # 节点中有source_id，通过text_chunks_db获取文本块
    all_text_ids = set()
    for n in node_datas:
        if "source_id" in n:
            cids = split_string_by_multi_markers(n["source_id"], [GRAPH_FIELD_SEP])
            for cid in cids:
                all_text_ids.add(cid)

    # 处理边的source_id，如果存在
    for (_, _, e_info) in selected_edges:
        if "source_id" in e_info:
            cids = split_string_by_multi_markers(e_info["source_id"], [GRAPH_FIELD_SEP])
            for cid in cids:
                all_text_ids.add(cid)

    all_text_units = []
    for cid in all_text_ids:
        chunk_data = await text_chunks_db.get_by_id(cid)
        if chunk_data and "content" in chunk_data:
            all_text_units.append(chunk_data)

    # 截断文本超长部分
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    # 对edges进行结构化处理
    edges_context_data = []
    for i, (s, t, e_info) in enumerate(selected_edges):
        # 计算edge的rank(即edge_degree)
        deg = await knowledge_graph_inst.edge_degree(s, t)
        edges_context_data.append({
            "src_tgt": (s, t),
            "description": e_info.get("description", "UNKNOWN"),
            "keyword": e_info.get("keyword", "UNKNOWN"),
            "weight": e_info.get("weight", 1.0),
            "rank": deg
        })

    # 截断边的描述文本
    edges_context_data = truncate_list_by_token_size(
        edges_context_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    # 构建CSV上下文
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keyword", "weight", "rank"]
    ]
    for i, e in enumerate(edges_context_data):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keyword"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(all_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    context = f"""
-----Entities-----
csv
{entities_context}

-----Relationships-----
csv
{relations_context}
-----Sources-----
csv
{text_units_context}

"""

    if query_param.only_need_context:
        return context

    if context is None:
        return PROMPTS["fail_response"]

    # 第6步：调用模型生成最终回答
    print("Start generating response...")
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    print("Done.")
    return response

def save_paths_to_file(paths, filename):
    serializable_paths = []
    for path in paths:
        serializable_path = []
        for node, edge in path:
            if edge:
                keyword = edge.get('keyword', '')
            else:
                keyword = None
            serializable_path.append({
                'entity': node,
                'relation_keyword': keyword
            })
        serializable_paths.append(serializable_path)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_paths, f, ensure_ascii=False, indent=4)

async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["llm_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response

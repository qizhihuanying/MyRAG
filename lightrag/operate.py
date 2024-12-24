import asyncio
import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Set, Tuple, Union
from collections import Counter, defaultdict, deque
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
    # print("source: ", source)
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    
    keywords_list = split_string_by_multi_markers(
        edge_keywords, [',']
    )
    
    edges_data = []
    for keyword in keywords_list:
        keyword = clean_str(keyword).replace('<', '').replace('>', '').replace('("', '').replace('")', '').replace('<', '').replace('>', '').strip().strip('"')
        keyword = f'"{keyword}"'
        # print("keyword: ", keyword)
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

async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []
    
    entity_name = clean_str(entity_name)

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
        set([dp["description"] for dp in nodes_data] + already_description)
    )
    # 定义history属性存储所有description的拼接
    history = GRAPH_FIELD_SEP.join(
        set([dp["description"] for dp in nodes_data] + already_description)
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=description,
        history=history,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=dict(
        entity_type=entity_type,
        description=description,
        history=history,
        source_id=source_id,
        ),
    )
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
    
    src_id = clean_str(src_id)
    tgt_id = clean_str(tgt_id)
    keyword = clean_str(keyword)

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
        set([dp["description"] for dp in edges_data] + already_description)
    )
    # 定义history属性存储所有description的拼接
    history = GRAPH_FIELD_SEP.join(
        set([dp["description"] for dp in edges_data] + already_description)
    )
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
            keyword=keyword,
            weight=weight,
            description=description,
            history=history,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        keyword=keyword,
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        history=history,
        weight=weight,
        source_id=source_id,
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
    
    # 1. 在图构建完成后，遍历该多重图的所有节点，首先将每个节点的history用大模型做一次summary，然后将summary的结果作为节点的description，对边也做同样的处理 
    # 2. 在上面遍历的同时，每个节点构建指向自己的一条边, keyword为自己的entity_name，weight为节点的度数，description为节点的description 
    
    # 同步构建边时临时插入图中的节点到数据库中
    all_nodes = await knowledge_graph_inst.get_all_nodes()
    missing_nodes = set(all_nodes) - set([dp["entity_name"] for dp in all_entities_data])
    for node in missing_nodes:
        node_data = await knowledge_graph_inst.get_node(node)
        all_entities_data.append(
            dict(
                entity_name=node,
                entity_type=node_data["entity_type"],
                description=node_data["description"],
                history=node_data["description"],
                source_id=node_data["source_id"],
            )
        )
    for node_data in all_entities_data:
        print("node_names: ", node_data["entity_name"])
        # 构建指向自己的边并存到数据库中
        all_relationships_data.append(
            dict(
                src_id=node_data["entity_name"],
                tgt_id=node_data["entity_name"],
                keyword=node_data["entity_name"],
                description=node_data["description"],
                history=node_data["history"],
                weight = len(await knowledge_graph_inst.get_node_edges(node_data["entity_name"])) if await knowledge_graph_inst.get_node_edges(node_data["entity_name"]) is not None else 0,
                source_id=node_data["source_id"],
            )
        )
        entity_name = node_data["entity_name"]
        history = node_data["history"]
        all_description = await _handle_entity_relation_summary(
            entity_name, history, global_config
        )
        # 将节点的description更新为summary的结果
        await knowledge_graph_inst.upsert_node(
            entity_name,
            node_data=dict(
            entity_type=node_data["entity_type"],
            description=all_description,
            history=node_data["history"],
            source_id=node_data["source_id"],
            ),
        )
    print("all nodes in graph: ", await knowledge_graph_inst.get_all_nodes())
    for edge_data in all_relationships_data:
        src_id = edge_data["src_id"]
        tgt_id = edge_data["tgt_id"]
        keyword = edge_data["keyword"]
        history = edge_data["history"]
        all_description = await _handle_entity_relation_summary(
            (src_id, tgt_id, keyword), history, global_config
        )
        await knowledge_graph_inst.upsert_edge(
            src_id,
            tgt_id,
            keyword,
            edge_data=dict(
                keyword=keyword,
                weight=edge_data["weight"],
                description=all_description,
                history=history,
                source_id=edge_data["source_id"],
            ),
        )
    
    # 整个图构建完成以后，将所有的实体和关系数据存储到向量数据库中
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
                "entity_name": dp["entity_name"],
                "content": f"The entity '{dp['entity_name']}' is described as: {dp['description']}.",
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
                "content": f"The relationship '{dp['keyword']}' connects source ID '{dp['src_id']}' with target ID '{dp['tgt_id']}', and is described as: {dp['description']}.",
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


def decide_top_k_by_similarity(sim: float) -> int:
    """
    根据节点与 relation 内容的相似度分段来选择 top k 条边。
    你可以根据需要修改这个划分规则。
    """
    if sim > 0.9:
        return 100
    elif sim > 0.7:
        return 80
    elif sim > 0.5:
        return 60
    elif sim > 0.3:
        return 40
    elif sim > 0.1:
        return 20
    else:
        return 0

embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    )

edges_scores = {}

async def my_query(
    query: str,
    knowledge_graph_inst: "BaseGraphStorage",
    entities_vdb: "BaseVectorStorage",
    relationships_vdb: "BaseVectorStorage",
    text_chunks_db: "BaseKVStorage[TextChunkSchema]",
    query_param: "QueryParam",
    global_config: dict,
    working_dir: str = "./temp"
) -> str:
    use_model_func = global_config["llm_model_func"]

    print("Question: ", query)
    
    # 第1步：详细信息提取
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        entity_info = keywords_data.get("entity_info", [])
        relation_info = keywords_data.get("relation_info", [])
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
            entity_info = keywords_data.get("entity_info", [])
            relation_info = keywords_data.get("relation_info", [])
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return PROMPTS["fail_response"]

    if not entity_info and not relation_info:
        print("No entity or relation info found.")
        return PROMPTS["fail_response"]
    print("Entity Info: ", entity_info)
    print("Relation Info: ", relation_info)

    # 设置总需要的Top K数量
    top_k = 60

    # 第2步：构建内容并查询向量数据库
    # 2.1 对每个entity_info构建content并查询entities_vdb
    entity_content_list = [
        f"The entity '{entity['entity_name']}' is described as: {entity['description']}."
        for entity in entity_info
    ]
    entity_tasks = [
        entities_vdb.query(content, top_k=top_k) for content in entity_content_list
    ]
    entity_results_list = await asyncio.gather(*entity_tasks)
    entity_results = [item for sublist in entity_results_list for item in sublist]

    # 2.2 对每个relation_info构建content并查询relationships_vdb
    relation_content_list = [
        f"The relationship '{relation['keyword']}' connects source ID '{relation['src_id']}' with target ID '{relation['tgt_id']}', and is described as: {relation['description']}."
        for relation in relation_info
    ]
    relation_tasks = [
        relationships_vdb.query(content, top_k=top_k) for content in relation_content_list
    ]
    relation_results_list = await asyncio.gather(*relation_tasks)
    relation_results = [item for sublist in relation_results_list for item in sublist]

    # 2.3 聚集实体
    initial_nodes = set()
    for r in entity_results:
        initial_nodes.add(r["entity_name"])

    # 2.4 处理每个relation_info的关系
    relation_initial_nodes_groups = []  # 每个group对应一个relation_info
    all_source_nodes = set()  # 用于收集所有源节点
    for idx, relation_results in enumerate(relation_results_list):
        current_relation_info = relation_info[idx]
        current_relation_info = f"The relationship '{current_relation_info['keyword']}' connects source ID '{current_relation_info['src_id']}' with target ID '{current_relation_info['tgt_id']}', and is described as: {current_relation_info['description']}."
        # 提取relations
        current_relations = [
            (r["src_id"], r["tgt_id"], r["keyword"]) for r in relation_results
        ]
        initial_edges = current_relations

        # 将关系对应的节点加入初始节点集合
        current_initial_nodes = set(initial_nodes)  
        for (s, _, _) in initial_edges:
            current_initial_nodes.add(s)
            all_source_nodes.add(s)  # 收集所有源节点

        relation_initial_nodes_groups.append((current_relation_info, list(current_initial_nodes)))

    # 提取所有的相似度
    all_similarities = {}

    # 1) 收集任务
    tasks = []
    for r_content in relation_content_list:
        # 查询实体库
        task_entity = asyncio.create_task(entities_vdb.query(r_content, top_k=sys.maxsize))
        tasks.append((r_content, "entity", task_entity))

        # 查询关系库
        task_relation = asyncio.create_task(relationships_vdb.query(r_content, top_k=sys.maxsize))
        tasks.append((r_content, "relation", task_relation))

    # 2) 并发执行所有查询
    results = await asyncio.gather(*[task[2] for task in tasks])

    # 3) 将查询结果写入 all_similarities
    for i, (r_content, category, task_obj) in enumerate(tasks):
        query_result = results[i]  

        # 遍历查询得到的条目，写入相似度字典
        for item in query_result:
            content_key = item["content"]     
            metrics_info = item["__metrics__"] 
            all_similarities[(r_content, content_key)] = metrics_info
            
    # 将所有相似度写入文件
    os.makedirs("analyze", exist_ok=True)
    with open("analyze/similarities.txt", "w", encoding="utf-8") as f:
        for (r_content, content_key), sim in all_similarities.items():
            f.write(f"{r_content}\t{content_key}\t{sim}\n")

    # ============== 第3步：基于 BFS + DFS 的混合搜索逻辑==============
    MAX_PATH_LENGTH = 50

    async def bfs_dfs_walk_for_relation(
        relation_content: str,
        start_nodes: List[str],
        graph: "BaseGraphStorage",
        threshold: float = 0.5,
        max_path_length: int = MAX_PATH_LENGTH  # 添加最大路径长度参数
    ) -> List[Dict[str, List[str]]]:
        """
        针对某一个 relation_content，对一批初始节点进行 BFS+DFS 混合搜索。
        规则：
        1) 如果节点与 relation_content 的相似度 < threshold，则停止往下搜索。
        2) 否则，根据节点与 relation_content 的相似度决定挑选多少条最相似的边进行并行扩展。
        3) 整个过程中，不重复访问同一个节点或同一条边。
        4) 如果路径达到最大长度，则停止扩展并记录路径。
        5) 如果当前节点没有更多可扩展的边，则记录路径。
        
        返回值：所有遍历到的路径（含节点和边序列）。
        """
        visited_nodes: Set[str] = set()
        visited_edges: Set[Tuple[str, str, str]] = set()  # (src, tgt, edge_kw)
        result_paths: List[Dict[str, List[str]]] = []

        queue = deque()
        print("len(start_nodes): ", len(start_nodes))
        print("start_nodes: ", start_nodes)
        for node in start_nodes:
            node_content = f'The entity \'{node}\' is described as: {(await graph.get_node(node))["description"]}.'
            node_sim = all_similarities.get((relation_content, node_content))
            if node_sim < threshold:
                continue
            path = {
                "nodes": [node],
                "edges": []
            }
            queue.append(path)
            visited_nodes.add(node)
            
        while queue:
            current_path = queue.popleft()
            current_node = current_path["nodes"][-1]

            current_node_content = f'The entity \'{current_node}\' is described as: {(await graph.get_node(current_node))["description"]}.'
            node_sim = all_similarities.get((relation_content, current_node_content))
            
            # 检查相似度阈值
            if node_sim < threshold:
                # print("similarity below threshold")
                result_paths.append(current_path)  
                continue

            # 检查路径长度是否达到最大值
            if len(current_path["nodes"]) >= max_path_length:
                # print("max path length reached")
                result_paths.append(current_path)
                continue

            top_k_edges = decide_top_k_by_similarity(node_sim)

            edges = await graph.get_node_edges(current_node)
            edge_candidates = []
            for edge in edges:
                src, tgt, edge_kw, data = edge
                if src == tgt: 
                    continue
                edge_content = f'The relationship \'{edge_kw}\' connects source ID \'{src}\' with target ID \'{tgt}\', and is described as: {data["description"]}.'
                edge_content_reverse = f'The relationship \'{edge_kw}\' connects source ID \'{tgt}\' with target ID \'{src}\', and is described as: {data["description"]}.'
                edge_sim = all_similarities.get((relation_content, edge_content)) if (relation_content, edge_content) in all_similarities else all_similarities.get((relation_content, edge_content_reverse))
                edge_candidates.append((edge, edge_sim))

            if not edge_candidates:
                # print("no more edges to expand")
                result_paths.append(current_path)
                continue
            edge_candidates.sort(key=lambda x: x[1], reverse=True)
            chosen_edges = edge_candidates[:top_k_edges]

            for (edge, edge_sim) in chosen_edges:
                src, tgt, edge_kw, data = edge
                # 检测是否已访问
                if tgt in visited_nodes or (src, tgt, edge_kw) in visited_edges or (tgt, src, edge_kw) in visited_edges:
                    # 在检测到已访问时，将当前路径添加到结果中
                    # print("visited node or edge")
                    result_paths.append(current_path)
                    continue

                # 标记节点和边为已访问
                visited_nodes.add(tgt)
                visited_edges.add((src, tgt, edge_kw))
            
                new_path = {
                    "nodes": current_path["nodes"] + [tgt],
                    "edges": current_path["edges"] + [edge]
                }
                queue.append(new_path)  
                
        return result_paths

    # ============== 执行针对每个 group 的 BFS+DFS，收集路径 ==============
    all_paths = []
    tasks = []

    # 我们并行地对每个 relation_info 对应的 (relation_content, start_nodes) 去跑
    for (relation_content, start_nodes_name) in relation_initial_nodes_groups:
        task = bfs_dfs_walk_for_relation(
            relation_content=relation_content,
            start_nodes=start_nodes_name,
            graph=knowledge_graph_inst,
            threshold=0.1
        )
        tasks.append(task)

    # 并行执行
    all_paths_results = await asyncio.gather(*tasks)
    for paths in all_paths_results:
        all_paths.extend(paths)
        
    # 对路径进行去重
    all_paths = list({tuple(path["nodes"]): path for path in all_paths}.values())

    # 结果保存
    os.makedirs("analyze", exist_ok=True)
    save_paths_to_file(all_paths, "analyze/paths.json")

    # 第4步：从筛选出的路径中提取节点、边以及相应文本构建上下文，并进行评分
    # 收集路径中的节点和边
    selected_nodes = set()
    selected_edges = []
    for path in all_paths:
        nodes = path["nodes"]
        edges = path["edges"]
        for i in range(len(edges)):
            src_node = nodes[i]
            edge_info = edges[i]  # 边信息现在直接是一个元组(src, tgt, edge_kw, data)
            tgt_node = nodes[i + 1]
            selected_nodes.add(src_node)
            selected_nodes.add(tgt_node)
            selected_edges.append((src_node, tgt_node, edge_info))
        # 添加路径最后一个节点
        if nodes:
            selected_nodes.add(nodes[-1])

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
    def score_path(path: Dict[str, List[str]]) -> float:
        edges_info = path["edges"]
        # print("edges_info: ", edges_info)
        total_edge_weight = sum(e[3].get("weight", 1.0) for e in edges_info)
        total_kw_match = sum(edges_scores.get((path["nodes"][i], path["nodes"][i + 1], e[3].get("keyword")), 0.0) for i, e in enumerate(edges_info[:-1]))
        node_degree_sum = sum(node_info[n]["degree"] for n in path["nodes"] if n in node_info)
        path_length = len(path["nodes"])
        alpha, beta, gamma, delta = 1.0, 2.0, 1.0, 0.5
        score = alpha * total_edge_weight + beta * total_kw_match + gamma * node_degree_sum + delta * path_length
        return score

    scored_top_k = 1000
    scored_paths = [(p, score_path(p)) for p in all_paths]
    scored_paths.sort(key=lambda x: x[1], reverse=True)
    top_paths = [p for p, s in scored_paths[:scored_top_k]] if scored_paths else []

    if not top_paths:
        print("No top paths found.")
        return PROMPTS["fail_response"]

    selected_nodes = set()
    selected_edges = []
    for path in top_paths:
        nodes = path["nodes"]
        edges = path["edges"]
        for i in range(len(edges)):
            src_node = nodes[i]
            edge_info = edges[i]
            edge_keyword = edge_info[2]
            tgt_node = nodes[i + 1]
            selected_nodes.add(src_node)
            selected_nodes.add(tgt_node)
            selected_edges.append((src_node, tgt_node, edge_keyword, edge_info[-1]))
        if nodes:
            selected_nodes.add(nodes[-1])

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
        for (_, _, _, e_info) in selected_edges:
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
    for i, (s, t, kw, e_info) in enumerate(selected_edges):
        # 计算edge的rank(即edge_degree)
        deg = await knowledge_graph_inst.edge_degree(s, t)
        edges_context_data.append({
            "src_tgt": (s, t),
            "keyword": kw,
            "description": e_info.get("description", "UNKNOWN"),
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
        print("No context found.")
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
        serializable_path = {
            'nodes': path["nodes"],
            'edges': []
        }
        
        for i in range(len(path["nodes"]) - 1):
            src_node = path["nodes"][i]
            tgt_node = path["nodes"][i + 1]
            edge_info = path["edges"][i]  # edge_info 是 (src, tgt, edge_kw, data)

            edge_keyword = edge_info[2]
            edge_data = edge_info[-1]

            serializable_path['edges'].append({
                'source': src_node,
                'target': tgt_node,
                'relation_keyword': edge_keyword,
                'edge_data': edge_data
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
        print("No naive results found.")
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

import asyncio
import json
import math
import random
import re
from typing import Any, Dict, List, Tuple, Union
from collections import Counter, defaultdict
import warnings

from numpy import dot
from torch import norm
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
    ),

async def my_query(
    query: str,
    knowledge_graph_inst: "BaseGraphStorage",
    entities_vdb: "BaseVectorStorage",
    relationships_vdb: "BaseVectorStorage",
    text_chunks_db: "BaseKVStorage[TextChunkSchema]",
    query_param: "QueryParam",
    global_config: dict
) -> str:
    use_model_func = global_config["llm_model_func"]

    # 第1步：关键词提取
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        relation_keywords = keywords_data.get("relation_keywords", [])
        entity_keywords = keywords_data.get("entity_keywords", [])
        relation_keywords = ", ".join(relation_keywords)
        entity_keywords = ", ".join(entity_keywords)
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
            relation_keywords = keywords_data.get("relation_keywords", [])
            entity_keywords = keywords_data.get("entity_keywords", [])
            relation_keywords = ", ".join(relation_keywords)
            entity_keywords = ", ".join(entity_keywords)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]

    if not entity_keywords and not relation_keywords:
        return PROMPTS["fail_response"]

    # 第2步：利用关键词从entities_vdb和relationships_vdb中检索初始实体和关系集合
    # 通过entity关键词检索实体
    entity_results = await entities_vdb.query(entity_keywords, top_k=query_param.top_k) if entity_keywords else []
    relation_results = await relationships_vdb.query(relation_keywords, top_k=query_param.top_k) if relation_keywords else []

    # 将所有涉及的节点和关系提取为初始节点集和初始边集
    initial_nodes = set()
    for r in entity_results:
        initial_nodes.add(r["entity_name"])

    initial_edges = []
    for r in relation_results:
        initial_edges.append((r["src_id"], r["tgt_id"], r["keyword"]))

    # 将关系对应的节点加入初始节点集合
    for (s, _, _) in initial_edges:
        initial_nodes.add(s)

    initial_nodes = list(initial_nodes)

    # 第3步：基于多重图的加权随机游走（Weighted RWR）
    # 实现Weighted RWR的核心函数
    async def keyword_similarity(edge_keyword: str, query_keywords: str, embedding_func) -> float:
        # 使用embedding_func对edge_keyword和query_keywords进行嵌入
        embeddings = await embedding_func([edge_keyword, query_keywords])
        v1, v2 = embeddings[0], embeddings[1]
        sim = dot(v1, v2) / (norm(v1)*norm(v2))
        return sim

    async def perform_weighted_rwr(
        start_nodes: List[str],
        graph: "BaseGraphStorage",
        entity_kw: str,
        relation_kw: str,
        walk_steps: int = 100,
        restart_prob: float = 0.15,
        paths_to_collect: int = 500
    ) -> List[List[Tuple[str, Dict[str, Any]]]]:
        # 返回paths结构：List[Path], Path是List[ (node, edge_info) ]
        paths = []
        for start_node in start_nodes:
            for _ in range(paths_to_collect // max(1, len(start_nodes))):
                path = []
                current_node = start_node
                for _ in range(walk_steps):
                    edges = await graph.get_node_edges(current_node)
                    if not edges:
                        break
                    # edges为[(src, tgt, keyword, data), ...]
                    # 计算转移概率
                    weights = []
                    total_weight = 0.0
                    for (s, t, edge_keyword, data) in edges:
                        w = data.get("weight", 1.0)
                        # 关键词匹配度
                        k_sim = await keyword_similarity(edge_keyword, entity_kw + " " + relation_kw, embedding_func)
                        # 最终概率 = w * k_sim
                        p = w * k_sim
                        weights.append((t, data, p))
                        total_weight += p
                    if total_weight == 0:
                        break
                    # 归一化并随机选择下一步
                    rand_val = random.random()
                    cumulative = 0.0
                    chosen_node = None
                    chosen_edge = None
                    for (tnode, e_info, p) in weights:
                        cp = p / total_weight
                        cumulative += cp
                        if rand_val <= cumulative:
                            chosen_node = tnode
                            chosen_edge = e_info
                            break
                    if chosen_node is None:
                        break
                    path.append((current_node, chosen_edge))
                    current_node = chosen_node
                    # 重启机制
                    if random.random() < restart_prob:
                        current_node = start_node
                # 将终点节点也加入信息
                if path:
                    path.append((current_node, None))
                    paths.append(path)
        return paths

    paths = await perform_weighted_rwr(
        start_nodes=initial_nodes,
        graph=knowledge_graph_inst,
        entity_kw=entity_keywords,
        relation_kw=relation_keywords,
        walk_steps=50,           # 可调优
        restart_prob=0.15,       # 可调优
        paths_to_collect=500      # 可调优
    )

    # 第4步：对路径进行评分和筛选
    def score_path(path: List[Tuple[str, Dict[str, Any]]]) -> float:
        # path: [(node, edge_info), ...]
        # 去除最后None的边信息
        edges_info = [e for (_, e) in path if e is not None]
        # 评分考虑: sum(edge_weight) + sum(keyword_sim) + sum(node_degrees) - path_length
        total_edge_weight = 0.0
        total_kw_match = 0.0
        node_degree_sum = 0.0
        for e in edges_info:
            w = e.get("weight", 1.0)
            total_edge_weight += w
            # 简化关键词匹配得分
            total_kw_match += 1.0 if e["keyword"] in (entity_keywords + " " + relation_keywords) else 0.5
            # 假设节点度数为1.0，实际应根据节点数据获取
            node_degree_sum += 1.0
        path_length = len(path)
        # 参数设定可调
        alpha, beta, gamma, delta = 1.0, 1.0, 0.5, 0.5
        score = alpha * total_edge_weight + beta * total_kw_match + gamma * node_degree_sum - delta * path_length
        return score

    scored_paths = [(p, score_path(p)) for p in paths]
    scored_paths.sort(key=lambda x: x[1], reverse=True)
    top_paths = [p for p, s in scored_paths[:query_param.top_k]] if scored_paths else []

    if not top_paths:
        return PROMPTS["fail_response"]

    # 第5步：从筛选出的路径中提取节点、边以及相应文本构建上下文
    # 收集路径中的节点和边
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
        # 添加路径最后一个节点
        if path:
            selected_nodes.add(path[-1][0])

    selected_nodes = list(selected_nodes)

    # 获取实体信息
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(n) for n in selected_nodes]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(n) for n in selected_nodes]
    )
    node_datas = [
        {**data, "entity_name": name, "rank": deg} 
        for name, data, deg in zip(selected_nodes, node_datas, node_degrees) 
        if data is not None
    ]

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

    return response

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

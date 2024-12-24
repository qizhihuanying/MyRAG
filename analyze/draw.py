from pyvis.network import Network
import json

def visualize_paths_with_pyvis(paths_file: str, output_html: str = "./analyze/graph.html"):
    with open(paths_file, 'r', encoding='utf-8') as f:
        all_paths = json.load(f)

    # 初始化 Network 时设置 cdn_resources 为 'remote' 以避免警告
    net = Network(height="1200px", width="100%", directed=True, notebook=True, cdn_resources='remote')

    # 计算节点的度，用于动态调整节点大小
    degree_dict = {}
    for path in all_paths:
        for edge_info in path['edges']:
            src = edge_info['source']
            tgt = edge_info['target']
            degree_dict[src] = degree_dict.get(src, 0) + 1
            degree_dict[tgt] = degree_dict.get(tgt, 0) + 1

    # 定义关系类型和颜色映射（根据实际情况调整）
    color_map = {
        "relation_type_1": "red",
        "relation_type_2": "blue",
        # 根据实际关系类型添加更多映射
    }

    # 用于记录已添加的节点和边，避免重复添加
    added_nodes = set()
    added_edges = set()

    for path in all_paths:
        # 首先添加所有节点
        for node in path['nodes']:
            if node not in added_nodes:
                size = 10 + degree_dict.get(node, 0) * 2
                net.add_node(
                    node,
                    label=node,
                    size=size,
                    font={'size': 14, 'face': 'Arial', 'color': 'black'},
                    shape='dot',
                    color={'background': 'white'}
                )
                added_nodes.add(node)

        # 然后添加所有边
        for edge_info in path.get('edges', []):
            src = edge_info['source']
            tgt = edge_info['target']
            rel = edge_info['relation_keyword']
            desc = edge_info['edge_data'].get('description', '')

            # 根据关系类型选择颜色
            color = color_map.get(rel, "gray")

            # 创建一个唯一的边标识
            edge_key = (src, tgt, rel)

            if edge_key not in added_edges:
                # 添加边并记录
                net.add_edge(
                    src,
                    tgt,
                    label=rel,
                    title=f"Relation: {rel}\nDesc: {desc}",
                    font={'size': 12},
                    color=color
                )
                added_edges.add(edge_key)

    # 设置选项，确保传递的是有效的 JSON
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 14,
          "face": "Arial",
          "color": "black",
          "strokeWidth": 2,
          "strokeColor": "#ffffff",
          "background": "white",
          "align": "center"
        },
        "shape": "dot",
        "size": 20,
        "borderWidth": 2,
        "borderWidthSelected": 4
      },
      "edges": {
        "font": {
          "size": 12,
          "align": "middle"
        },
        "color": {
          "color": "gray",
          "highlight": "red",
          "inherit": false
        },
        "scaling": {
          "label": {
            "enabled": true
          }
        },
        "smooth": {
          "type": "dynamic"
        }
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75
      },
      "layout": {
        "improvedLayout": true
      }
    }
    """)

    # 生成 HTML 文件
    net.show(output_html)

if __name__ == '__main__':
    visualize_paths_with_pyvis("./analyze/paths.json", "./analyze/graph.html")

import json
import matplotlib.pyplot as plt

def load_paths_from_file(filename):
    """
    从JSON文件中加载路径。
    
    :param filename: str, 文件名
    :return: List[List[Dict[str, Any]]], 每条路径是一个由节点和关系组成的字典列表
    """
    with open(filename, 'r', encoding='utf-8') as f:
        paths = json.load(f)
    return paths

def draw_paths_in_lines(paths, filename='paths_visualization.png'):
    """
    将每条路径画成一条单独的水平直线：
    - 节点以圆圈表示，其中显示entity名称
    - 节点之间以线连接，并在中点处标注relation_keyword或其他描述信息
    """
    plt.figure(figsize=(36, 24))
    
    # 路径的数量
    num_paths = len(paths)
    
    for p_index, path in enumerate(paths):
        # y坐标随路径序号改变，让第0条路径在y=0，每向下移动一行
        # 这里采用 y = -p_index，确保多条路径上下分开
        y = -p_index
        
        # 画节点和边
        for i, node in enumerate(path['nodes']):  # 假设每条路径是一个字典，包含 "nodes" 和 "edges"
            entity = node["entity"].strip('"')  # 去掉双引号
            rel_keyword = node["relation_keyword"]
            
            # 画节点（圆圈）
            plt.scatter(i, y, s=1200, facecolors='white', edgecolors='black', zorder=3)
            # 在节点中心写上entity名
            plt.text(i, y, entity, ha='center', va='center', fontsize=8, wrap=True)
            
            # 如果不是最后一个节点，且下一个节点关系存在，则画边
            if i < len(path['nodes']) - 1:
                next_rel = path['edges'][i]["relation_keyword"]  # 假设路径字典中包含边的关键信息
                if next_rel is not None:
                    # 在(i, y) 和 (i+1, y)之间画线
                    plt.plot([i, i+1], [y, y], 'k-', linewidth=2, zorder=2)
                    # 在边的中点上方标注关系关键词
                    mid_x = (i + (i+1)) / 2
                    plt.text(mid_x, y+0.1, next_rel, ha='center', va='bottom', fontsize=8, color='red')

    # 调整绘图区域
    max_len = max(len(path['nodes']) for path in paths)
    plt.xlim(-0.5, max_len - 0.5)
    # y范围根据路径数目适当调整
    plt.ylim(-num_paths + 0.5, 0.5)
    
    # 去除坐标轴
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图形并显示
    plt.savefig(filename, dpi=300)
    plt.show()

# 使用修改后的函数
paths = load_paths_from_file('analyze/paths.json')
draw_paths_in_lines(paths, 'analyze/paths_visualization.png')

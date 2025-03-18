import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import pandas as pd
from py2neo import Graph

# 初始化 Neo4j 图数据库连接
def initialize_neo4j(uri, auth):
    try:
        graph = Graph(uri, auth=auth)
        print("Connected to Neo4j database.")
        return graph
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

# 从 Neo4j 中加载评论网络
def load_comment_network(graph):
    query = """
    MATCH (c1:Comment)-[:MENTIONS|RELATED_TO]->(e:Entity)<-[:MENTIONS|RELATED_TO]-(c2:Comment)
    WHERE c1 <> c2
    RETURN c1.text AS comment1, c2.text AS comment2
    """
    result = graph.run(query).data()
    return result

# 构建 NetworkX 图
def build_network(data):
    G = nx.Graph()
    for record in data:
        comment1 = record['comment1']
        comment2 = record['comment2']
        G.add_edge(comment1, comment2)
    return G

# 应用 Louvain 社区发现算法
def detect_communities(G):
    partition = community_louvain.best_partition(G)
    return partition

# 可视化社区
def visualize_communities(G, partition):
    pos = nx.spring_layout(G)
    cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title("Community Detection in Comment Network")
    plt.show()

# 主函数
def main():
    # 配置 Neo4j 连接
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "cqcen123"  # 替换为你的密码

    # 初始化 Neo4j 图数据库
    graph = initialize_neo4j(NEO4J_URI, (NEO4J_USER, NEO4J_PASSWORD))
    if not graph:
        print("Failed to connect to Neo4j. Please check your configuration.")
        return

    # 加载评论网络
    print("Loading comment network from Neo4j...")
    comment_network_data = load_comment_network(graph)
    if not comment_network_data:
        print("No data found in Neo4j. Please check your database.")
        return

    # 构建 NetworkX 图
    print("Building NetworkX graph...")
    G = build_network(comment_network_data)

    # 应用 Louvain 社区发现算法
    print("Detecting communities using Louvain algorithm...")
    partition = detect_communities(G)

    # 可视化社区
    print("Visualizing communities...")
    visualize_communities(G, partition)

    # 保存社区结果到文件
    community_df = pd.DataFrame(list(partition.items()), columns=['Comment', 'Community'])
    community_df.to_csv("community_results.csv", index=False)
    print("Community detection results saved to 'community_results.csv'.")

if __name__ == "__main__":
    main()
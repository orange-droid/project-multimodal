import json
import tqdm
import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher
import argparse

def compute_graph_metrics(graph):
    """计算并添加网络指标到图谱"""
    print("计算图谱指标...")
    
    # 使用Cypher查询计算度中心性
    degree_query = """
    MATCH (n)
    WITH n, COUNT { (n)--() } as degree
    SET n.degree_centrality = toFloat(degree)
    """
    graph.run(degree_query)
    
    # 计算聚类系数
    clustering_query = """
    MATCH (n)-->(m)
    WITH n, collect(m) as neighbors
    MATCH (x)--(y)
    WHERE x IN neighbors AND y IN neighbors AND x <> y
    WITH n, neighbors, count(DISTINCT [x,y]) as triangles
    WITH n, neighbors, triangles
    WHERE size(neighbors) > 1
    SET n.clustering_coefficient = toFloat(2 * triangles) / (size(neighbors) * (size(neighbors) - 1))
    """
    graph.run(clustering_query)

def detect_bullying_communities(graph):
    """检测具有高霸凌倾向的社区"""
    print("检测霸凌社区...")
    
    try:
        # 计算每个社区的霸凌分数
        community_score_query = """
        MATCH (v:Video)-[:HAS_COMMENT]->(c:Comment)
        WITH v, avg(c.bullying_score) as avg_score
        WHERE avg_score > 0.2
        RETURN v.id as video_id, avg_score
        ORDER BY avg_score DESC
        """
        
        results = graph.run(community_score_query)
        community_scores = {row['video_id']: row['avg_score'] for row in results}
        
        print(f"发现 {len(community_scores)} 个高霸凌倾向社区")
        return community_scores
        
    except Exception as e:
        print(f"社区检测失败: {str(e)}")
        return {}

def extract_media_sessions(graph):
    """
    从知识图谱中提取所有媒体会话
    每个会话包含：视频信息、所有评论、用户信息和相关属性
    """
    print("提取媒体会话...")
    
    # 查询所有视频及其相关信息
    video_query = """
    MATCH (v:Video)
    OPTIONAL MATCH (v)-[:IS_TYPE]->(vt:BullyingType)
    RETURN v.id as video_id,
           v.avg_bullying_score as avg_bullying_score,
           vt.label as bullying_type
    """
    
    media_sessions = []
    videos = list(graph.run(video_query))
    
    for video in tqdm.tqdm(videos, desc="处理媒体会话"):
        video_id = video['video_id']
        
        # 查询视频相关的所有评论和用户信息
        session_query = """
        MATCH (v:Video {id: $video_id})
        OPTIONAL MATCH (v)<-[:COMMENTS_ON]-(c:Comment)
        OPTIONAL MATCH (c)<-[:WROTE]-(u:User)
        OPTIONAL MATCH (c)-[:IS_TYPE]->(ct:BullyingType)
        RETURN c.id as comment_id,
               c.text as comment_text,
               c.bullying_score as bullying_score,
               c.bullying_category as bullying_category,
               c.created as created_time,
               u.id as user_id,
               u.label as username,
               u.verified as verified,
               u.location as location,
               ct.label as comment_type
        """
        
        comments = list(graph.run(session_query, video_id=video_id))
        
        # 构建媒体会话对象
        session = {
            'video': {
                'id': video_id,
                'avg_bullying_score': video['avg_bullying_score'],
                'bullying_type': video['bullying_type']
            },
            'comments': []
        }
        
        # 添加评论信息
        for comment in comments:
            comment_obj = {
                'id': comment['comment_id'],
                'text': comment['comment_text'],
                'bullying_score': comment['bullying_score'],
                'bullying_category': comment['bullying_category'],
                'created_time': comment['created_time'],
                'type': comment['comment_type'],
                'user': {
                    'id': comment['user_id'],
                    'username': comment['username'],
                    'verified': comment['verified'],
                    'location': comment['location']
                }
            }
            session['comments'].append(comment_obj)
        
        media_sessions.append(session)
    
    return media_sessions

def save_media_sessions(media_sessions, output_file):
    """保存媒体会话到JSON文件"""
    print(f"保存媒体会话到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(media_sessions, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='分析知识图谱并提取媒体会话')
    parser.add_argument('--output', default='media_sessions.json', help='输出文件路径')
    parser.add_argument('--compute-metrics', action='store_true', help='是否计算图谱指标')
    parser.add_argument('--detect-communities', action='store_true', help='是否检测霸凌社区')
    args = parser.parse_args()
    
    # 连接到Neo4j数据库
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "cqcen123"))
    
    # 计算图谱指标
    if args.compute_metrics:
        compute_graph_metrics(graph)
    
    # 检测霸凌社区
    if args.detect_communities:
        community_scores = detect_bullying_communities(graph)
        print("\n高霸凌倾向社区:")
        for video_id, score in sorted(community_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"视频 {video_id}: 霸凌分数 {score:.2f}")
    
    # 提取媒体会话
    media_sessions = extract_media_sessions(graph)
    
    # 保存媒体会话
    save_media_sessions(media_sessions, args.output)
    
    print(f"\n处理完成！共提取 {len(media_sessions)} 个媒体会话")
    print(f"结果已保存到: {args.output}")

if __name__ == "__main__":
    main()

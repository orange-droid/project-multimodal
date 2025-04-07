import json
import argparse
import tqdm
from collections import defaultdict
import pandas as pd
import random
from py2neo import Graph, Node, Relationship, NodeMatcher

def load_comments(input_file):
    """Load enriched comments from JSON file"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            try:
                # Try parsing as JSON array
                comments = json.loads(content)
                if not isinstance(comments, list):
                    comments = [comments]  # Convert single object to list
            except json.JSONDecodeError:
                # Try parsing as JSON Lines
                comments = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
        
        return comments
    except Exception as e:
        print(f"Error loading comments: {str(e)}")
        return []

def create_knowledge_graph(comments_dict, labels_df, graph):
    """
    创建用于网络霸凌检测的知识图谱
    
    图谱包含：
    - 用户节点（包含验证状态等属性）
    - 评论节点（包含霸凌分数等属性）
    - 视频节点（被评论的媒体内容）
    - 霸凌类型节点（bullying、non_bullying）
    - 节点间的关系
    """
    print("构建知识图谱...")
    
    # 清空现有图谱
    graph.delete_all()
    
    # 创建霸凌类型节点
    type_bullying = Node("BullyingType",
                        id='type_bullying',
                        label='Bullying',
                        type='bullying_type',
                        description='Confirmed bullying content')
    
    type_non_bullying = Node("BullyingType",
                            id='type_non_bullying',
                            label='Non-Bullying',
                            type='bullying_type',
                            description='Normal social interaction')
    
    # 创建霸凌类型节点
    graph.create(type_bullying)
    graph.create(type_non_bullying)
    
    # 跟踪已创建的节点
    user_nodes = set()
    
    # 创建节点匹配器
    matcher = NodeMatcher(graph)
    
    # 处理所有视频和评论
    for video_id, video_comments in tqdm.tqdm(comments_dict.items(), desc="处理视频和评论"):
        # 计算视频的总体霸凌倾向
        bullying_scores = [c.get('bullyingScore', 0) for c in video_comments]
        avg_bullying_score = sum(bullying_scores) / len(bullying_scores) if bullying_scores else 0
        
        # 获取视频的原始标签
        video_label = labels_df[labels_df['postid'] == video_id]['question2'].iloc[0] if not labels_df[labels_df['postid'] == video_id].empty else 'non_bullying'
        
        # 创建视频节点
        video_node = Node("Video",
                        id=f'video_{video_id}',
                        label=f'Video-{video_id}',
                        type='video',
                        avg_bullying_score=avg_bullying_score,
                        original_label=video_label)
        graph.create(video_node)
        
        # 根据原始标签连接到相应的类型节点
        if video_label == 'bullying':
            graph.create(Relationship(video_node, "IS_TYPE", type_bullying))
        else:
            graph.create(Relationship(video_node, "IS_TYPE", type_non_bullying))
        
        # 处理该视频的所有评论
        for comment in video_comments:
            comment_id = comment.get('commentId')
            if not comment_id:
                continue
            
            user_id = comment.get('userId')
            username = comment.get('username', f'User-{user_id}')
            comment_text = comment.get('commentText', '')
            bullying_score = comment.get('bullyingScore', 0.0)
            bullying_category = comment.get('bullyingCategory', 'non_bullying')
            
            # 创建用户节点
            if user_id and user_id not in user_nodes:
                user_node = Node("User",
                               id=f'user_{user_id}',
                               label=username,
                               type='user',
                               verified=comment.get('verified') == '1',
                               location=comment.get('location', 'Unknown'))
                graph.create(user_node)
                user_nodes.add(user_id)
            
            # 创建评论节点
            comment_node = Node("Comment",
                              id=f'comment_{comment_id}',
                              label=comment_text[:30] + '...' if len(comment_text) > 30 else comment_text,
                              type='comment',
                              text=comment_text,
                              bullying_score=bullying_score,
                              bullying_category=bullying_category,
                              created=comment.get('created'))
            graph.create(comment_node)
            
            # 创建关系
            if user_id:
                user_node = matcher.match("User", id=f'user_{user_id}').first()
                graph.create(Relationship(user_node, "WROTE", comment_node))
            
            graph.create(Relationship(comment_node, "COMMENTS_ON", video_node))
            
            # 根据霸凌分类连接到相应的类型节点（将possible_bullying归类为non_bullying）
            if bullying_category == 'bullying':
                graph.create(Relationship(comment_node, "IS_TYPE", type_bullying))
            else:
                graph.create(Relationship(comment_node, "IS_TYPE", type_non_bullying))
    
    return graph

def filter_posts_for_simplified_graph(labels_df, comments_dict):
    """
    为简易版知识图谱筛选帖子
    - 保留所有霸凌视频
    - 随机选择相同数量的非霸凌视频
    """
    # 获取霸凌视频
    bullying_posts = labels_df[labels_df['question2'] == 'bullying']['postid'].unique()
    print(f"找到 {len(bullying_posts)} 个霸凌视频")
    
    # 获取非霸凌视频
    non_bullying_posts = labels_df[labels_df['question2'] != 'bullying']['postid'].unique()
    
    # 随机选择相同数量的非霸凌视频
    selected_non_bullying = random.sample(list(non_bullying_posts), min(len(bullying_posts), len(non_bullying_posts)))
    print(f"随机选择 {len(selected_non_bullying)} 个非霸凌视频")
    
    # 合并选中的帖子ID
    selected_posts = list(bullying_posts) + selected_non_bullying
    
    # 筛选评论
    filtered_comments = {}
    for post_id in selected_posts:
        if post_id in comments_dict:
            filtered_comments[post_id] = comments_dict[post_id]
    
    return selected_posts, filtered_comments

def main():
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='为网络霸凌检测生成知识图谱')
    parser.add_argument('--simplified', action='store_true', help='是否生成简易版知识图谱（只包含部分视频和评论）')
    args = parser.parse_args()
    
    # 文件路径
    labels_path = "data/vine labeled data/vine_labeled_cyberbullying_data.csv"
    url_to_postid_path = "data/Vine_videos/urls_to_postids.txt"
    comments_path = "data/enriched_comments.json"
    
    # 加载 URL 到 PostID 的映射
    url_to_postid = {}
    with open(url_to_postid_path, "r") as f:
        for line in f:
            postid, url = line.strip().split(",")
            url_to_postid[url] = postid

    # 加载标签数据
    labels_df = pd.read_csv(labels_path)
    labels_df['postid'] = labels_df['videolink'].apply(lambda x: url_to_postid.get(x, None))
    labels_df = labels_df.dropna(subset=['postid'])  # 删除无法映射到postid的行
    labels_df['postid'] = labels_df['postid'].astype(str)
    
    # 加载评论数据
    comments = load_comments(comments_path)
    if not comments:
        print("未找到评论或加载评论时出错")
        return
    
    # 将评论按帖子ID组织
    comments_dict = {}
    for comment in comments:
        post_id = comment.get('postId', '')
        if post_id:
            if post_id not in comments_dict:
                comments_dict[post_id] = []
            comments_dict[post_id].append(comment)
    
    # 如果选择简易版，筛选帖子和评论
    if args.simplified:
        print("\n生成简易版知识图谱...")
        postids, comments_dict = filter_posts_for_simplified_graph(labels_df, comments_dict)
        print(f"简易版图谱将包含 {len(postids)} 个视频")
        total_comments = sum(len(comments) for comments in comments_dict.values())
        print(f"包含 {total_comments} 条评论")
    else:
        postids = labels_df["postid"].unique()
        print(f"\n生成完整知识图谱...")
        print(f"包含 {len(postids)} 个视频")
        total_comments = sum(len(comments) for comments in comments_dict.values())
        print(f"包含 {total_comments} 条评论")
    
    # 连接到Neo4j数据库
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "cqcen123"))
    
    # 创建知识图谱
    graph = create_knowledge_graph(comments_dict, labels_df, graph)
    
    print("\n知识图谱生成完成")

if __name__ == "__main__":
    main()
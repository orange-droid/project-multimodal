import os
import json
import pandas as pd
import spacy
import numpy as np
from py2neo import Graph, Node, Relationship

# 初始化 SpaCy NER 模型
nlp = spacy.load("en_core_web_sm")

# 初始化 Neo4j 图数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "cqcen123"))

def build_knowledge_graph(postids, comments_dict, labels_df):
    graph.delete_all()
    
    # 创建霸凌相关的实体节点
    aggression_node = Node("CyberbullyingType", name="Aggression")
    none_aggression_node = Node("CyberbullyingType", name="noneAggression")
    bullying_node = Node("CyberbullyingType", name="Bullying")
    none_bullying_node = Node("CyberbullyingType", name="noneBullying")
    
    graph.create(aggression_node)
    graph.create(none_aggression_node)
    graph.create(bullying_node)
    graph.create(none_bullying_node)

    # 创建视频节点
    for postid in postids:
        label_info = labels_df[labels_df['postid'] == postid]
        if not label_info.empty:
            question1 = label_info['question1'].values[0]
            question2 = label_info['question2'].values[0]
        else:
            question1 = None
            question2 = None
        
        video_node = Node("Video", id=postid, question1=question1, question2=question2)
        graph.create(video_node)
        
        # 建立视频与霸凌类型的关系
        if question1 == "aggression":
            graph.create(Relationship(video_node, "HAS_TYPE", aggression_node))
        else:
            graph.create(Relationship(video_node, "HAS_TYPE", none_aggression_node))
        
        if question2 == "bullying":
            graph.create(Relationship(video_node, "HAS_TYPE", bullying_node))
        else:
            graph.create(Relationship(video_node, "HAS_TYPE", none_bullying_node))
    
    # 创建评论节点并提取实体
    for postid, comments in comments_dict.items():
        for comment in comments:
            doc = nlp(comment)
            comment_node = Node("Comment", text=comment)
            graph.create(comment_node)
            
            # 判断评论是否与霸凌相关
            if "aggression" in comment.lower() or "bullying" in comment.lower():
                graph.create(Relationship(comment_node, "RELATED_TO", aggression_node))
                graph.create(Relationship(comment_node, "RELATED_TO", bullying_node))
            else:
                graph.create(Relationship(comment_node, "RELATED_TO", none_aggression_node))
                graph.create(Relationship(comment_node, "RELATED_TO", none_bullying_node))
            
            for ent in doc.ents:
                entity_node = Node("Entity", name=ent.text, type=ent.label_)
                graph.merge(entity_node, "Entity", "name")
                graph.create(Relationship(comment_node, "MENTIONS", entity_node))
            
            # 创建评论与视频的关系
            video_node = graph.nodes.match("Video", id=postid).first()
            if video_node:
                graph.create(Relationship(video_node, "HAS_COMMENT", comment_node))

def kg_embedding(graph, postids):
    nodes = list(graph.nodes.match())
    embeddings = {}
    for node in nodes:
        embeddings[node["id"] or node["text"] or node["name"]] = np.random.rand(128)
    return embeddings

if __name__ == "__main__":
    # 文件路径
    labels_path = "data/vine labeled data/vine_labeled_cyberbullying_data.csv"
    url_to_postid_path = "data/Vine_videos/urls_to_postids.txt"
    comments_path = "data/sampled_post-comments_vine.json"
    
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
    postids = labels_df["postid"].unique()

    # 加载评论数据
    comments_dict = {}
    with open(comments_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                postid = data.get('postId', '')
                if postid not in comments_dict:
                    comments_dict[postid] = []
                comments_dict[postid].append(data.get('commentText', ''))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    
    # 构建知识图谱
    build_knowledge_graph(postids, comments_dict, labels_df)
    
    # 提取知识图谱嵌入
    kg_embeddings = kg_embedding(graph, postids)
    np.save("kg_embeddings.npy", kg_embeddings)
    print(f"Knowledge Graph embeddings saved to kg_embeddings.npy")
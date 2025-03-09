import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 定义加载视频特征的函数
def load_video_features(video_features_folder, postid):
    video_feature_path = os.path.join(video_features_folder, f"{postid}.npy")
    if os.path.exists(video_feature_path):
        try:
            video_feature = np.load(video_feature_path)
            if video_feature.ndim == 2 and video_feature.shape[1] == 512:  # 检查特征维度是否正确
                return video_feature
            else:
                print(f"Invalid video feature file for postid: {postid}, shape: {video_feature.shape}")
        except Exception as e:
            print(f"Error loading video feature file for postid {postid}: {e}")
    else:
        print(f"Video feature file not found for postid: {postid}")
    return None

# 特征融合
def feature_engineering(video_features, text_features):
    # 对每个视频的帧特征取平均值，生成固定长度的特征表示
    video_features_avg = np.array([np.mean(features, axis=0) for features in video_features])
    
    # 特征拼接
    fused_features = np.concatenate((video_features_avg, text_features), axis=1)
    
    # 标准化
    scaler = StandardScaler()
    fused_features_scaled = scaler.fit_transform(fused_features)
    
    # PCA降维
    pca = PCA(n_components=512)
    fused_features_pca = pca.fit_transform(fused_features_scaled)
    
    return fused_features_pca

# 主函数
if __name__ == "__main__":
    # 文件路径
    video_features_folder = "data/Vine_videos/frame_features"  # 视频特征文件夹
    text_features_path = "data/vine_bert_features.npy"  # 评论特征文件
    text_file = "data/Vine_videos/urls_to_postids.txt"  # 包含视频URL和PostID的文本文件
    output_features_path = "data/fused_features.npy"  # 保存融合特征的路径
    failed_videos_file = "data/failed_videos.txt"  # 记录失败的视频ID
    labels_file = "data/vine labeled data/vine_labeled_cyberbullying_data.csv"  # 标签文件路径

    # 读取文本文件中的视频URL和PostID映射
    url_to_postid = {}
    postid_to_index = {}
    with open(text_file, mode='r') as file:
        for index, line in enumerate(file):
            postid, url = line.strip().split(',')
            url_to_postid[url] = postid
            postid_to_index[postid] = index

    # 加载评论特征
    text_features = np.load(text_features_path)

    # 加载标签数据
    labels_df = pd.read_csv(labels_file)
    labels_df['postid'] = labels_df['videolink'].apply(lambda x: url_to_postid.get(x, None))
    labels_df = labels_df.dropna(subset=['postid'])  # 删除无法映射到postid的行
    labels_df['postid'] = labels_df['postid'].astype(str)

    # 加载视频特征并跳过空文件或损坏的文件
    video_features_list = []
    valid_postids = []
    failed_postids = []
    for postid in url_to_postid.values():
        video_feature = load_video_features(video_features_folder, postid)
        if video_feature is not None:
            video_features_list.append(video_feature)
            valid_postids.append(postid)
        else:
            failed_postids.append(postid)

    # 保存失败的视频ID
    if failed_postids:
        with open(failed_videos_file, 'w') as f:
            for postid in failed_postids:
                f.write(f"{postid}\n")
        print(f"Failed video IDs saved to {failed_videos_file}")

    if not video_features_list:
        print("No valid video features found. Exiting.")
        exit()

    # 确保特征和标签的顺序一致
    valid_indices = [postid_to_index[postid] for postid in valid_postids]
    text_features = text_features[valid_indices]

    # 特征融合
    fused_features = feature_engineering(video_features_list, text_features)

    # 保存融合特征
    np.save(output_features_path, fused_features)
    print(f"Fused features saved to {output_features_path}")
    print(f"Fused features shape: {fused_features.shape}")

    # 保存与融合特征对应的标签
    aligned_labels_df = labels_df[labels_df['postid'].isin(valid_postids)]
    aligned_labels_df.to_csv("data/aligned_labels.csv", index=False)
    print("Aligned labels saved to data/aligned_labels.csv")
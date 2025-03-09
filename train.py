import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 加载融合特征和标签
def load_data(features_path, labels_path):
    features = np.load(features_path)  # 加载融合特征
    labels_df = pd.read_csv(labels_path)  # 加载标签文件

    # 提取标签列
    labels = labels_df[['question1', 'question2']].values
    confidence_scores = labels_df[['question1:confidence', 'question2:confidence']].values

    # 检查特征和标签数量是否一致
    if len(features) != len(labels):
        print(f"Feature and label counts do not match: {len(features)} features, {len(labels)} labels")
        # 如果不一致，只保留与特征对应的标签
        labels_df = labels_df.iloc[:len(features)]
        labels = labels_df[['question1', 'question2']].values
        confidence_scores = labels_df[['question1:confidence', 'question2:confidence']].values
        print(f"Adjusted labels to match features: {len(labels)} labels")

    return features, labels, confidence_scores

# 训练模型
def train_model(features, labels):
    # 初始化模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 分别对每个目标列进行训练和评估
    for i in range(labels.shape[1]):
        print(f"\nTraining model for target column {i + 1}...")
        
        # 提取当前目标列
        y = labels[:, i]
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测测试集
        y_pred = model.predict(X_test)
        
        # 评估模型
        print(f"Classification Report for target column {i + 1}:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy for target column {i + 1}: {accuracy_score(y_test, y_pred)}")
        
        # 保存模型（可选）
        import joblib
        joblib.dump(model, f"data/trained_model_target_{i + 1}.pkl")
        print(f"Model for target column {i + 1} saved to data/trained_model_target_{i + 1}.pkl")

# 主函数
if __name__ == "__main__":
    features_path = "data/fused_features.npy"
    labels_path = "data/aligned_labels.csv"
    
    # 加载数据
    features, labels, confidence_scores = load_data(features_path, labels_path)
    
    # 训练模型
    train_model(features, labels)
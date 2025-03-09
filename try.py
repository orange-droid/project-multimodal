import json
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class VineDataset(Dataset):
    def __init__(self, file_path, max_length=128):
        """
        初始化Vine数据集
        file_path: JSON文件路径
        max_length: BERT输入的最大长度
        """
        self.comments = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
        # 读取数据
        print("开始读取JSON文件...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if isinstance(item, dict):
                        # 打印第一条数据的结构
                        if len(self.comments) == 0:
                            print("数据字段:", item.keys())
                        
                        # 尝试不同可能的字段名
                        comment_text = None
                        for field in ['commentText', 'comment', 'text', 'content', 'message']:
                            if field in item:
                                comment_text = str(item[field])
                                break
                        
                        if comment_text and len(comment_text.strip()) > 0:
                            self.comments.append(comment_text)
                except json.JSONDecodeError as e:
                    print(f"警告：第{line_num}行解析失败: {str(e)}")
                    continue
                except Exception as e:
                    print(f"警告：处理第{line_num}行时出错: {str(e)}")
                    continue
        
        print(f"成功加载评论数量: {len(self.comments)}")
        if len(self.comments) > 0:
            print("第一条评论示例:", self.comments[0][:100])  # 只打印前100个字符
                    
    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, idx):
        text = self.comments[idx]
        
        # 使用BERT tokenizer处理文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': text
        }

class BertTextProcessor:
    def __init__(self, batch_size=16):  # 减小batch_size以降低内存使用
        """
        初始化BERT文本处理器
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.batch_size = batch_size
        
    def process_dataset(self, dataset):
        """
        处理整个数据集并获取BERT特征
        """
        if len(dataset) == 0:
            raise ValueError("数据集为空，无法处理")
            
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="处理数据"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 获取BERT输出
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 使用[CLS]标记的输出作为特征表示
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_features.append(features)
                
                # 释放GPU内存
                del outputs
                torch.cuda.empty_cache()
        
        if not all_features:
            raise ValueError("没有生成任何特征")
            
        return np.vstack(all_features)

def main():
    try:
        # 设置文件路径
        file_path = 'data/sampled_post-comments_vine.json'
        
        # 创建数据集
        print("加载数据集...")
        dataset = VineDataset(file_path)
        print(f"总共加载了 {len(dataset)} 条评论")
        
        if len(dataset) == 0:
            raise ValueError("没有成功加载任何评论数据")
        
        # 初始化BERT处理器
        print("初始化BERT模型...")
        processor = BertTextProcessor()
        
        # 处理数据
        print("开始处理文本数据...")
        features = processor.process_dataset(dataset)
        
        # 输出特征信息
        print(f"\n特征提取完成:")
        print(f"特征矩阵形状: {features.shape}")
        print(f"每条评论的特征维度: {features.shape[1]}")
        
        # 可以保存特征供后续使用
        np.save('vine_bert_features.npy', features)
        print("\n特征已保存到 vine_bert_features.npy")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("\n请检查数据文件格式和内容是否正确")

if __name__ == "__main__":
    main()

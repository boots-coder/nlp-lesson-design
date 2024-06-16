import pandas as pd
from sklearn.model_selection import train_test_split
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import os

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 读取数据集
data_path = '../data/SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/dev_triplets.txt'

# 加载数据
data = []
with open(data_path, 'r') as file:
    for line in file:
        text, label = line.strip().split('####')
        data.append((text, label))

# 转换为DataFrame
df = pd.DataFrame(data, columns=['sentence', 'label'])

# 简单预处理标签
df['label'] = df['label'].apply(lambda x: 0 if 'NEG' in x else (1 if 'NEU' in x else 2))

# 划分训练和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_texts = train_df['sentence'].tolist()
train_labels = train_df['label'].tolist()
test_texts = test_df['sentence'].tolist()
test_labels = test_df['label'].tolist()

# 设置HTTP和HTTPS代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Remove batch dimension
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

# 初始化BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, output_hidden_states=True)
model.to(device)  # 将模型移动到设备

# 创建数据集和数据加载器
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练模型
start_time = time.time()
model.train()
for epoch in range(10):  # 训练3个epoch
    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}  # 将数据移动到设备
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

end_time = time.time()
training_time = end_time - start_time

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}  # 将数据移动到设备
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

# 评估模型并提取第一句话的向量表示
with torch.no_grad():
    # 取第一个batch中的第一句话
    first_batch = next(iter(test_loader))
    first_batch = {k: v.to(device) for k, v in first_batch.items()}  # 将数据移动到设备
    outputs = model(**first_batch)
    cls_embedding = outputs.hidden_states[-1][:, 0, :]  # BERT最后一层的[CLS]标记向量
    print("First sentence CLS embedding:", cls_embedding[0])

print("BERT Accuracy:", correct / total)
print("Training Time:", training_time, "seconds")
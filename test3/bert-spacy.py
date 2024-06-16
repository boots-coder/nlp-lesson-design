import pandas as pd
from sklearn.model_selection import train_test_split
import time
import torch
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from torch.utils.data import DataLoader, Dataset
import spacy
import os

# 检查是否有MPS设备
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
    def __init__(self, texts, labels, tokenizer, max_length, nlp):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.nlp = nlp

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 提取句法特征
        doc = self.nlp(text)
        syntax_features = [token.dep_ for token in doc]

        # 使用BERT分词器对文本进行编码
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True,
                                max_length=self.max_length)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Remove batch dimension

        # 初始化句法特征张量
        syntax_tensor = torch.zeros(self.max_length, len(self.nlp.get_pipe("parser").labels))
        for i, dep in enumerate(syntax_features[:self.max_length]):
            if dep in self.nlp.get_pipe("parser").labels:
                syntax_tensor[i][self.nlp.get_pipe("parser").labels.index(dep)] = 1

        # 计算句法特征的平均值
        syntax_features_avg = torch.mean(syntax_tensor, dim=0)

        # 将句法特征张量添加到inputs字典中
        inputs['syntax'] = syntax_features_avg
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs


# 初始化spaCy模型
nlp = spacy.load("en_core_web_sm")

# 初始化BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 定义自定义BERT模型
class CustomBERTModel(torch.nn.Module):
    def __init__(self, num_labels, syntax_feature_size):
        super(CustomBERTModel, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.output_hidden_states = True  # 启用hidden states输出
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=config)
        self.classifier = torch.nn.Linear(config.hidden_size + syntax_feature_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, syntax=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token output
        combined_output = torch.cat((cls_output, syntax), dim=1)  # Combine with syntax features
        logits = self.classifier(combined_output)
        return logits, outputs.hidden_states


# 初始化自定义BERT模型
syntax_feature_size = len(nlp.get_pipe("parser").labels)
model = CustomBERTModel(num_labels=3, syntax_feature_size=syntax_feature_size).to(device)

# 创建数据集和数据加载器
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=128, nlp=nlp)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length=128, nlp=nlp)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 优化器和损失函数
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
start_time = time.time()
model.train()
for epoch in range(10):  # 训练3个epoch
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
        syntax = batch['syntax'].to(device)
        labels = batch['labels'].to(device)

        outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           syntax=syntax)
        loss = criterion(outputs, labels)
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
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
        syntax = batch['syntax'].to(device)
        labels = batch['labels'].to(device)

        outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           syntax=syntax)
        predictions = torch.argmax(outputs, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

# 评估模型并提取第一句话的向量表示
with torch.no_grad():
    # 取第一个batch中的第一句话
    first_batch = next(iter(test_loader))
    input_ids = first_batch['input_ids'].to(device)
    attention_mask = first_batch['attention_mask'].to(device)
    token_type_ids = first_batch['token_type_ids'].to(device) if 'token_type_ids' in first_batch else None
    syntax = first_batch['syntax'].to(device)

    outputs, hidden_states = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                   syntax=syntax)
    cls_embedding = hidden_states[-1][:, 0, :]  # BERT最后一层的[CLS]标记向量
    print("First sentence CLS embedding:", cls_embedding[0])

print("BERT Accuracy:", correct / total)
print("Training Time:", training_time, "seconds")
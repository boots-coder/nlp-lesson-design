import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import time

# 读取数据集路径
data_path = '/Users/bootscoder/PycharmProjects/nlp-lesson-design/data/SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/dev_triplets.txt'

# 加载数据，将每行数据读取并存储在列表中
data = []
with open(data_path, 'r') as file:
    for line in file:
        text, label = line.strip().split('####')
        data.append((text, label))

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['sentence', 'label'])

# 简单预处理标签，将文本标签转换为数值标签
df['label'] = df['label'].apply(lambda x: 0 if 'NEG' in x else (1 if 'NEU' in x else 2))

# 划分训练和测试集，80%用于训练，20%用于测试
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_texts = train_df['sentence'].tolist()
train_labels = train_df['label'].tolist()
test_texts = test_df['sentence'].tolist()
test_labels = test_df['label'].tolist()

# 加载GloVe词向量
glove = GloVe(name='6B', dim=100)
tokenizer = get_tokenizer("basic_english")

# 创建单词到索引的映射
vocab = {word: idx for idx, word in enumerate(glove.itos)}


# 自定义数据集类
class GloveDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text)
        indices = [self.vocab[token] if token in self.vocab else self.vocab['unk'] for token in tokens]
        return torch.tensor(indices), torch.tensor(label)


# 自定义collate函数，用于将不同长度的文本序列填充到相同长度
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab['pad'])
    labels = torch.tensor(labels)
    return texts, labels, lengths


# 创建数据集和数据加载器
train_dataset = GloveDataset(train_texts, train_labels, vocab, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_dataset = GloveDataset(test_texts, test_labels, vocab, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


# 定义基于GloVe词向量的模型，增加注意力机制
class GloveAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(GloveAttentionModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(glove.vectors)  # 加载预训练的GloVe词向量
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)  # 定义LSTM层
        self.attention = nn.Linear(hidden_dim, 1)  # 定义注意力层
        self.fc = nn.Linear(hidden_dim, num_classes)  # 定义全连接层

    def forward(self, x, lengths):
        x = self.embedding(x)  # 获取词向量
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)  # 传递LSTM层
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 注意力机制
        attn_weights = torch.tanh(self.attention(output)).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_output = torch.sum(output * attn_weights.unsqueeze(-1), dim=1)

        x = self.fc(attn_output)  # 通过全连接层得到输出
        return x


# 初始化模型、损失函数和优化器
model = GloveAttentionModel(len(vocab), 100, 128, 3)  # hidden_dim设为128，可以根据需要调整
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
start_time = time.time()
model.train()
for epoch in range(3):  # 训练3个epoch
    for texts, labels, lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
end_time = time.time()
training_time = end_time - start_time  # 计算训练时间

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for texts, labels, lengths in test_loader:
        outputs = model(texts, lengths)
        predictions = torch.argmax(outputs, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

# 打印模型准确性和训练时间
print("GloVe Attention Model Accuracy:", correct / total)
print("Training Time:", training_time, "seconds")
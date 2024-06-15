import os
import re
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch_geometric
# 确保下载 spaCy 模型
import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# 加载数据集
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')
    return data

# 数据清洗
def preprocess_data(data):
    sentences = []
    aspects = []
    opinions = []
    sentiments = []
    for line in data:
        sentence, labels = line.split('####')
        labels = eval(labels)
        for label in labels:
            aspects.append(label[0])
            opinions.append(label[1])
            sentiments.append(label[2])
        sentences.append(sentence)
    return sentences, aspects, opinions, sentiments

# 加载和预处理数据
data = load_data('/Users/bootscoder/PycharmProjects/nlp-lesson-design/data/SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/dev_triplets.txt')
sentences, aspects, opinions, sentiments = preprocess_data(data)

# 标签编码
le = LabelEncoder()
sentiments_encoded = le.fit_transform(sentiments)

# 加载依存句法分析模型
nlp = spacy.load('en_core_web_sm')

def dependency_parse(sentence):
    doc = nlp(sentence)
    edges = []
    for token in doc:
        for child in token.children:
            edges.append((token.i, child.i))
    return doc, edges

def build_graph(sentence, sentiment):
    doc, edges = dependency_parse(sentence)
    G = nx.Graph()
    G.add_nodes_from([(i, {"word": token.text, "pos": token.pos_, "x": token.vector}) for i, token in enumerate(doc)])
    G.add_edges_from(edges)
    data = from_networkx(G)
    data.y = torch.tensor([sentiment], dtype=torch.long)
    return data

# 构建图结构
graph_data = [build_graph(sentence, sentiment) for sentence, sentiment in zip(sentences, sentiments_encoded)]

# 数据划分
train_data, test_data = train_test_split(graph_data, test_size=0.2, random_state=42)

# 数据加载器
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 图注意力机制模型
class GATModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATModel, self).__init__()
        self.gat = GATConv(in_features, out_features)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return x

# 情感分析模型
class SentimentAnalysisModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(SentimentAnalysisModel, self).__init__()
        self.gat = GATModel(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        features = self.gat(x, edge_index)
        features = torch_geometric.nn.global_mean_pool(features, batch)
        output = self.fc(features)
        return F.log_softmax(output, dim=1)

# 加载词嵌入（假设我们使用预训练的嵌入）
embedding_dim = 96  # 使用 spaCy 的词向量维度
hidden_dim = 64
output_dim = len(le.classes_)  # 'POS', 'NEG', 'NEU'

model = SentimentAnalysisModel(embedding_dim, hidden_dim, output_dim)
device = torch.device('cpu')  # 强制使用 CPU
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练模型
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试模型
def test(loader):
    model.eval()
    correct = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            predictions.extend(pred.cpu().numpy())
            targets.extend(data.y.cpu().numpy())
    acc = correct / len(loader.dataset)
    f1 = f1_score(targets, predictions, average='weighted')
    return acc, f1

if __name__ == "__main__":
    # 训练和评估模型
    for epoch in range(1, 15):
        loss = train()
        train_acc, train_f1 = test(train_loader)
        test_acc, test_f1 = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
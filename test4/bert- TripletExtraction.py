import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
import re
import time
import numpy as np


class TripletExtractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, annotations = self.data[index]
        encoding = self.tokenizer(sentence, max_length=self.max_len, truncation=True, padding='max_length',
                                  return_tensors='pt')
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        target_labels = [-100] * self.max_len
        opinion_labels = [-100] * self.max_len
        sentiment_labels = [-100] * self.max_len

        for (target_pos, opinion_pos, sentiment) in annotations:
            for pos in target_pos:
                if pos < self.max_len:
                    target_labels[pos] = 1
            for pos in opinion_pos:
                if pos < self.max_len:
                    opinion_labels[pos] = 1
            for pos in target_pos:
                if pos < self.max_len:
                    sentiment_labels[pos] = 1 if sentiment == 'POS' else (2 if sentiment == 'NEG' else 3)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_labels': torch.tensor(target_labels, dtype=torch.long),
            'opinion_labels': torch.tensor(opinion_labels, dtype=torch.long),
            'sentiment_labels': torch.tensor(sentiment_labels, dtype=torch.long),
        }


def parse_annotations(annotation_str):
    triplets = re.findall(r'\(\[(.*?)\], \[(.*?)\], \'(.*?)\'\)', annotation_str)
    annotations = []
    for target_pos_str, opinion_pos_str, sentiment in triplets:
        target_pos = list(map(int, target_pos_str.split(', ')))
        opinion_pos = list(map(int, opinion_pos_str.split(', ')))
        annotations.append((target_pos, opinion_pos, sentiment))
    return annotations


def preprocess_data(file_path, tokenizer):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                sentence, annotation_str = line.strip().split('####')
                annotations = parse_annotations(annotation_str)
                data.append((sentence, annotations))
            except Exception as e:
                print(f"意外错误: {e} 在行: {line}")
    return TripletExtractionDataset(data, tokenizer)


class TripletExtractionModel(torch.nn.Module):
    def __init__(self, model_name):
        super(TripletExtractionModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(model_name,
                                                               num_labels=4)  # 0: no label, 1: target, 2: opinion, 3: sentiment

    def forward(self, input_ids, attention_mask, target_labels=None, opinion_labels=None, sentiment_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=target_labels)
        loss = outputs.loss

        opinion_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=opinion_labels)
        loss += opinion_outputs.loss

        sentiment_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=sentiment_labels)
        loss += sentiment_outputs.loss

        return loss, outputs.logits, opinion_outputs.logits, sentiment_outputs.logits


from transformers import AdamW, get_linear_schedule_with_warmup


def train_model(model, dataloader, optimizer, scheduler, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()  # 记录epoch开始时间
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_labels = batch['target_labels'].to(device)
            opinion_labels = batch['opinion_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)

            loss, _, _, _ = model(input_ids, attention_mask, target_labels, opinion_labels, sentiment_labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time  # 计算epoch所用时间
        print(f'Epoch {epoch + 1}, 平均损失: {avg_loss:.4f}, 耗时: {epoch_time:.2f} 秒')


def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    correct_target_predictions = 0
    correct_opinion_predictions = 0
    correct_sentiment_predictions = 0
    total_target_predictions = 0
    total_opinion_predictions = 0
    total_sentiment_predictions = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_labels = batch['target_labels'].to(device)
        opinion_labels = batch['opinion_labels'].to(device)
        sentiment_labels = batch['sentiment_labels'].to(device)

        with torch.no_grad():
            loss, target_logits, opinion_logits, sentiment_logits = model(input_ids, attention_mask, target_labels,
                                                                          opinion_labels, sentiment_labels)
            total_loss += loss.item()

            target_preds = torch.argmax(target_logits, dim=-1)
            opinion_preds = torch.argmax(opinion_logits, dim=-1)
            sentiment_preds = torch.argmax(sentiment_logits, dim=-1)

            correct_target_predictions += (target_preds == target_labels).sum().item()
            correct_opinion_predictions += (opinion_preds == opinion_labels).sum().item()
            correct_sentiment_predictions += (sentiment_preds == sentiment_labels).sum().item()

            total_target_predictions += torch.sum(target_labels != -100).item()  # 忽略填充部分
            total_opinion_predictions += torch.sum(opinion_labels != -100).item()
            total_sentiment_predictions += torch.sum(sentiment_labels != -100).item()

    avg_loss = total_loss / len(dataloader)
    target_accuracy = correct_target_predictions / total_target_predictions
    opinion_accuracy = correct_opinion_predictions / total_opinion_predictions
    sentiment_accuracy = correct_sentiment_predictions / total_sentiment_predictions

    print(f'评估平均损失: {avg_loss:.4f}')
    print(f'目标准确率: {target_accuracy:.4f}')
    print(f'观点准确率: {opinion_accuracy:.4f}')
    print(f'情感准确率: {sentiment_accuracy:.4f}')


# 确保设备分配正确
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TripletExtractionModel('bert-base-uncased').to(device)

# 加载数据
train_dataset = preprocess_data(
    '/Users/bootscoder/PycharmProjects/nlp-lesson-design/data/SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/train_triplets.txt',
    tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = preprocess_data(
    '/Users/bootscoder/PycharmProjects/nlp-lesson-design/data/SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/test_triplets.txt',
    tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 设置优化器和调度器
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_dataloader) * 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型
train_model(model, train_dataloader, optimizer, scheduler, num_epochs=3)

# 评估模型
evaluate_model(model, val_dataloader)
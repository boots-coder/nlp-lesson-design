import spacy
import torch
from transformers import BertModel, BertTokenizer

# 加载依存句法分析工具和预训练模型
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 示例句子
sentence = "This laptop meets every expectation and Windows 7 is great!"

# 依存句法分析1
doc = nlp(sentence)
for token in doc:
    print(token.text, token.dep_, token.head.text)

# BERT表示
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
sentence_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()

print("Sentence Embedding:", sentence_embedding)
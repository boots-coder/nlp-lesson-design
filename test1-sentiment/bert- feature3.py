import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# 示例句子
sentences = [
    "He went to the bank to deposit money.",  # 使用 "bank"
    "He went to the store to deposit money."  # 使用 "store"
]

# 要比较的词
target_words = ["bank", "store"]

word_embeddings = []

for sentence, target_word in zip(sentences, target_words):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'][0]  # 获取分词后的输入 ID
    tokens = tokenizer.convert_ids_to_tokens(input_ids)  # 将 ID 转换为对应的词
    target_indices = [i for i, token in enumerate(tokens) if token == target_word]  # 找到目标词的索引位置

    with torch.no_grad():
        outputs = model(**inputs)

    # 提取目标词的向量表示
    for index in target_indices:
        word_embedding = outputs.hidden_states[-1][0][index]  # 最后一层的目标词标记向量
        word_embeddings.append(word_embedding.numpy())
        print(f"Sentence: {sentence}")
        print(f"Target word: {target_word}")
        print(f"Token index: {index}")
        print(f"Embedding: {word_embedding[:5]}")  # 只打印前5维向量以简化输出

# 计算两个词向量的余弦相似度
cos_sim = cosine_similarity([word_embeddings[0]], [word_embeddings[1]])[0][0]
print(f"Cosine similarity between 'bank' and 'store' in similar contexts: {cos_sim}")


print("###########################################################################################")
print("###########################################################################################")
print("###########################################################################################")
print("###########################################################################################")

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# 示例句子
sentences = ["He went to the bank to deposit money.", "The river bank was covered in snow."]

bank_embeddings = []

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'][0]  # 获取分词后的输入 ID
    tokens = tokenizer.convert_ids_to_tokens(input_ids)  # 将 ID 转换为对应的词
    bank_indices = [i for i, token in enumerate(tokens) if token == 'bank']  # 找到 "bank" 的索引位置

    with torch.no_grad():
        outputs = model(**inputs)

    # 提取 "bank" 词的向量表示
    for index in bank_indices:
        bank_embedding = outputs.hidden_states[-1][0][index]  # 最后一层的 "bank" 标记向量
        bank_embeddings.append(bank_embedding.numpy())
        print(f"Sentence: {sentence}")
        print(f"Bank token index: {index}")
        print(f"Bank embedding: {bank_embedding[:5]}")  # 只打印前5维向量以简化输出

# 计算两个 "bank" 向量的余弦相似度
cos_sim = cosine_similarity([bank_embeddings[0]], [bank_embeddings[1]])[0][0]
print(f"Cosine similarity between 'bank' in different contexts: {cos_sim}")
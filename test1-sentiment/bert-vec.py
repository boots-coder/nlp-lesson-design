import torch
from transformers import BertTokenizer, BertModel

# 初始化BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# 示例句子
sentence = "BERT is a powerful language model."


# 对句子进行分词
inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', truncation=True, max_length=128)

# 将分词后的输入传递给模型
with torch.no_grad():
    outputs = model(**inputs)

# 获取[CLS]标记的向量表示
cls_embedding = outputs.hidden_states[-1][:, 0, :]  # 最后一层的[CLS]标记向量

# 打印[CLS]标记的向量表示
print("CLS embedding for the sentence:", cls_embedding[0])
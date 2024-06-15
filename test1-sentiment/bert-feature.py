import torch
from transformers import BertTokenizer, BertModel

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# 示例句子
sentences = ["He went to the bank to deposit money.", "The river bank was covered in snow."]

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取[CLS]标记的向量表示
    cls_embedding = outputs.hidden_states[-1][:, 0, :]  # 最后一层的[CLS]标记向量
    print(f"Sentence: {sentence}")
    print(f"CLS embedding: {cls_embedding[0][:5]}")  # 只打印前5维向量以简化输出
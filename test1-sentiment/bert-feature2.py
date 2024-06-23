import torch
from transformers import BertTokenizer, BertModel

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# 示例句子
sentences = ["He went to the bank to deposit money.", "The river bank was covered in snow."]

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
        print(f"Sentence: {sentence}")
        print(f"Bank token index: {index}")
        print(f"Bank embedding: {bank_embedding[:5]}")  # 只打印前5维向量以简化输出
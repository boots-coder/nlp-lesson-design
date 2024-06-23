import numpy as np

# 加载GloVe词向量
embedding_index = {}
with open('../test3/.vector_cache/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# 定义一个函数获取GloVe词向量
def get_glove_embedding(word):
    return embedding_index.get(word, np.zeros(100))

# 示例句子
sentence1 = "He went to the bank to deposit money."
sentence2 = "The river bank was covered in snow."

# 获取“bank”的词向量
bank_embedding1 = get_glove_embedding("bank")
bank_embedding2 = get_glove_embedding("bank")

print("GloVe embedding for 'bank' in sentence 1:", bank_embedding1[:5])  # 打印前5维以简化输出
print("GloVe embedding for 'bank' in sentence 2:", bank_embedding2[:5])  # 打印前5维以简化输出

# 比较两个向量是否相同
print("Are the embeddings the same?", np.array_equal(bank_embedding1, bank_embedding2))
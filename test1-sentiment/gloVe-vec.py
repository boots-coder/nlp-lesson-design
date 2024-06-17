import torch
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vectors
# 加载预训练的GloVe词向量
path = '/Users/bootscoder/PycharmProjects/nlp-lesson-design/test3/.vector_cache/glove.6B.100d.txt'
glove = Vectors(path)

# 创建单词到索引的映射
vocab = {word: idx for idx, word in enumerate(glove.itos)}

# 定义分词器
tokenizer = get_tokenizer("basic_english")

# 示例句子
sentence = "This is an example sentence."

# 将句子分词并转换为GloVe词向量
tokens = tokenizer(sentence)
indices = [vocab[token] if token in vocab else vocab['unk'] for token in tokens]
vectors = [glove.vectors[idx] for idx in indices]

# 打印结果
print("Tokens:", tokens)
print("Indices:", indices)
print("Vectors:", vectors)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载 GloVe 向量
def load_glove_model(glove_file):
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    return model

# 计算余弦相似度
def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# 文件路径
glove_file = '/Users/bootscoder/PycharmProjects/nlp-lesson-design/test3/.vector_cache/glove.6B.100d.txt'

# 加载模型
glove_model = load_glove_model(glove_file)

# 获取单词的向量
king_vec = glove_model.get("king")
woman_vec = glove_model.get("woman")
queen_vec = glove_model.get("queen")

if king_vec is not None and woman_vec is not None and queen_vec is not None:
    # 计算 king + woman 的向量
    king_plus_woman = king_vec + woman_vec

    # 计算 king + woman 和 queen 的余弦相似度
    similarity = cosine_sim(king_plus_woman, queen_vec)

    # 打印向量
    print(f"'King' vector (first 5 dimensions): {king_vec[:5]}")
    print(f"'Woman' vector (first 5 dimensions): {woman_vec[:5]}")
    print(f"'Queen' vector (first 5 dimensions): {queen_vec[:5]}")
    print(f"'King + Woman' vector (first 5 dimensions): {king_plus_woman[:5]}")
    print(f"Cosine similarity between 'king + woman' and 'queen': {similarity}")

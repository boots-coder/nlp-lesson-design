import numpy as np

# 示例glove_vectors_list
glove_vectors_list = [
    np.random.rand(10, 100),  # 第一组向量序列，10个100维向量
    np.random.rand(8, 100)    # 第二组向量序列，8个100维向量
]
labels = [1, 1]  # 标签：1表示POS

# 计算Self-Attention得分矩阵
def compute_attention_scores(vectors):
    score_matrix = np.dot(vectors, vectors.T)
    return score_matrix

# 寻找和标签最接近的短语
def find_best_matching_phrase(attention_scores, label):
    # 这里假设最接近标签的短语是得分最高的两个向量
    # 简单示例：找到最大值及其索引
    max_score = np.max(attention_scores)
    indices = np.unravel_index(np.argmax(attention_scores, axis=None), attention_scores.shape)
    return indices

# 寻找与短语最接近的主语
def find_closest_subject(vectors, phrase_indices):
    phrase_vector = np.mean(vectors[list(phrase_indices)], axis=0)
    scores = np.dot(vectors, phrase_vector)
    closest_index = np.argmax(scores)
    return closest_index

# 主函数
def main(glove_vectors_list, labels):
    results = []
    for i, vectors in enumerate(glove_vectors_list):
        attention_scores = compute_attention_scores(vectors)
        phrase_indices = find_best_matching_phrase(attention_scores, labels[i])
        closest_subject_index = find_closest_subject(vectors, phrase_indices)
        results.append((list(phrase_indices), [closest_subject_index], 'POS' if labels[i] == 1 else 'NEU' if labels[i] == 0 else 'NEG'))
    return results

# 执行主函数
result = main(glove_vectors_list, labels)
print(result)
import re


# 定义一个函数来解析单个评论及其标签
def parse_review(review_line):
    # 使用正则表达式分割评论和标签部分
    parts = re.split(r'####', review_line)
    review_text = parts[0].strip()
    tags_part = parts[1].strip()

    # 使用 eval 解析标签部分 (注意：eval 有安全风险，仅在明确数据源安全时使用)
    tags = eval(tags_part)

    return review_text, tags


# 读取文件并解析每一行
reviews = []
with open(    '/Users/bootscoder/PycharmProjects/nlp-lesson-design/data/SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/train_triplets.txt', 'r') as file:
    for line in file:
        review_text, tags = parse_review(line)
        reviews.append((review_text, tags))

# 输出解析后的数据
for review in reviews:
    print(review)





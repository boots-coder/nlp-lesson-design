import torch
from transformers import BertTokenizer
import re
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


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess_data(texts, annotations, tokenizer):
    input_ids = []
    attention_masks = []
    labels = []

    for text, ann in zip(texts, annotations):
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

        # 初始化标签为O
        label = ['O'] * len(encoded_dict['input_ids'][0])

        # 根据注释标记标签
        for (subj_indices, obj_indices, relation) in ann:
            for idx in subj_indices:
                label[idx + 1] = 'B-SUB'
            for idx in obj_indices:
                label[idx + 1] = 'B-OBJ'

        labels.append(label)

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), labels


input_ids, attention_masks, labels = preprocess_data(review_text, annotations, tokenizer)
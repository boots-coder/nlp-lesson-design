import spacy
from spacy.tokens import Span

# 加载Spacy模型
nlp = spacy.load("en_core_web_sm")

# 提取的句子
sentences = [
    "In the shop, these MacBooks are encased in a soft rubber enclosure - so you will never know about the razor edge until you buy it, get it home, break the seal and use it (very clever con).",
    "This laptop meets every expectation and Windows 7 is great!"
]

# 定义一个函数返回基于句法依存树的句子分割
def syntactic_span_segmentation(doc):
    spans = []
    for token in doc:
        # 找到谓语动词及其子树，作为一个跨度
        if token.dep_ == 'ROOT':
            subtree = list(token.subtree)
            if len(subtree) <= 5:
                spans.append((subtree[0].i, subtree[-1].i))
        # 找到所有的名词短语及其子树，作为一个跨度
        if token.dep_ in ('nsubj', 'dobj', 'pobj', 'attr'):
            subtree = list(token.subtree)
            if len(subtree) <= 5:
                spans.append((subtree[0].i, subtree[-1].i))
    return spans

# 定义一个函数，将跨度转换为单词列表
def spans_to_words(doc, spans):
    words = []
    for start, end in spans:
        span_words = doc[start:end+1]
        words.append(span_words.text)
    return words

# 生成基于句法依存树的跨度列表，并转换为单词列表
spans_list = []
words_list = []
for sentence in sentences:
    doc = nlp(sentence)
    spans = syntactic_span_segmentation(doc)
    spans_list.append(spans)
    words = spans_to_words(doc, spans)
    words_list.append(words)

print(spans_list)
# 输出结果
for words in words_list:
    print(words)

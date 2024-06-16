import spacy
from textblob import TextBlob

# 加载英文模型
nlp = spacy.load('en_core_web_sm')

# 示例文本
text = """
This laptop meets every expectation and Windows 7 is great!
Drivers updated ok but the BIOS update froze the system up and the computer shut down.
It rarely works and when it does it's incredibly slow.
The battery life is amazing and the screen is very clear.
The keyboard feels cheap and the touchpad is unresponsive.
"""

# 处理文本
doc = nlp(text)


# 提取情感形容词及其修饰的名词短语的跨度
def extract_sentiment_adjectives(doc):
    all_triplets = []
    for sentence in doc.sents:
        sentence_triplets = []
        # 获取句子的整体情感
        sentence_sentiment = TextBlob(sentence.text).sentiment.polarity
        sentence_sentiment_label = 'POS' if sentence_sentiment > 0 else 'NEG' if sentence_sentiment < 0 else 'NEU'
        for token in sentence:
            if token.pos_ == 'ADJ':
                # 查找修饰的名词
                noun = None
                if token.head.pos_ == 'NOUN':
                    noun = token.head
                else:
                    for ancestor in token.ancestors:
                        if ancestor.pos_ == 'NOUN':
                            noun = ancestor
                            break
                if noun is None:
                    for child in token.children:
                        if child.dep_ in ('nsubj', 'dobj') and child.pos_ == 'NOUN':
                            noun = child
                            break
                if noun is None:
                    for sibling in token.head.children:
                        if sibling.pos_ == 'NOUN' and sibling != token:
                            noun = sibling
                            break

                if noun:
                    # 获取形容词的情感极性
                    adjective_sentiment = TextBlob(token.text).sentiment.polarity
                    adjective_sentiment_label = 'POS' if adjective_sentiment > 0 else 'NEG' if adjective_sentiment < 0 else 'NEU'

                    # 仅保留与句子情感一致的形容词
                    if adjective_sentiment_label == sentence_sentiment_label:
                        # 获取形容词和名词的词序起始位置
                        adjective_index = token.i - sentence.start
                        noun_index = noun.i - sentence.start

                        sentence_triplets.append(f"({noun_index}, {adjective_index}, {adjective_sentiment_label})")

        if sentence_triplets:
            all_triplets.append((sentence.text.strip(), sentence_triplets))

    return all_triplets


# 提取并输出结果
all_triplets = extract_sentiment_adjectives(doc)
print("Extracted Sentiment Triplets:")
for sentence, triplets in all_triplets:
    print(f"Sentence: {sentence}")
    print("Triplets:", " ".join(triplets))
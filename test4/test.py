import spacy
from textblob import TextBlob

# 加载英文模型
nlp = spacy.load('en_core_web_sm')

# 示例评论
reviews = [
    "This laptop meets every expectation and Windows 7 is great!",
    "Drivers updated ok but the BIOS update froze the system up and the computer shut down.",
    "It rarely works and when it does it's incredibly slow.",
    "The battery life is amazing and the screen is very clear.",
    "The keyboard feels cheap and the touchpad is unresponsive."
]


# 提取情感三元组
def extract_sentiment_triplets(reviews):
    triplets = []
    for review in reviews:
        doc = nlp(review)
        sentiment = TextBlob(review).sentiment.polarity
        if sentiment > 0:
            sentiment_label = 'POS'
        elif sentiment < 0:
            sentiment_label = 'NEG'
        else:
            sentiment_label = 'NEU'

        # print(f"Review: {review}")
        # print(f"Overall Sentiment: {sentiment_label}")

        # 获取所有名词短语作为潜在的实体和属性
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        # print(f"Noun Phrases: {noun_phrases}")

        for chunk in doc.noun_chunks:
            noun_phrase = chunk.text
            # 查找修饰名词短语的形容词
            adjectives = [token.text for token in chunk.root.head.children if token.dep_ in ("amod", "acomp")]

            for adjective in adjectives:
                # 分析形容词的情感
                phrase_sentiment = TextBlob(adjective).sentiment.polarity
                if phrase_sentiment > 0:
                    phrase_sentiment_label = 'POS'
                elif phrase_sentiment < 0:
                    phrase_sentiment_label = 'NEG'
                else:
                    phrase_sentiment_label = 'NEU'

                # print(f"Noun Phrase: {noun_phrase}, Adjective: {adjective}, Sentiment: {phrase_sentiment_label}")

                # 只考虑与整体情感一致的名词短语
                if phrase_sentiment_label == sentiment_label:
                    triplets.append((noun_phrase, adjective, sentiment_label))
                    # print(f"Triplet: ({noun_phrase}, {adjective}, {sentiment_label})")

    return triplets


# 输出结果
triplets = extract_sentiment_triplets(reviews)
print("Extracted Sentiment Triplets:")
for triplet in triplets:
    print(triplet)
from textblob import TextBlob

text = "TextBlob is a simple library for processing textual data."
blob = TextBlob(text)
print(blob.tags)
# 输出: [('TextBlob', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('simple', 'JJ'), ('library', 'NN'), ('for', 'IN'), ('processing', 'VBG'), ('textual', 'JJ'), ('data', 'NNS')]
print(blob.noun_phrases)
# 输出: ['textblob', 'simple library', 'textual data']
testimonial = TextBlob("TextBlob is amazingly simple to use. What a great library!")
print(testimonial.sentiment)
# 输出: Sentiment(polarity=0.39166666666666666, subjectivity=0.4357142857142857)


from textblob import TextBlob

text = "This laptop meets every expectation and Windows 7 is great! However, the keyboard feels cheap and the touchpad is unresponsive. i hate it"

# 创建 TextBlob 对象
blob = TextBlob(text)

# 提取名词短语
print("Noun Phrases:")
for np in blob.noun_phrases:
    print(np)

# 情感分析
print("\nSentiment Analysis:")
for sentence in blob.sentences:
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {sentence.sentiment}")
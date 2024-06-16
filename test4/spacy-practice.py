import spacy

# 加载英文模型
nlp = spacy.load('en_core_web_sm')

# 示例文本
text = """
This laptop meets every expectation and Windows 7 is great!
Drivers updated ok but the BIOS update froze the system up and the computer shut down.
It rarely works and when it does it's incredibly slow.
"""

# 处理文本
doc = nlp(text)

# 输出每个词的跨度
for sentence in doc.sents:
    print(f"Sentence: {sentence}")
    for token in sentence:
        print(f"Token: {token.text}, Start: {token.idx}, End: {token.idx + len(token.text)}")
    print()
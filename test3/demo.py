import pandas as pd
import numpy as np
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Layer
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 加载GloVe词嵌入
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_file = '/Users/bootscoder/PycharmProjects/nlp-lesson-design/test3/.vector_cache/glove.6B.100d.txt'  # 替换为你的GloVe文件路径
embeddings_index = load_glove_embeddings(glove_file)

# 加载数据集
data = pd.read_csv('demo_dataset.csv')
texts = data['text']
labels = data['label']

# 将标签转换为数字
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# 使用Tokenizer对文本进行编码
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 填充序列
max_sequence_length = 100
data = pad_sequences(sequences, maxlen=max_sequence_length)

# 创建词嵌入矩阵
word_index = tokenizer.word_index
embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 自定义注意力层
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# 构建带有注意力机制的LSTM模型
input = Input(shape=(max_sequence_length,))
embedding = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False)(input)
x = Bidirectional(LSTM(100, return_sequences=True))(embedding)
x = AttentionLayer()(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=input, outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(data, labels, epochs=30, batch_size=32)

# 去掉验证数据集 ，容易使得 model 学到歪门邪道
# model.fit(data, labels, epochs=20, batch_size=32, validatation_slpit = 0.2)

# 预测和评估
predictions = model.predict(data)
y_pred = np.argmax(predictions, axis=1)

print(classification_report(labels, y_pred, target_names=label_encoder.classes_))
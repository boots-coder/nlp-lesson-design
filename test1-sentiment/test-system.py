import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

import requests

proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
}

try:
    response = requests.get('https://api.github.com', proxies=proxies)
    print("代理测试成功，状态码：", response.status_code)
except Exception as e:
    print("代理测试失败：", e)


import os
from transformers import BertTokenizer, BertModel

# 设置代理环境变量
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 尝试下载BERT模型
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print("模型下载成功")
except Exception as e:
    print("模型下载失败：", e)


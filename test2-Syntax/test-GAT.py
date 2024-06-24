import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 设置代理环境变量（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 尝试下载BERT模型
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, output_hidden_states=True)
    model.to(device)  # 将模型移动到设备
    print("模型加载成功")
except Exception as e:
    print("模型加载失败：", e)


import torch
import torch.nn as nn
import torch.optim as optim

# 假设的实际值
actual = torch.tensor(
    [-100, -100, -100, 1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
     -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
     -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
     -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
     -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
     -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
     -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100], device='mps:0')

# 将 actual 中的值转换为 0 和 1
labels = (actual == 1).float()


# 创建简单的模型
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, x):
        return self.linear(x)


# 模型实例化
model = SimpleModel(input_size=actual.size(0)).to('mps:0')

# 损失函数和优化器
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练数据
inputs = torch.ones_like(actual).unsqueeze(0).float().to('mps:0')

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = loss_fn(outputs, labels.unsqueeze(0))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    predicted = model(inputs)
    predicted_labels = (predicted.squeeze() > 0).float()
    print("Predicted labels:", predicted_labels)
    print("Actual labels:", labels)
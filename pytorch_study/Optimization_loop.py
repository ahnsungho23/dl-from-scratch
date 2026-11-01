import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# ===== 재료 1: 데이터 (Dataset 섹션) =====
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# ===== 재료 2: 모델 (Build NN 섹션) =====
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)

model = NeuralNetwork()

# ===== 재료 3: 손실 함수 =====
loss_fn = nn.CrossEntropyLoss()

# ===== 재료 4: 옵티마이저 =====
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# ===== 학습 =====
for epoch in range(5):
    total_loss = 0
    for X, y in train_dataloader:
        pred = model(X)             # 예측
        loss = loss_fn(pred, y)     # 손실
        loss.backward()             # 각 가중치마다 본인의 기울기 갖게 된다.
        optimizer.step()            # 가중치 갱신
        optimizer.zero_grad()       # 기울기 지우기
        total_loss += loss.item()   # 손실 기록
    print(epoch, total_loss / len(train_dataloader))
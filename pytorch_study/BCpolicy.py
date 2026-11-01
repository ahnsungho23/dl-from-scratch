import torch
from torch import nn

layer = nn.Linear(10, 256)
print(layer.weight.shape)
print(layer.bias.shape)

x = torch.randn(1, 10)
y = layer(x)
print(y.shape)

relu = nn.ReLU()
x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])
print(relu(x))

net = nn.Sequential(
    nn.Linear(10, 256),
    nn.ReLU(),
    nn.Linear(256, 7),
)

x = torch.randn(1, 10)
print(net(x).shape)

#nn.Module : 가중치를 대신 관리해주는 틀, 책의 Twolayernet에서 W1, b1, W2, b2를 
# 직접 모아서 관리를 하지 않아도 알아서 해준다.
class BCpolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
            )

    def forward(self, x):
        return self.net(x)

policy = BCpolicy()
# 테스트 A: DataLoader에서 나온 배치 (카드 64장)
s_batch = torch.randn(64, 10)
print(policy(s_batch).shape)          # 예상: 64, 7

# 테스트 B: 시뮬에서 받은 상태 하나
s = torch.randn(10)
print(policy(s.unsqueeze(0)).shape)   # 예상: 1, 7
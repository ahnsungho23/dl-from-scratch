import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


# ===== 네가 짠 DemoDataset 복붙 =====
class DemoDataset(Dataset):
    def __init__(self, obs, actions):
        self.obs = torch.from_numpy(obs).float()
        self.actions = torch.from_numpy(actions).float()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

# ===== 네가 짠 BCPolicy 복붙 =====
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

# ===== 가짜 전문가 데모 (이미 완성) =====
W_true = np.random.randn(10, 7)
obs = np.random.randn(1000, 10)
actions = np.tanh(obs @ W_true)    # 전문가의 규칙: 상태 → 행동

# ===== 재료 1: 데이터 =====
ds = DemoDataset(obs, actions)
loader = DataLoader(ds, batch_size = 64, shuffle = True)

# ===== 재료 2: 모델 =====
policy = BCpolicy()
# ===== 재료 3: 손실 함수 =====
loss_fn = nn.MSELoss()

# ===== 재료 4: 옵티마이저 ===== 여기서 넘겨받는 policy.parameters는 원본 파라미터를 그대로 넘긴다.
optimizer = torch.optim.SGD(policy.parameters(), lr = 0.01)
print(policy.state_dict().keys())
# ===== 학습 =====
for epoch in range(50):
    total_loss = 0
    for s_batch, a_batch in loader:
        train = policy(s_batch)
        loss = loss_fn(train, a_batch)
        loss.backward()     #각 가중치마다 본인의 기울기를 갖게 된다.
        optimizer.step()    
        optimizer.zero_grad()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(epoch, total_loss / len(loader))
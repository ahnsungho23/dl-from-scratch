import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# DemoDataset 클래스
class DemoDataset(Dataset):
    def __init__(self, obs, actions):
        self.obs = torch.from_numpy(obs).float()
        self.actions = torch.from_numpy(actions).float()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]
    
# 1. 가짜 데이터: 상태 1000개(각 10개), 행동 1000개(각 7개)
obs = np.random.randn(1000, 10)
actions = np.random.randn(1000, 7)

# 2. 카드 상자 만들기
Card = DemoDataset(obs, actions)

# 3. 64장씩 섞어서 묶는 조교 만들기
loader = DataLoader(Card, batch_size = 64, shuffle = True)

# 4. for문으로 돌면서 배치 번호와 s_batch 모양 출력
for i, (s_batch, a_batch) in enumerate(loader):
    print(i, s_batch.shape)
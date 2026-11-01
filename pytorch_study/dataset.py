import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
)

labels_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

figure = plt.figure(figsize = (8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size = (1, )).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class SquareDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):  #dataset의 길이 반환
        return self.n

    def __getitem__(self, idx):     #dataset[idx]로 접근했을 때 반환되는 값
        return idx, idx ** 2

class DemoDataset(Dataset):
    def __init__(self, obs, actions):
        self.obs = torch.from_numpy(obs).float()
        self.actions = torch.from_numpy(actions).float()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


# 테스트용 가짜 데모 데이터
obs = np.random.randn(1000, 10)
actions = np.random.randn(1000, 7)

ds = DemoDataset(obs, actions)
print(len(ds))                    # 1000이 나와야 함
s, a = ds[0]
print(s)
print(s.shape, a.shape, s.dtype)  # torch.Size([10]) torch.Size([7]) torch.float32


# DataLoader를 이용한 배치 처리
train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = False)


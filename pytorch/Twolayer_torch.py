# 5장 TwoLayerNet(NumPy)을 PyTorch로 다시 구현
# 하이퍼파라미터는 ch05/train_neuralnet.py와 같게 맞춰서 결과를 비교한다.
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(ROOT, 'data')        # MNIST 다운로드 위치 (.gitignore에 /data/ 있음)
RESULT_DIR = os.path.join(ROOT, 'results')   # 그래프 저장 위치
os.makedirs(RESULT_DIR, exist_ok=True)

torch.manual_seed(0)   # 실행할 때마다 같은 결과가 나오도록 난수 고정

# ===== 하이퍼파라미터 (5장과 동일) =====
hidden_size = 50
batch_size = 100
learning_rate = 0.1
epochs = 16            # 5장: 10000 iteration / (60000 / 100) ≈ 16.7 epoch

# ===== 재료 1: 데이터 =====
# 5장의 load_mnist(normalize=True)에 해당: 이미지 → 텐서, 0~1로 정규화
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)    # 학습용 (섞기)
train_eval_loader = DataLoader(train_data, batch_size=1000, shuffle=False)    # 훈련 정확도 측정용
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)           # 테스트 정확도 측정용


# ===== 재료 2: 모델 =====
# 5장 TwoLayerNet: Affine1 → Relu1 → Affine2 → (SoftmaxWithLoss)
class TwoLayerNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=50, output_size=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                          # (N, 1, 28, 28) → (N, 784)
            nn.Linear(input_size, hidden_size),    # Affine1
            nn.ReLU(),                             # Relu1
            nn.Linear(hidden_size, output_size),   # Affine2
        )

    def forward(self, x):
        return self.net(x)   # 소프트맥스 전 점수(logits). 소프트맥스는 손실 함수 안에 있음


model = TwoLayerNet(hidden_size=hidden_size)

# ===== 재료 3: 손실 함수 =====
loss_fn = nn.CrossEntropyLoss()   # 5장 SoftmaxWithLoss 계층

# ===== 재료 4: 옵티마이저 =====
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # 5장 params[key] -= lr * grad[key]


def accuracy(loader):
    """5장 TwoLayerNet.accuracy()에 해당"""
    model.eval()
    correct = 0
    with torch.no_grad():                                  # 평가할 때는 기울기 기록 끄기
        for X, y in loader:
            pred = model(X)
            correct += (pred.argmax(dim=1) == y).sum().item()   # 가장 큰 점수의 번호 = 예측
    model.train()
    return correct / len(loader.dataset)


train_loss_list = []
train_acc_list = []
test_acc_list = []

# 학습 전 정확도 (5장 그래프의 epoch 0에 해당)
train_acc_list.append(accuracy(train_eval_loader))
test_acc_list.append(accuracy(test_loader))
print(f"epoch  0 | train acc {train_acc_list[-1]:.4f} | test acc {test_acc_list[-1]:.4f}")

# ===== 학습 =====
for epoch in range(1, epochs + 1):
    for X, y in train_loader:
        pred = model(X)              # ① 예측
        loss = loss_fn(pred, y)      # ② 손실
        loss.backward()              # ③ 기울기 구하기 (5장 network.gradient())
        optimizer.step()             # ④ 가중치 갱신
        optimizer.zero_grad()        # ⑤ 기울기 지우기
        train_loss_list.append(loss.item())

    train_acc_list.append(accuracy(train_eval_loader))
    test_acc_list.append(accuracy(test_loader))
    print(f"epoch {epoch:2d} | train acc {train_acc_list[-1]:.4f} | test acc {test_acc_list[-1]:.4f}")

print(f"\n최종 훈련 정확도: {train_acc_list[-1]:.4f}")
print(f"최종 테스트 정확도: {test_acc_list[-1]:.4f}")

# ===== 그래프 저장 (5장과 같은 형식) =====
# 1) 손실 곡선 (iteration 단위)
plt.figure()
plt.plot(range(len(train_loss_list)), train_loss_list, label='train loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.savefig(os.path.join(RESULT_DIR, 'ch05_torch_loss.png'), dpi=150, bbox_inches='tight')
plt.close()

# 2) train/test 정확도 (epoch 단위)
epoch_axis = range(len(train_acc_list))
plt.figure()
plt.plot(epoch_axis, train_acc_list, marker='o', label='train acc')
plt.plot(epoch_axis, test_acc_list, marker='s', linestyle='--', label='test acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig(os.path.join(RESULT_DIR, 'ch05_torch_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()
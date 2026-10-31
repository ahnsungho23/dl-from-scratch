import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist
from neuralnet import TwoLayerNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

result_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(result_dir, exist_ok=True)

#데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #오차역전파법으로 기울기를 구한다.
    grad = network.gradient(x_batch, t_batch)

    #갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


# 1) 손실 곡선 (iteration 단위)
plt.figure()
plt.plot(np.arange(len(train_loss_list)), train_loss_list, label='train loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.savefig(os.path.join(result_dir, 'ch05_loss.png'), dpi=150, bbox_inches='tight')
plt.close()

# 2) train/test 정확도 (epoch 단위)
epochs = np.arange(len(train_acc_list))
plt.figure()
plt.plot(epochs, train_acc_list, marker='o', label='train acc')
plt.plot(epochs, test_acc_list, marker='s', linestyle='--', label='test acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig(os.path.join(result_dir, 'ch05_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()

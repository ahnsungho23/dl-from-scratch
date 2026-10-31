import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)     #복사

np_array = np.array(data)
x_np = torch.from_numpy(np_array)   #공유

#like함수는 다른 텐서와 같은 모양으로 새로운 텐서를 만든다.
#ones_like는 같은모양에 값은 1로
#rand_like는 0과 1사이의 난수로 채운다.
x_ones = torch.ones_like(x_data)    
x_rand = torch.rand_like(x_data, dtype=torch.float)
#print(x_ones)
#print(x_rand)

#shape에 맞게 각각 rand, ones, zeros 텐서를 생성
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
'''
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)
print(zeros_tensor.device)'''

#torch.accelerator.is_available은 현재 가속기의 유무
#.to(torch.accelerator.current_accelerator())는
#현재 있는 가속기에 옮겨서 다시저장(여기서 가속기는 GPU)
if torch.accelerator.is_available():
    tensor = zeros_tensor.to(torch.accelerator.current_accelerator())
    print(tensor.device)

tensor = torch.ones(3,4)
print(tensor)
print(tensor[:, 0])
print(tensor[..., -1])
tensor[:, 1] = 0
print(tensor)

#concatenate(연결하다)
#.cat(, dim = 1)이면 옆으로 붙이고
#dim = 0이면 아래로 붙인다.
t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)

#행렬곱
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print(y1)
print(y2)

#각 원소별 곱
z1 = tensor * tensor
z2 = tensor.mul(tensor)
print(z1)
print(z2)

#모든 원소의 합
agg = tensor.sum()
print(agg)  #0차원 숫자
agg_item = agg.item()   #.item은 파이썬 숫자로 꺼내줌
print(agg_item, type(agg_item)) #type = float

#각 원소에 5를 더해줌
tensor = tensor.add(5)
print(tensor)

#.numpy는 텐서를 numpy배열(ndarray)로 바꾸는 함수
t = torch.ones(5)
print(t)
n = t.numpy()
print(n)

#torch.from_numpy(n)은 numpy배열을 텐서로 바꾸는 함수
print(torch.from_numpy(n))
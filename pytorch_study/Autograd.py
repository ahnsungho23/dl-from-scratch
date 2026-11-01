import torch

#requires_grad = True는 어떻게 연산이 되었는지
#알아서 기록하는 블랙박스이다.
#이거를 써줘야 역으로 기울기를 계산 가능하다.
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad = True)
b = torch.randn(3, requires_grad = True)
z = torch.matmul(x, w) + b
#binary_cross_entropy_with_logits는 손실을 계산해주는 손실함수이다.
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
#5장에서 직접 역으로 기울기를 계산한 것을 .backward()로
#알아서 계산을 해준다.
loss.backward()
print(w.grad)
print(b.grad)

#torch.no_grad()는 이 블록에서 계산 그래프를
#기록하지 않는다는 것이다.
#이거는 보통 평가할 때 사용한다.
with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)
    
x = torch.tensor(3.0, requires_grad = True)

for i in range(3):
    y = x ** 2 + 2 * x
    y.backward()
    print(x.grad)
    #.zero_()를 써주지 않으면 기울기가 계속 쌓이기 때문에
    #학습이 망가진다.
    x.grad.zero_()

#예제
x = torch.tensor(2.0, requires_grad = True)
y = x ** 3
y.backward()
print(x.grad())

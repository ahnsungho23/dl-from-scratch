import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2

#현재 위치에서 편미분 한번.
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

        it.iternext()

    return grad

#numerical_gradient를 step_num만큼 반복하여, 
# 최적의 가중치를 찾는것.(loss가 가장 작은 가중치)
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x

if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])

    result = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
#이때 lr은 학습률을 의미하고, step_num은 경사법을 반복하는 횟수를 의미한다.
#학습률과 같은 매개변수를 하이퍼파라미터(hyperparameter)라고 한다. 하이퍼파라미터는 사람이 직접 설정해야 하는 매개변수이다.
    print(result)

#이렇게 경사하강법을 통해서 최솟값을 구한다.
#이 경사하강법을 손실함수에 적용하여, 최적의 가중치 매개변수를 구하는 것이 신경망 학습의 핵심이다.
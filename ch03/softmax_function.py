import numpy as np

def softmax(a):
    c = np.max(a, axis=-1, keepdims=True)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)
    y = exp_a / sum_exp_a
    return y

'''
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))  #무조건 1이 나와야함.
'''

#소프트맥스 함수의 출력은 0 ~ 1사이 실수이고, 출력의 총합이 1이기 
#때문에 출력을 확률로 해석할 수 있다.
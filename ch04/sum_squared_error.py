import numpy as np

def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 정답 레이블(원-핫 인코딩)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 신경망의 출력
print(sum_squared_error(np.array(y), np.array(t)))  #값이 0에 가까울수록 좋다.

#손실함수의 두번째 예시인 sum_squared_error이다.
import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size

if __name__ == '__main__':
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 정답 레이블(원-핫 인코딩)
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 신경망의 출력
    print(cross_entropy_error(np.array(y), np.array(t)))    #값이 0에 가까울수록 좋다.

#손실함수의 첫번째 예시인 cross_entropy_error이다.
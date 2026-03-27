import numpy as np

def fake_quantize(x, ratio, noise_level):
    # 압축: 일부 값만 남기거나 스케일 줄이기
    k = int(len(x) * ratio)
    indices = np.random.choice(len(x), k, replace=False)
    
    compressed = x[indices]
    
    # 노이즈 추가 (복원 오차 시뮬레이션)
    noise = np.random.normal(0, noise_level, size=compressed.shape)
    compressed += noise
    
    return compressed, indices


def fake_dequantize(compressed, indices, original_size):
    x = np.zeros(original_size)
    x[indices] = compressed
    return x

import numpy as np

def fake_quantize(x, ratio, noise_level):
    # 실제 양자화 시뮬레이션: 데이터를 버리는 게 아니라 정밀도를 낮춤
    # 16-bit -> 4-bit (ratio 0.25) 과정에서 발생하는 양자화 오차 재현
    
    # 1. 데이터의 스케일 계산 (Max-Abs Scaling)
    scale = np.max(np.abs(x)) + 1e-9
    
    # 2. 양자화 레벨 설정 (4-bit일 경우 2^4 = 16레벨)
    levels = 2 ** (16 * ratio) 
    
    # 3. 양자화 수행 (Normalize -> Quantize -> De-normalize)
    quantized = np.round((x / scale) * (levels / 2)) / (levels / 2) * scale
    
    # 4. 통신 노이즈 추가
    noise = np.random.normal(0, noise_level, size=x.shape)
    compressed = quantized + noise
    
    return compressed, scale # indices 대신 scale을 반환

def fake_dequantize(compressed, scale, original_size):
    # 양자화된 데이터를 그대로 복원 (이미 x.shape와 같으므로)
    return compressed

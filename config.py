# 데이터 크기 (bytes) - 100MB activation
DATA_SIZE = 100 * 1024 * 1024  

# 네트워크 대역폭 (bytes/sec) - 10 GB/s (약 80Gbps 수준)
BANDWIDTH = 10 * 1024 * 1024 * 1024  

# 압축 비율 (FP16 16-bit 기준)
# 4-bit로 줄이면 4/16 = 0.25
COMPRESSION_RATIO = 0.25  

# 인코딩/디코딩 속도 (bytes/sec) - NPU 가속 시 50 GB/s 설정
ENCODE_SPEED = 50 * 1024 * 1024 * 1024  
DECODE_SPEED = 50 * 1024 * 1024 * 1024

# 복원 오차 시뮬레이션을 위한 노이즈 수준
NOISE_LEVEL = 0.01

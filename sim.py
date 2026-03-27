from config import *
from model import *
from workload import generate_activation
from quant import fake_quantize, fake_dequantize

import numpy as np

def run():
    print("=== TurboQuant Simulation ===")
    
    # Config 요약 출력 (README의 예시와 매칭)
    # BANDWIDTH를 bps 단위에서 Gbps로 변환하여 표시 (예: 100e9 -> 100Gbps)
    bw_gbps = BANDWIDTH / 1e9
    # DATA_SIZE를 바이트 단위에서 GB로 변환 (예: 1e9 -> 1GB)
    data_gb = DATA_SIZE / 1e9
    print(f"Config: BW={bw_gbps:.0f}Gbps, Data={data_gb:.0f}GB, Ratio={COMPRESSION_RATIO} ({int(16*COMPRESSION_RATIO)}-bit)")
    print()

    # latency 비교
    is_better, raw, tq = golden_cross(
        DATA_SIZE,
        BANDWIDTH,
        COMPRESSION_RATIO,
        ENCODE_SPEED,
        DECODE_SPEED
    )
    
    # 세부 시간 계산 (README 출력용)
    t_enc = DATA_SIZE / ENCODE_SPEED
    t_trans = (DATA_SIZE * COMPRESSION_RATIO) / BANDWIDTH
    t_dec = DATA_SIZE / DECODE_SPEED

    print(f"Raw transfer time: {raw:.6f} sec")
    # 전체 TQ 시간 뒤에 세부 지표(Enc, Trans, Dec) 추가
    print(f"TurboQuant time: {tq:.6f} sec (Enc: {t_enc:.3f}s, Trans: {t_trans:.3f}s, Dec: {t_dec:.3f}s)")
    print(f"TurboQuant wins? {is_better}")
    print()
    
    # accuracy 테스트
    x = generate_activation(1000000)
    
    compressed, idx = fake_quantize(x, COMPRESSION_RATIO, NOISE_LEVEL)
    restored = fake_dequantize(compressed, idx, len(x))
    
    error = np.mean((x - restored) ** 2)
    
    print(f"Reconstruction MSE: {error:.6f}")


if __name__ == "__main__":
    run()

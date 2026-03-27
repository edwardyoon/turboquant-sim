import config
from model import *
from workload import generate_activation
from quant import fake_quantize, fake_dequantize

import numpy as np

def run():
    print("=== TurboQuant Simulation ===")
    
    # BANDWIDTH를 bps 단위에서 Gbps로 변환 (10e9 -> 10Gbps)
    bw_gbps = config.BANDWIDTH / 1e9
    # DATA_SIZE를 바이트 단위에서 GB로 변환 (100MB -> 0.1GB)
    data_gb = config.DATA_SIZE / 1e9
    
    print(f"Config: BW={bw_gbps:.0f}Gbps, Data={data_gb:.2f}GB, Ratio={config.COMPRESSION_RATIO} ({int(16*config.COMPRESSION_RATIO)}-bit)")
    print()

    # latency 비교
    is_better, raw, tq = golden_cross(
        config.DATA_SIZE,
        config.BANDWIDTH,
        config.COMPRESSION_RATIO,
        config.ENCODE_SPEED,
        config.DECODE_SPEED
    )
    
    # 세부 시간 계산
    t_enc = config.DATA_SIZE / config.ENCODE_SPEED
    t_trans = (config.DATA_SIZE * config.COMPRESSION_RATIO) / config.BANDWIDTH
    t_dec = config.DATA_SIZE / config.DECODE_SPEED

    print(f"Raw transfer time: {raw:.6f} sec")
    print(f"TurboQuant time: {tq:.6f} sec (Enc: {t_enc:.3f}s, Trans: {t_trans:.3f}s, Dec: {t_dec:.3f}s)")
    print(f"TurboQuant wins? {is_better}")
    print()
    
    # accuracy 테스트
    x = generate_activation(1000000)
    
    # quant.py에서 수정된 scale 기반 양자화 로직 적용
    compressed, meta = fake_quantize(x, config.COMPRESSION_RATIO, config.NOISE_LEVEL)
    restored = fake_dequantize(compressed, meta, len(x))

    error = np.mean((x - restored) ** 2)
    
    print(f"Reconstruction MSE: {error:.6f}")


if __name__ == "__main__":
    run()

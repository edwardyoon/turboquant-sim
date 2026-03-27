from config import *
from model import *
from workload import generate_activation
from quant import fake_quantize, fake_dequantize

import numpy as np

def run():
    print("=== TurboQuant Simulation ===")
    
    # latency 비교
    is_better, raw, tq = golden_cross(
        DATA_SIZE,
        BANDWIDTH,
        COMPRESSION_RATIO,
        ENCODE_SPEED,
        DECODE_SPEED
    )
    
    print(f"Raw transfer time: {raw:.6f} sec")
    print(f"TurboQuant time: {tq:.6f} sec")
    print(f"TurboQuant wins? {is_better}")
    
    # accuracy 테스트
    x = generate_activation(1000000)
    
    compressed, idx = fake_quantize(x, COMPRESSION_RATIO, NOISE_LEVEL)
    restored = fake_dequantize(compressed, idx, len(x))
    
    error = np.mean((x - restored) ** 2)
    
    print(f"Reconstruction MSE: {error:.6f}")


if __name__ == "__main__":
    run()


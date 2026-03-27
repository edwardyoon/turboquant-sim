import config
from model import *
from workload import generate_activation
from quant import fake_quantize, fake_dequantize

import numpy as np

# ratio → bits 변환 (16-bit base 기준)
def ratio_to_bits(ratio):
    return max(1, round(16 * ratio))

def run():
    print("=== TurboQuant Simulation ===")

    GiB      = 1024 ** 3
    bw_gbps  = config.BANDWIDTH / GiB * 8          # bytes/s → Gbps
    data_mib = config.DATA_SIZE / (1024 ** 2)      # bytes → MiB
    bits     = ratio_to_bits(config.COMPRESSION_RATIO)

    print(f"Config: BW={bw_gbps:.0f}Gbps, Data={data_mib:.0f}MiB, "
          f"Ratio={config.COMPRESSION_RATIO} ({bits}-bit)")
    print(f"        Enc/Dec Speed={config.ENCODE_SPEED / GiB:.0f}GiB/s")
    print()

    # --- Latency 비교 ---
    is_better, raw, tq = golden_cross(
        config.DATA_SIZE,
        config.BANDWIDTH,
        config.COMPRESSION_RATIO,
        config.ENCODE_SPEED,
        config.DECODE_SPEED
    )

    t_enc   = config.DATA_SIZE / config.ENCODE_SPEED
    t_trans = (config.DATA_SIZE * config.COMPRESSION_RATIO) / config.BANDWIDTH
    t_dec   = config.DATA_SIZE / config.DECODE_SPEED

    print(f"Raw transfer time : {raw*1000:.4f} ms")
    print(f"TurboQuant time   : {tq*1000:.4f} ms  "
          f"(Enc: {t_enc*1000:.4f}ms, Trans: {t_trans*1000:.4f}ms, Dec: {t_dec*1000:.4f}ms)")
    print(f"Speedup           : {raw/tq:.2f}x")
    print(f"TurboQuant wins?  : {is_better}")
    print()

    # --- Accuracy 테스트 ---
    x = generate_activation(1000000)

    compressed, meta = fake_quantize(x, bits=bits,
                                     noise_level=config.NOISE_LEVEL)
    restored = fake_dequantize(compressed, meta)

    error = np.mean((x - restored) ** 2)
    print(f"Reconstruction MSE: {error:.8f}")


if __name__ == "__main__":
    run()
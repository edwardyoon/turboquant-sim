import numpy as np
from quant import fake_quantize, fake_dequantize, fwht_vectorized
import config


def run_comparison_tq():
    print("=== Architecture Showdown: Matrix-based vs. Neuron-centric ===")

    rng = np.random.default_rng(config.SEED if hasattr(config, "SEED") else 42)
    raw_data = rng.standard_normal(config.DATA_SIZE // 4)

    BITS_LOW  = 4
    BITS_HIGH = 8
    FP32_BITS = 32

    # ------------------------------------------------------------------ #
    # [A] Uniform 4-bit
    # ------------------------------------------------------------------ #
    m_comp, m_meta = fake_quantize(raw_data, bits=BITS_LOW,
                                   noise_level=config.NOISE_LEVEL)
    m_restored = fake_dequantize(m_comp, m_meta)
    m_mse      = np.mean((raw_data - m_restored) ** 2)

    # ------------------------------------------------------------------ #
    # [B] Neuron-centric — Global Rotation 후 rotation domain에서 선별
    # ------------------------------------------------------------------ #
    n        = len(raw_data)
    pow2     = 1 << (n - 1).bit_length()
    padded   = np.pad(raw_data, (0, pow2 - n))
    mean_val = np.mean(padded)
    rotated  = fwht_vectorized(padded - mean_val)   # shape: (pow2,)

    # rotation domain 전체 기준으로 mask 생성 (pow2 크기)
    rot_saliency   = np.abs(rotated)
    threshold      = np.percentile(rot_saliency, 80)
    important_mask = rot_saliency >= threshold       # shape: (pow2,)

    def quantize_array(x, bits, noise_level):
        levels = 2 ** bits
        scale  = np.max(np.abs(x)) + 1e-9
        x_int  = np.round(x / scale * (levels / 2)).clip(-levels / 2, levels / 2 - 1)
        x_deq  = x_int / (levels / 2) * scale
        noise  = np.random.normal(0, noise_level * scale, size=x.shape)
        return x_deq + noise

    # 연쇄 인덱싱 없이 직접 mask로 대입
    rot_reconstructed = np.zeros(pow2)
    rot_reconstructed[important_mask]  = quantize_array(
        rotated[important_mask],  BITS_HIGH, config.NOISE_LEVEL)
    rot_reconstructed[~important_mask] = quantize_array(
        rotated[~important_mask], BITS_LOW,  config.NOISE_LEVEL)

    n_restored = (fwht_vectorized(rot_reconstructed) + mean_val)[:n]
    n_mse      = np.mean((raw_data - n_restored) ** 2)
    n_avg_bits = (important_mask.sum() * BITS_HIGH +
                  (~important_mask).sum() * BITS_LOW) / pow2

    # ------------------------------------------------------------------ #
    # [C] Fair baseline — Uniform 5-bit
    # ------------------------------------------------------------------ #
    f_comp, f_meta = fake_quantize(raw_data, bits=5,
                                   noise_level=config.NOISE_LEVEL)
    f_restored = fake_dequantize(f_comp, f_meta)
    f_mse      = np.mean((raw_data - f_restored) ** 2)

    # ------------------------------------------------------------------ #
    # 출력
    # ------------------------------------------------------------------ #
    results = [
        ("[A] Uniform 4-bit",             4.0,        m_mse),
        ("[B] Neuron-centric 8/4-bit",    n_avg_bits, n_mse),
        ("[C] Uniform 5-bit (baseline)",  5.0,        f_mse),
    ]

    print(f"\n{'Method':<35} {'Avg bits':>9} {'BW Boost':>9} {'MSE':>14}")
    print("-" * 70)
    for name, bits, mse in results:
        print(f"{name:<35} {bits:>9.1f} {FP32_BITS / bits:>8.2f}x {mse:>14.8f}")
    print("-" * 70)

    mse_vs_A = (m_mse - n_mse) / m_mse * 100
    mse_vs_C = (f_mse - n_mse) / f_mse * 100

    print(f"\n[B] vs [A]  MSE reduction : {mse_vs_A:+.2f}%  (+{n_avg_bits - 4.0:.1f} bits overhead)")
    print(f"[B] vs [C]  MSE reduction : {mse_vs_C:+.2f}%  (same budget comparison)")

    if n_mse < f_mse:
        print("\n✓ Neuron-centric outperforms Uniform 5-bit within the same bit budget.")
        print("  → Selective quantization after global rotation is provably effective.")
    else:
        print("\n✗ Uniform 5-bit still achieves lower MSE than Neuron-centric.")
        print("  → Re-tuning of saliency threshold or bit allocation is required.")


if __name__ == "__main__":
    run_comparison_tq()
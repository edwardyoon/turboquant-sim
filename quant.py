import numpy as np

def fwht(a):
    """정규화된 Walsh-Hadamard Transform (self-inverse)"""
    n = len(a)
    assert (n & (n - 1)) == 0, "Length must be power of 2"
    a = a.copy().astype(np.float64)
    d = 1
    while d < n:
        a = a.reshape(-1, 2 * d)
        x, y = a[:, :d].copy(), a[:, d:].copy()
        a[:, :d] = x + y
        a[:, d:] = x - y
        d *= 2
    return a.reshape(-1) / np.sqrt(n)

# comparison.py에서 import하는 이름
fwht_vectorized = fwht

def quantize_to_bits(x, bits):
    levels = 2 ** bits
    scale  = np.max(np.abs(x)) + 1e-9
    x_int  = np.round(x / scale * (levels / 2)).clip(-levels / 2, levels / 2 - 1)
    x_deq  = x_int / (levels / 2) * scale
    return x_deq, scale

def fake_quantize(x, bits, noise_level=0.0):
    n         = len(x)
    next_pow2 = 1 << (n - 1).bit_length()
    x_padded  = np.pad(x, (0, next_pow2 - n))

    mean_val  = np.mean(x_padded)
    x_centered = x_padded - mean_val
    x_rotated  = fwht(x_centered)

    x_quantized, scale = quantize_to_bits(x_rotated, bits)

    noise   = np.random.normal(0, noise_level * scale, size=x_quantized.shape)
    x_noisy = x_quantized + noise

    meta = {'scale': scale, 'mean': mean_val, 'n_orig': n, 'n_pad': next_pow2}
    return x_noisy, meta

def fake_dequantize(compressed, meta):
    # fwht는 self-inverse이므로 한 번 더 적용하면 원본 복원
    restored_centered = fwht(compressed)

    restored_padded = restored_centered + meta['mean']
    return restored_padded[:meta['n_orig']]
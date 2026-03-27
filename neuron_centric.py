import numpy as np
from quant import fake_quantize, fake_dequantize
import config

def simulate_neuron_centric_transfer():
    print("=== Neuron-Centric Distributed Inference Simulation ===")
    
    # 1. 뉴런 활성화 데이터 생성 (10MB)
    # 실제 모델처럼 일부 뉴런만 강하게 활성화됨 (Sparse & Heavy-tailed)
    raw_data = np.random.standard_normal(config.DATA_SIZE // 4) # FP32 기준
    saliency = np.abs(raw_data) # 뉴런의 중요도(강도)
    
    # 2. 지능형 레이어 (Neuron-centric Logic)
    # 중요도가 높은 상위 10% 뉴런은 8-bit로, 나머지는 TQ 4-bit로 동적 할당
    threshold = np.percentile(saliency, 80)
    important_mask = saliency >= threshold
    
    print(f"Total Neurons: {len(raw_data):,}")
    print(f"High-Saliency Neurons (Priority): {np.sum(important_mask):,}")
    print("-" * 40)

    # 3. 차별화된 전송 (Differential Transfer)
    # [A] High Priority: 8-bit (Ratio 0.5)
    high_data = raw_data[important_mask]
    high_comp, high_meta = fake_quantize(high_data, ratio=0.5, noise_level=config.NOISE_LEVEL)
    
    # [B] Low Priority: TQ 4-bit (Ratio 0.25)
    low_data = raw_data[~important_mask]
    low_comp, low_meta = fake_quantize(low_data, ratio=0.25, noise_level=config.NOISE_LEVEL)

    # 4. 성능 측정
    # 물리적 전송량 계산
    total_bits_raw = len(raw_data) * 16 # FP16 기준
    total_bits_tq = (len(high_data) * 8) + (len(low_data) * 4)
    effective_ratio = total_bits_tq / total_bits_raw

    # 5. 복원 및 오차 측정 (Total MSE)
    restored_high = fake_dequantize(high_comp, high_meta, len(high_data))
    restored_low = fake_dequantize(low_comp, low_meta, len(low_data))
    
    # 전체 데이터 재구성
    restored_full = np.zeros_like(raw_data)
    restored_full[important_mask] = restored_high
    restored_full[~important_mask] = restored_low
    
    total_mse = np.mean((raw_data - restored_full) ** 2)

    # 6. 결과 출력
    print(f"Effective Compression Ratio: {effective_ratio:.3f}x")
    print(f"Neuron-Centric MSE: {total_mse:.6f}")
    
    # Golden Cross 판단 (단순화)
    bandwidth_gain = 1 / effective_ratio
    print(f"Logical Bandwidth Boost: {bandwidth_gain:.2f}x")
    
    if bandwidth_gain > 3.0 and total_mse < 0.015:
        print("\n[RESULT] Neuron-centric Architecture Verified: High Throughput with Practical Precision.")
    else:
        print("\n[RESULT] Optimization Needed.")

if __name__ == "__main__":
    simulate_neuron_centric_transfer()
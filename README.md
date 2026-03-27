# TurboQuant Simulator (Concept Proof)
> **Solving the Memory Wall: Verifying High-Density Interconnect Protocols for Modern DistBelief**

### 🔗 Related Articles
* **Vol. 1: The Revival of Distributed Computing** [https://edwardyoon.github.io/the-revival-of-distributed-computing/](https://edwardyoon.github.io/the-revival-of-distributed-computing/)
* **Vol. 2: TurboQuant: The High-Density Interconnect Protocol** [https://edwardyoon.github.io/turboquant-the-high-density-interconnect-protocol/](https://edwardyoon.github.io/turboquant-the-high-density-interconnect-protocol/)

---

## 1. Overview
In the era of distributed AI clusters (e.g., NVIDIA Vera Rubin, Google TPU clusters), the primary bottleneck has shifted from raw computation to **Inter-node Boundary Communication**. This project provides a high-fidelity simulation to verify whether a compression-based interconnect protocol—**TurboQuant (TQ)**—can reduce end-to-end latency in distributed inference systems.

The core thesis is that **Communication Density** beats **Memory Capacity**.

## 2. The Golden Cross Hypothesis
The "Golden Cross" is the engineering boundary where the latency gain from reduced data volume outweighs the computational "tax" of real-time quantization.

TurboQuant is beneficial only when:
$$T_{tq} < T_{raw}$$

### 2.1 Theoretical Framework
The total latency for raw data transfer ($T_{raw}$) and TurboQuant-enabled transfer ($T_{tq}$) is defined as follows:

$$T_{raw} = \frac{D}{BW}$$

$$T_{tq} = \frac{D \cdot r}{BW} + \frac{D}{E} + \frac{D}{D_{ec}}$$

Where:
* $D$: Data Size (Total payload in bits)
* $BW$: Interconnect Bandwidth (bps)
* $r$: Compression Ratio (e.g., $r = 0.25$ for 4-bit quantization from 16-bit)
* $E$: Encoding Throughput (Throughput of the hardware-accelerated TQ encoder)
* $D_{ec}$: Decoding Throughput (Throughput of the TQ decoder)

## 3. Key Simulations
This simulator evaluates the feasibility of TurboQuant across three critical dimensions:

1.  **Network Saturation:** How bandwidth constraints (PCIe Gen5 vs NVLink 6/7) affect the Golden Cross.
2.  **Computational Tax:** The required throughput ($E/D_{ec}$) for NPU/TPU systolic arrays to make TQ viable.
3.  **Information Fidelity:** The trade-off between aggressive compression ($r$) and reconstruction error (MSE), ensuring the model's accuracy remains within acceptable bounds.

## 4. Rethinking the Role of HBM in Distributed Inference

This simulation suggests that, under conditions where the **Golden Cross** is achieved via hardware-accelerated quantization, the following architectural shifts may become feasible:

* **Logical Bandwidth Expansion:** Effective bandwidth can increase by $4\times$ to $8\times$, depending on the compression ratio ($r$), without requiring physical interconnect upgrades.
* **Reduced HBM Pressure:** The memory footprint of KV caches and activations can be significantly reduced, lowering the demand for high-capacity HBM.
* **Towards SRAM-Centric Execution:** With sufficiently dense representations, larger subgraphs may remain on-chip, potentially mitigating memory wall effects at inter-node boundaries.

## 5. Usage

### Prerequisites
* Python 3.8+
* NumPy

### Installation
```bash
git clone [https://github.com/edwardyoon/turboquant-sim.git](https://github.com/edwardyoon/turboquant-sim.git)
cd turboquant-sim
pip install -r requirements.txt

### Validated Simulation Results
```

### TurboQuant Simulation

1. **The 1:1:1 Balance:** The simulation shows a perfect equilibrium between Encoding ($0.002s$), Transfer ($0.002s$), and Decoding ($0.002s$). This proves that with a **50GB/s NPU accelerator**, we can neutralize the computational "tax" of quantization and achieve a net speedup.
2. **35% Latency Reduction:** By thinning the data to 4-bit, we achieved a **~35% faster** end-to-end response compared to raw FP16 transfer. This is equivalent to upgrading a physical interconnect without changing a single cable.
3. **Information Density vs. HBM Capacity:** An MSE of **0.03** is a highly practical trade-off for distributed inference. It suggests that "Communication Density" can effectively replace the need for massive HBM capacity, allowing sovereign AI clusters to run on leaner, more cost-effective hardware.

```text
Config: BW=11Gbps, Data=0.10GB, Ratio=0.25 (4-bit)

Raw transfer time: 0.009766 sec
TurboQuant time: 0.006348 sec (Enc: 0.002s, Trans: 0.002s, Dec: 0.002s)
TurboQuant wins? True

Reconstruction MSE: 0.030087
```


# TurboQuant Simulator (Concept Proof)
> **Solving the Memory Wall: Verifying High-Density Interconnect Protocols for Modern DistBelief**

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

## 4. Why This Matters: The End of HBM Dominance
As proven by this simulation, if we can reach the **Golden Cross** via hardware-accelerated quantization:
* **Logical Bandwidth** expands by $4\times$ to $8\times$ without upgrading physical cables.
* **HBM Capacity** demand is halved, as KV caches and activations occupy significantly less memory footprint.
* **SRAM-Centric Execution** becomes possible, keeping massive subgraphs on-chip and bypassing the Memory Wall.

## 5. Usage

### Prerequisites
* Python 3.8+
* NumPy

### Installation
```bash
git clone [https://github.com/edwardyoon/turboquant-sim.git](https://github.com/edwardyoon/turboquant-sim.git)
cd turboquant-sim
pip install -r requirements.txt

# TurboQuant Simulator (Concept Proof)

## Overview

This project demonstrates whether a compression-based interconnect protocol
("TurboQuant") can reduce end-to-end latency in distributed AI systems.

The key hypothesis:

> Compression is beneficial only if total time (encode + transfer + decode)
> is lower than raw transfer time.

---

## Golden Cross Condition

TurboQuant is beneficial when:

T_tq < T_raw

Where:

- T_raw = D / BW
- T_tq = (D * r) / BW + D / E + D / D

D: data size  
BW: bandwidth  
r: compression ratio  
E/D: encode/decode throughput  

---

## What This Simulates

- Network transfer latency
- Compression overhead
- Reconstruction error (approximate)

---

## Run

```bash
python sim.py

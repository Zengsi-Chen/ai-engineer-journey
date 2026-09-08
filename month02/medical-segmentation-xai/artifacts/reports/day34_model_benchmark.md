# Day 34 — U-Net CPU Benchmark

## Experiment Setup

- Device: CPU
- Input size: 256x256
- Input channels: 3
- Output channels: 1
- Batch size: 2
- Warmup iterations: 3
- Benchmark iterations: 10
- Precision: FP32

## Results

| Metric | Baseline U-Net | Small U-Net |
|---|---:|---:|
| Features | (32,64,128,256) | (16,32,64,128) |
| Parameters | 7,763,041 | 1,942,577 |
| Trainable parameters | 7,763,041 | 1,942,577 |
| Parameter memory | 29.61 MB | 7.41 MB |
| Forward time / batch | 1.3974 sec | 0.5683 sec |
| Throughput | 1.43 images/sec | 3.52 images/sec |

## Analysis

The Small U-Net reduces the parameter count by approximately 75%.

CPU forward latency decreases from 1.3974 seconds to 0.5683 seconds per batch, corresponding to approximately a 2.46x speedup.

CPU throughput increases from 1.43 images/sec to 3.52 images/sec, also approximately 2.46x.

Parameter storage decreases from 29.61 MB to 7.41 MB.

## Engineering Decision

The Small U-Net provides a substantially better CPU performance profile and will be useful for rapid development, debugging, loss-function experiments, XAI experiments, and API development.

The Baseline U-Net will be retained as the primary model baseline.

Final model selection will be based on both computational performance and segmentation quality, including Dice and IoU.
\# aidc-khawlah-server



Personal repository for AI Data Center Bootcamp exercises, week 1 through week 7.



\## Week 2, Day 1: Model memory and precision



Environment: Colab, Tesla T4 GPU

Model: Qwen/Qwen2.5-1.5B-Instruct



\### Predicted vs measured memory



| Precision | Predicted GB | Measured GB | Tokens/s |

|-----------|--------------|-------------|----------|

| fp16      | 3.00         | 3.29        | 31.0     |

| int8      | 1.50         | 1.87        | 5.4      |

| int4      | 0.75         | 1.24        | 12.3     |



\### Notes



\- Measured memory is consistently higher than predicted, due to fixed CUDA

&#x20; context and allocator overhead. This overhead matters more for smaller

&#x20; models, since it is a larger share of a smaller total.

\- int8 and int4 use less memory but generate slower than fp16. Quantisation

&#x20; saves memory here, not speed.

\- Files: `artifacts/w2d1/generate.py`, `artifacts/w2d1/results.json`


\# Week 2, Day 1: Model memory and precision



Environment: Colab, Tesla T4 GPU

Model: Qwen/Qwen2.5-1.5B-Instruct



\## Predicted vs measured memory



| Precision | Predicted GB | Measured GB | Tokens/s |

|-----------|--------------|-------------|----------|

| fp16      | 3.00         | 3.29        | 31.0     |

| int8      | 1.50         | 1.87        | 5.4      |

| int4      | 0.75         | 1.24        | 12.3     |



\## Notes



\- Measured memory is consistently higher than predicted, due to fixed CUDA context and allocator overhead. This overhead matters more for smaller models, since it is a larger share of a smaller total.

\- int8 and int4 use less memory but generate slower than fp16. Quantisation saves memory here, not speed.



\## Extra lab: memory budget solver



Given a 16 GB budget, 4 concurrent users, and a 4096-token context, the

largest model that fits is Llama-3.2-3B-Instruct at fp16 (9.80 GB total).

All 15 model/precision combinations in the catalog fit this scenario.



Files: generate.py, results.json, budget\_solution.json


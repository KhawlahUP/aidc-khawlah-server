# Model Lock Record

- Model: Qwen/Qwen2.5-1.5B-Instruct-AWQ
- Quantization: awq
- Flags: --dtype half --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes
- Smoke Test Score: 10/10 (distractor_majority_clean: True)
- Decision reason: AWQ matched fp16's perfect smoke score while running faster (55.7 vs 47.9 tokens/s) and providing more KV-cache headroom (22955 GPU blocks vs fewer under fp16)
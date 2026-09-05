# Structural refactor benchmark

The same Qwen3-0.6B checkpoint and Meson release build (`-O3`, assertions unchanged) were used before and after the refactor. Metal shaders are unchanged. The original executable was saved before edits and benchmarked again immediately before the final executable. Each prompt had one excluded warmup and three measured runs. Values below are medians from that final back-to-back run, recorded in the task's tool output.

| Prompt | Metric | Before | After | Change |
|---|---|---:|---:|---:|
| 23 input tokens | Prefill tokens/s | 484.58 | 475.26 | -1.92% |
| 23 input tokens | Decode tokens/s | 81.43 | 81.19 | -0.29% |
| 23 input tokens | Decode p50 ms | 12.31 | 12.34 | +0.24% |
| 23 input tokens | Decode p95 ms | 12.89 | 13.01 | +0.93% |
| 23 input tokens | Whole-process seconds | 1.170 | 1.173 | +0.24% |
| 389 input tokens | Prefill tokens/s | 1039.05 | 1059.71 | +1.99% |
| 389 input tokens | Decode tokens/s | 77.52 | 77.50 | -0.03% |
| 389 input tokens | Decode p50 ms | 12.81 | 12.81 | 0.00% |
| 389 input tokens | Decode p95 ms | 13.33 | 13.33 | 0.00% |
| 389 input tokens | Whole-process seconds | 1.342 | 1.340 | -0.16% |

Decode and process time are effectively unchanged in this basic benchmark. Short-prompt prefill is 1.9% lower (about 0.9 ms); long-prompt prefill is 2.0% higher. These short runs do not establish statistical significance or cover every workload.

All eight output hashes from each executable matched by prompt, including the warmups. The short prompt generated 61 tokens and the long prompt generated 46 tokens. SHA-256 of CLI stdout:

- Short: `d8c5006481a85a997ba993d892c8653a2bca2c8df4af666b22cbe6f065ecd07c`
- Long: `d5e5d36305d10ab160d16a355bf98c1ad20237f32907cff2ac8f3e89a33fbd78`

All 12 Meson suites pass, including numerical Metal tests. Focused new coverage checks transactional non-paged state and incremental UTF-8 decoding; the ragged-batch test also verifies sparse sample-to-sequence mapping.

Run `scripts/benchmark.py` with a preserved executable and the new build to repeat the comparison. It emits per-run JSON, output hashes, and measured medians. The initial temporary binaries and JSON files were cleared between task turns; this report preserves the recorded final results.

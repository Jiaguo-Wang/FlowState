# WP3B Formal K=4 Paired Result

## Experimental design

- Recurrent-state budget: K=4 checkpoints
- Checkpoint size: 49.125 MiB
- Logical recurrent-state budget: 196.5 MiB
- Paired repetitions: n=5
- Policy order alternates across repetitions
- Each arm uses a fresh engine/cache rebuild

## Validity

- Complete/state-valid arms: 10/10
- Expected-path arms: 10/10

## Policy-level result

- Prompt-LRU TTFT median: 1412.019 ms
- Prompt-LRU TTFT mean: 1415.020 ms
- Workflow-K TTFT median: 38.074 ms
- Workflow-K TTFT mean: 38.887 ms

## Paired effect

- Mean paired TTFT reduction: 1376.133 ms
- Median paired TTFT reduction: 1375.293 ms
- 95% t-CI for mean paired TTFT reduction: [1367.828, 1384.437] ms
- Mean relative TTFT reduction: 97.25%
- Median speedup: 36.97x
- Mean replay/gap reduction: 32768.0 tokens

## Cross-check against WP2

- WP2 predicted saving for 32K recovery: 1504.474 ms
- Formal mean measured paired TTFT saving / prediction: 0.915x

## Per-repetition

| Rep | Prompt TTFT ms | Workflow TTFT ms | Delta ms | Reduction | Speedup | Prompt gap | Workflow gap | Delta gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1407.742 | 38.074 | 1369.668 | 97.30% | 36.97x | 32769 | 1 | 32768 |
| 2 | 1407.204 | 37.311 | 1369.893 | 97.35% | 37.72x | 32769 | 1 | 32768 |
| 3 | 1412.019 | 36.726 | 1375.293 | 97.40% | 38.45x | 32769 | 1 | 32768 |
| 4 | 1423.477 | 42.274 | 1381.203 | 97.03% | 33.67x | 32769 | 1 | 32768 |
| 5 | 1424.657 | 40.051 | 1384.606 | 97.19% | 35.57x | 32769 | 1 | 32768 |

> Interpretation should only be finalized after checking all 10 raw logs and validity gates.

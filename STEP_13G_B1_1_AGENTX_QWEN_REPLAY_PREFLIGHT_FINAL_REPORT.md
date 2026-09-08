# Step 13G-B1.1 AgentX Qwen3.5-9B Replay Fidelity Preflight

**Overall Status:** `AGENTX_QWEN_REPLAY_PREFLIGHT_READY`

## Tokenizer / Model Identity

- Model path: `/home/wjg/models/qwen3.5-9b`
- Tokenizer class: `Qwen2Tokenizer`
- Tokenizer vocab size: `248044`
- Tokenizer length: `248077`
- Config vocab size: `248320`
- Max position embeddings: `262144`
- EOS token id: `248046`
- PAD token id: `248044`
- Safetensors index present: `True`

## Fidelity Gate Results

- Snapshots validated: `23`
- Token range pass: `True`
- Exact length pass: `True`
- 64-token block fidelity pass: `True`
- Context <= 131200 pass: `True`
- Prefix topology pass: `True`
- Total invalid tokens: `0`
- Determinism mismatches: `0`

## SGLang Direct-Token-ID Path

- Present: `True` in `/home/wjg/code/FlowState/evaluation/sota_latency_runtime.py` around line `146`

## Representative Snapshots

- `5ed3de2b572d237576f034af300c1a1f65d3` (t=6101.814, reason=max_pending, pending=8, candidates=17, fork_depth=12)
- `d77fa1e0115f7fca67ca88b95a053d3a41bc` (t=5402.628, reason=max_fork_depth, pending=2, candidates=30, fork_depth=1976)
- `ecd3ea07309dd8a6b62b2f90fd8c60fa92b2` (t=2077.167, reason=max_candidate, pending=4, candidates=149, fork_depth=1511)

## Artifacts

All artifacts frozen under: `/home/wjg/data/agentx/audits/agentx_qwen_replay_preflight_20260905_105318`

- `AGENTX_QWEN_REPLAY_PREFLIGHT_READY`

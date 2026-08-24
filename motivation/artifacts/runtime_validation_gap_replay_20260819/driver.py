"""Minimal runtime validation driver (FlowState motivation, 2026-08-19).

Workload (concurrency=1, sequential):
  S1       : 16384-token head of P32            -> builds shallow ckpt @16384
  R1       : 32768-token P32                    -> builds deep ckpt @32768 (leaf)
  EXT      : P32 + 2048 unique                  -> makes R1's ckpt node INTERNAL
  F00..F23 : HEAD + 2048 unique each            -> mamba pool (16) pressure ->
                                                LRU-evicts deep ckpts (R1, EXT),
                                                refreshes S1 every match,
                                                FA-KV retained
  R2       : same P32                           -> MEASURED: gap replay

Expected at R2: full_kv_hit=32767, exec_prefix=16384, gap=16383,
extend_len=16384, dup_free~=16383, forced track @32512, new ckpt @32768.
"""
from __future__ import annotations

import json
import time
import urllib.request

URL = "http://127.0.0.1:49930/generate"
VOCAB = 248320
N_FILLERS = 24


def toks(seed: int, n: int) -> list[int]:
    return [(seed + i * 7919) % VOCAB for i in range(n)]


def gen(rid: str, ids: list[int], n: int = 8):
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": n, "temperature": 0, "ignore_eos": True},
        "rid": rid,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    print(f"{rid}: {dt:.2f}s prompt={len(ids)} finish={meta.get('finish_reason')}", flush=True)
    return {"rid": rid, "prompt_len": len(ids), "wall_s": round(dt, 3), "meta": meta}


def main():
    P32 = toks(1000, 32768)
    HEAD = P32[:16384]

    log = []
    log.append(gen("S1", HEAD))
    log.append(gen("R1", P32))
    log.append(gen("EXT", P32 + toks(2000, 2048)))
    for k in range(N_FILLERS):
        log.append(gen(f"F{k:02d}", HEAD + toks(3000 + 100 * k, 2048)))
    log.append(gen("R2", P32))

    with open("/tmp/fsval_driver_results.json", "w") as f:
        json.dump(log, f, indent=2)
    print("DRIVER_DONE", flush=True)


if __name__ == "__main__":
    main()

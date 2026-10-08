# LoRA adapters: correctness and cost (2026-10-08)

Engine 0.1.40.2 + `--lora`, one NVIDIA B200 (CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=100`). Model: Qwen4Exp
IQ2_XS pack (`--native`, `--spec 4 --prefill auto --expert-profile data/expert-profile.bin --expert-cache auto`),
with and without the MTP draft layer. Adapter: rank 50, alpha 50 (s = 1), F32, on `ssm_out` / `attn_output` of all 48
layers (80 MiB of VRAM).

## Correctness (`e2e.py`, `lora-e2e-*.out`)

One engine with `--lora` answers each prompt as `lora=1`, `lora=0`, `lora=1`; an engine without an adapter answers
the same prompts. Greedy, 48 tokens, with and without MTP:

- `lora=0` == no adapter, token for token: yes (both prompts, both modes)
- `lora=1` == `lora=1` again (the switch is reversible, the graphs are shared): yes
- `lora=1` != `lora=0`: on the coding prompt (the adapter answers with code at once); a one-line factual answer
  ("The capital of France is Paris.") is the same either way

`lora_parity` (`lora-parity.out`): the kernels against double precision (rel err <= 8e-7 for 1..300 tokens, fp32
and fp16 activations), switched off bitwise unchanged (eager and from a graph captured while on).

## Cost (`bench.py`, `lora-bench-*.out`)

Per setting: one warm-up, then 3 prompt reads (about 2,700 tokens, a different text each, `--prompt-cache 0`) and 3
generations of 256 tokens.

| | decode, MTP | decode, no MTP | prompt read, per 1K tokens |
| --- | --- | --- | --- |
| no adapter | 213.2 tok/s | 118.5 tok/s | 214 ms |
| adapter loaded, `lora=0` | 210.1 tok/s (-1.5%) | 115.6 tok/s (-2.4%) | 216 ms (+1%) |
| adapter loaded, `lora=1` | 202.2 tok/s (-5.2%) | 111.7 tok/s (-5.7%) | 226 ms (+6%) |

The kernels per adapted layer (`lora_parity` timing): 1 token 10.2 us, 4 tokens 12.3 us, 8 tokens 16.4 us,
300 tokens 47.1 us.

## Retest on v0.1.40.4 (`6674a00` + the PR's 5 commits)

Same machine, scripts and settings (`*-0404*.out`). `lora_parity` all ok, with identical kernel timings;
`cvec_parity` passes; the end-to-end checks are identical.

| | decode, MTP | decode, no MTP | prompt read, per 1K tokens |
| --- | --- | --- | --- |
| no adapter | 213.2 tok/s | 118.5 tok/s | 215 ms |
| adapter loaded, `lora=0` | 210.1 tok/s (-1.5%) | 115.0 tok/s (-3.0%) | 215 ms (±0%) |
| adapter loaded, `lora=1` | 202.2 tok/s (-5.2%) | 110.1 tok/s (-7.1%) | 227 ms (+6%) |

## The overlapped up kernel (v0.1.40.4, `perf(lora)` commit)

The small path's second kernel (y += B h) is a programmatic dependent launch: it starts while h = A x runs, loads
its column of B, then waits for h. `lora_parity` (`lora-pdl-parity.out`) all ok and now also times 48 adapted
layers captured in one graph, the way decode runs them:

| per adapted layer | before | overlapped |
| --- | --- | --- |
| 1 token, one call | 10.2 us | 6.2 us |
| 8 tokens, one call | 16.4 us | 13.3 us |
| 1 token, in a 48-layer graph | 6.06 us | 4.74 us |
| 8 tokens, in a 48-layer graph | 13.05 us | 11.48 us |
| 300 tokens (tiled path, unchanged) | 47.1 us | 47.1 us |

A single kernel (the B h blocks spinning on a counter until the A x blocks are done) was tried too: 7.17 us per
layer in the graph, slower, dropped.

End to end (`lora-pdl-bench.out`): the checks above identical in both modes. MTP one benchmark, no MTP the mean of 3:

| | decode, MTP | decode, no MTP | prompt read, per 1K tokens |
| --- | --- | --- | --- |
| no adapter | 213.2 tok/s | 118.2 tok/s | 216 ms |
| adapter loaded, `lora=0` | 210.4 tok/s (-1.3%) | 114.9 tok/s (-2.8%) | 209 ms |
| adapter loaded, `lora=1` | 206.1 tok/s (-3.3%) | 111.7 tok/s (-5.5%) | 227 ms (+5%) |

Each setting reads its own prompt texts, so prompt read per 1K tokens moves by a few percent between settings
without any change in the prompt path.

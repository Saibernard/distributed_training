# Old run — degraded-interconnect node (kept for comparison)

These results are from an **earlier 8x A100 run on a cloud node whose 8 GPUs were
NOT all on a single NVSwitch** (a mixed-topology box). They are kept on purpose
to show that multi-GPU scaling depends on the node's interconnect, not just the
GPU count. The **main, clean results are in `../examples/`** (a proper NVSwitch
node).

## What went wrong here

Everything was identical to the good run — same code, same model, same A100 80GB
GPUs — except the physical wiring between the 8 GPUs on this particular rented
node. At 8 GPUs the all-reduce/all-gather collectives fell to a slow path, so:

| 8-GPU, DDP 1B | this node (degraded) | clean NVSwitch node (`../examples/`) |
|---|---|---|
| throughput | 21,531 tok/s | 86,561 tok/s |
| NCCL comm overhead | 48.6% | 15.7% |
| scaling efficiency | ~20% | ~82% |

Going from 4 to 8 GPUs actually *lowered* throughput here, because ~half of each
step was spent waiting on communication. On the clean node, scaling stayed smooth.

## The lesson

8-GPU scaling efficiency is **interconnect-topology dependent**. Two nodes with
the same 8x A100 80GB spec can behave very differently if one has all 8 GPUs on a
single NVSwitch (fast all-to-all NVLink) and the other splits them across domains
(slower links). This is why `launch/run_all.sh` now runs a fail-fast 8-GPU
interconnect check up front and aborts on a mixed-topology node.

The 1/2/4-GPU numbers here match the clean run; only the 8-GPU points differ.

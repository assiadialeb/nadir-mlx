# Mac-aware model catalog

The **Models → Hugging Face** tab badges each `mlx-community` checkpoint against your local Mac profile (unified RAM and a coarse bandwidth class derived from the CPU brand string).

## Footprint estimate (MLX-127)

Catalog badges use the same **planned TEXT footprint** as LRU eviction planning:

- **Weights** — **total** parameter count × quantization bits × runtime factor (1.15)
- **KV cache** — generic transformer shape by parameter tier × `NADIR_MEMORY_PLANNING_CONTEXT_TOKENS` × parallel sequences cap
- **Prompt cache** — RAM-aware entry count × prefix KV budget

Parameter count prefers Hugging Face `safetensors.total` when the API returns it (`expand=safetensors`); otherwise the name regex (`235B`, `30B`, …) is used.

!!! important "MoE keeps all experts in memory"
    Names like `30B-A3B` or `235B-A22B` advertise **active** experts for generation bandwidth.
    Resident weights still include **every** expert. Nadir sizes memory from the **total**
    parameter count. The active count only boosts the interactive **speed** score
    (sort *Recommended for Mac*), never the fit tier.

!!! warning "Advisory only"
    Badges **never block** download. Peak runtime memory can exceed the estimate when you raise context, concurrency, multimodal inputs, or prompt-cache limits.

## Fit tiers

| Tier | UI copy | Meaning |
|------|---------|---------|
| Comfortable | Recommended for your Mac | Planned footprint ≤ ~55% of (RAM − reserve) |
| Tight | May fit — tight | Planned footprint ≤ `min(~85% of budget, ~75% of unified RAM)` |
| Too large | Likely too large | Above tight threshold |
| Unknown | (no badge) | Missing parameter hint or RAM detection failed |

Reserve defaults to `NADIR_MEMORY_RESERVE_GB` (4 GiB). The 75% unified-RAM ceiling mirrors the typical mlx GPU wired pool so a “tight” badge is less likely to exceed what the Metal allocator can hold.

Planning knobs are documented in the [server config reference](server-config-reference.md) (`NADIR_MEMORY_PLANNING_*`).

RAM and chip class are read once per catalog page from the host (`sysctl hw.memsize` / CPU brand) and cached briefly in-process — not hard-coded to any Mac SKU.

## Recommendation scoring (MLX-130)

Sort **Recommended for Mac** boosts models that:

- Fit comfortably or tightly on this machine
- Are **MoE** (lower active weights → better bandwidth utilization for tok/s)
- Use **4-bit / DWQ-class** quantizations for interactive chat
- Match instruct/chat naming

**TTFT** (time to first token) is driven by cold load, prefill, and thinking — use warm instances, prompt cache, and smaller **active** weights (MoE). **Tok/s** is mostly memory-bandwidth bound; speculative decode / MTP improves generation speed, not first-token latency.

Optional filter **Fits my Mac** hides checkpoints marked *Likely too large*.

## Related

- Instance LRU and footprint: [instance lifecycle](instance-lifecycle.md)
- Gateway concurrency (parallel sequences affect planning): [Nadir Gateway](nadir-gateway.md)

# AGENTS.md

This file captures durable lessons for code agents working in `/home/luke/projects/b12x`.

## Kernel Debugging Lessons

- When debugging a fused kernel, reduce the problem aperture aggressively before changing the main path:
  use one expert, one token, one tile, or a tiny local slice that still exercises the real kernel path.

- For CuTe DSL kernels, keep compile and execution separate.
  If something "hangs", first prove whether it is compile-time or runtime by running the compile step alone.
  CuTe compile is usually fast here; long stalls are often kernel/runtime issues, not codegen.

- Prefer trace parity against the shipped working kernel over naive host algebra when using tiny apertures.
  Tiny high-TP or single-tile harnesses can be badly conditioned numerically, so they are often good for fragment/layout truth but poor as final-output oracles.

- If a fused or debug aperture is supposed to test only the consumer half, reuse the exact runtime objects from the proven launch path whenever possible:
  same packed tensors, same pointer objects, same scale pointers, same alpha/tile metadata objects.
  "Equivalent-looking" rebuilt CuTe views can still drift.

- Treat join-boundary issues as higher-probability than inner-loop math bugs when a copied compute body mostly matches the working kernel source.
  Before rewriting math, verify the exact source tensors, TMA partitions, pointer objects, and metadata handoff at the seam.

- Never fully trust CuTe tensor reads/writes when shared-memory layout is under suspicion.
  To disprove layout hypotheses, use raw pointers, `st.shared` / `ld.shared`, or inline PTX and compare physical addresses/bytes directly.

- For swizzled `sA` / `sSFA` style shared-memory handoffs, use established precedents from existing kernels in this repo.
  These layouts are easy to get subtly wrong; if the formulas and the consumer views are both "copied", the remaining gap is often the coordinate basis or object/view construction.

- `printf` is often the shortest path out of a fused-kernel bug.
  Print on both sides of a write or copy, and narrow to the smallest real piece of work that reproduces the issue.

- If a new kernel variant is already near parity on a real or semi-real harness, stop over-investing in probes and run it through the real end-to-end path.
  Use probes when the real path is unavailable or still failing.

- When a researcher provides a parity-focused rewrite, treat their changes as hypotheses about the seam, not just style differences.
  In particular, parity hardening around `prequantized_input` modes, producer-owned clearing, and authoritative runtime shapes is often more valuable than ad hoc debug surgery.

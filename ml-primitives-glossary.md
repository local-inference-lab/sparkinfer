# ML Primitive Glossary

Shape symbols:

- `T` = active tokens
- `B` = active sequences
- `L` = live sequence length
- `H` = hidden size
- `V` = vocabulary size
- `E` = experts
- `K` = top-k experts
- `I` = MLP/expert intermediate size
- `A` = attention heads
- `G` = KV heads
- `D` = head dimension
- `D_rope` = RoPE/positional feature width
- `D_nope` = noPE/content feature width
- `C` = compressed/latent KV width

| Term | Typical Shape | Meaning / Scaling Notes |
|---|---:|---|
| `Token` | scalar position | One model position. In prefill, `T` can be thousands. In decode, `T ~= B`, usually one new token per live sequence. |
| `Activation` / hidden state | `[T, H]`, e.g. `[T, 4096]` | The per-token vector flowing through layers. Usually BF16/FP16; some kernels quantize internally to FP8/FP4. |
| `Logits` | `[T, V]` | Complete raw score vector over the vocabulary. Softmax turns this into the next-token probability distribution. |
| `Softmax` | `[T, V]`, `[T, E]`, or attention scores | Converts raw scores into normalized probabilities or weights. Used for vocab sampling, routing, and attention. |
| `Linear` / `GEMM` | `[T, H] @ [H, O] -> [T, O]` | Core matrix multiply. Cost scales with `T * H * O`. |
| `Attention` | Q `[T, A, D]`, K/V cache `[B, L, G, D]` | Lets current tokens read previous tokens. Decode attention grows with live sequence length. |
| `Q / K / V` | Q `[T, A, D]`; K/V `[T or L, G, D]` | Query asks what to look for; key identifies stored positions; value is the data mixed into the output. |
| `RoPE` | often a subset like `[T, heads, D_rope]` | Rotary Positional Embedding. It rotates Q/K features based on token position so attention scores include position information. In MLA-style caches, the RoPE part is the positional component. |
| `noPE` | often a subset like `[T or L, heads, D_nope]` | "No positional embedding" features. These Q/K dimensions are not RoPE-rotated, so they carry more content-like matching information. In b12x compressed MLA, noPE cache data is commonly stored as FP8 plus scales. |
| `GQA` | Q `[T, A, D]`, K/V `[B, L, G, D]`, with `G < A` | Grouped-Query Attention: many query heads share fewer KV heads. It keeps multiple query heads while reducing KV cache size and decode read bandwidth. |
| `KV cache` | logical `[B, L, G, D]`; paged `[blocks, block_size, G, D]` | Stores past K/V so decode avoids recomputing old tokens. Long-context decode still has to read/cache-score history. |
| `Live sequence length` | `L` per sequence | Number of cached tokens still visible to a sequence. Main driver of decode attention cost. |
| `Prefill` | `T = total prompt tokens` | Prompt ingestion. Large dense batches, high parallelism, throughput-oriented kernels. |
| `Decode` | `T ~= B` | Autoregressive generation. Small token batches, long KV reads, irregular MoE routing, latency-sensitive kernels. |
| `Paged attention` | KV `[pages, page_size, G, D]` | Serving-friendly KV layout for variable-length sequences. KV may be BF16/FP16 or FP8. |
| `Sparse attention` / `NSA` | selected blocks from `[B, L, G, D]` | Reads selected history blocks instead of all of `L`. More useful as context grows. |
| `MLA` | shared latent/noPE cache `[B, L, C]` plus RoPE cache `[B, L, D_rope]` | Multi-head Latent Attention replaces per-head cached K/V vectors with a single compressed cache vector shared across all attention heads. Each attention head uses learned fixed projection weights to turn that shared vector into that head's effective key/value behavior. |
| `Indexer` / `Top-k` | logits over blocks/tokens, e.g. `[B, candidates]` | Scores cache blocks and selects which ones sparse attention should read. |
| `MoE` | activations `[T, H]`, router logits `[T, E]` | Routes each token to `K` expert MLPs. Cost scales with `T * K * expert_size`, not live sequence length. |
| `Router` | `[T, E] -> ids/weights [T, K]` | Scores experts and selects top-k experts per token. |
| `Expert` | `[tokens_for_expert, H] @ [H, I]`, then `[I, H]` | One MLP inside MoE. Prefill batches well; decode often creates tiny uneven expert batches. |
| `Residual` / norm helper | `[T, H]` | Adds or normalizes activation streams. Usually BF16; fusing reduces memory traffic. |
| `Quantization` | payload plus scales | Stores values in FP8/FP4 with scale metadata. Saves bandwidth/storage; accuracy depends on format and scale granularity. |
| `FP8` | often E4M3 payloads | Used for MXFP8 GEMMs, KV cache, indexer data, and compressed MLA noPE vectors. |
| `FP4` / `NVFP4` | 4-bit payloads plus scales | Used for low-bandwidth GEMM and MoE paths. |
| `W4A16` | weights 4-bit, activations `[T, H]` BF16 | FP4/NVFP4 weights with BF16 activations and inline weight dequantization. |
| `TP sharding` | split tensor dims across GPUs | Tensor parallelism shards big matrices or heads. Requires collectives like all-reduce/all-gather. |
| `EP sharding` | split experts across GPUs | Expert parallelism assigns experts to GPUs. Tokens route to expert owners, usually via all-to-all communication. |

## Mental Models

### Attention Is History Work

Attention is the part of the model that looks backward. In decode, each new token may need to compare against the sequence's existing cache, so cost grows with `L`, the live sequence length. The KV cache avoids recomputing old keys and values, but it does not make history free; it turns recomputation into cache reads and score/mix work.

### GQA Shrinks The KV Side

Standard multi-head attention has one K/V head per query head, so `G = A`. GQA keeps `A` query heads but stores only `G` KV heads, with each KV head shared by a group of query heads. Multi-Query Attention is the extreme case where `G = 1`. This mostly helps decode, because the KV cache and cache reads scale with `G * D` rather than `A * D`.

### MLA Stores One Shared Vector Instead Of Per-Head K/V

Use `A` for attention heads; `H` is the hidden size. In traditional multi-head attention, a model might have `A = 16` attention heads. Each past token stores separate K/V vectors for those heads, roughly K `[B, L, A, D]` and V `[B, L, A, D]`. GQA reduces this by storing only `G` KV heads, but it still stores expanded K/V vectors.

MLA changes what gets cached. As a mental model, it replaces those many per-head K/V vectors with a single compressed cache vector shared across all attention heads, plus a smaller RoPE positional component. Each attention head has learned fixed projection weights, like a per-head decompression matrix, that map from the shared compressed space into that head's effective key/value space.

That is how MLA satisfies the same contract as plain attention. Plain attention needs something it can score against the query, like `score_t = q dot k_t`, and something it can mix after softmax, like `sum_t softmax(score)_t * v_t`. MLA stores less data, but the head-specific learned projections let the kernel recover the effective K/V behavior needed for those two operations.

The decompressed per-head K/V vectors usually do not have to be materialized in memory. Optimized MLA kernels can absorb the decompression projection into the attention math: instead of expanding every cached token into K/V first, they operate directly on the shared compressed cache vector plus the per-head projection weights.

As a concrete dimensionality example, a conventional GQA-style cache with `G = 8` KV heads and `D = 128` head dimension stores both K and V, or `2 * G * D = 2 * 8 * 128 = 2048` scalar cache values per token. A compressed MLA-style cache might store roughly `D_nope = 448` content/noPE values plus `D_rope = 64` positional values, or about `512` scalar cache values per token, plus scale metadata. That is about a 4x reduction in cache element count in this example.

### RoPE And noPE Split Position From Content

RoPE is how many transformer attention layers inject token position into Q/K matching: positions become rotations in feature space, so the dot product depends on where tokens are. noPE is the part that skips that rotation. MLA-style designs can cache and process those parts differently: the noPE portion is usually larger and more compressible, while the RoPE portion preserves the positional signal needed for attention scoring.

### GEMM And MoE Are Token Transformation Work

Dense linear layers and MoE layers transform the current token activations. Their cost mostly grows with `T`, `H`, `I`, and, for MoE, `K`. They do not directly care how long the conversation is. A decode token with 100K cached history still has roughly the same MLP/MoE work as a decode token with 1K cached history.

### Prefill Is Throughput, Decode Is Latency

Prefill has many prompt tokens, so kernels can run large dense tiles and keep the GPU busy. Decode usually has one new token per sequence, so batches are smaller, cache reads are longer, and routing is more irregular. That is why the same model operation often needs different prefill and decode kernels.

### Sparse Attention Trades Selection For Fewer Reads

Sparse attention adds an indexer/top-k step to decide which history blocks matter. That extra step only makes sense if it avoids enough KV reads afterward. The longer `L` gets, the more attractive this trade becomes.

### MoE Is Sparse In Parameters, Not Time History

MoE models may have many experts, but each token only activates a few. That increases parameter capacity without running every expert for every token. The hard serving problem is routing tokens efficiently, especially in decode when expert batches are small and uneven.

### TP Splits Math, EP Splits Experts

TP shards tensor dimensions: columns, rows, or attention heads. It reduces per-GPU matrix size but introduces collectives. EP shards the expert set: different GPUs own different experts. It scales MoE capacity, but introduces token routing and load-balancing problems.

### Quantization Mostly Buys Bandwidth

FP8/FP4 reduce memory movement and storage. Scales recover useful numeric range, and kernels often dequantize inline. Weight quantization helps when weights dominate bandwidth; activation/KV quantization helps when runtime tensors or long-context cache reads dominate.

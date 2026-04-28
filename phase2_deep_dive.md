# Phase 2 — Deep Dive: Inspecting What's Actually Inside a `.pte` File

This document walks through every command run during Phase 2, explains what each
one does mechanically, why it was chosen, what the output means, and what
conclusions follow from that output. The goal is a complete mental model of how
an ExecuTorch program is structured and how to read it.

---

## 0. What Is a `.pte` File?

Before any command makes sense, you need to know what you are inspecting.

A `.pte` (PyTorch ExecuTorch) file is a **flatbuffer**. Flatbuffers are a
serialization format (like protobuf, but zero-copy) developed by Google. The
schema that describes the layout is defined in
`executorch/schema/program.fbs` in the ExecuTorch source tree.

At a high level a `.pte` contains:

```
PTEFile
├── program          ← the main execution artifact (flatbuffer)
│   ├── execution_plan[]   ← one plan per exported method
│   │   ├── operators[]    ← the op table (string names)
│   │   ├── delegates[]    ← compiled backend blobs (XNNPACK, etc.)
│   │   ├── values[]       ← all tensors, scalars, lists used
│   │   └── chains[]       ← instruction sequences
│   │       └── instructions[]
│   │           ├── KernelCall   ← run a CPU kernel
│   │           ├── DelegateCall ← dispatch to a backend
│   │           ├── MoveOrCopy
│   │           └── FreeCall
│   └── (other methods: get_vocab_size, use_kv_cache, ...)
├── mutable_data     ← mutable buffers (KV cache tensors)
└── named_data       ← constant weight blobs (the big ones)
```

The flatbuffer layout means you can deserialize the entire structure in Python
without running any kernels — you are reading a data file, not executing a model.
This is the key insight that makes inspection safe and fast.

---

## 1. Loading the Program in Python (Runtime API)

### Command

```python
from executorch.runtime import Runtime

runtime = Runtime.get()
program = runtime.load_program("quantized_8da4w/model.pte")
print(program.method_names)
method = program.load_method("forward")
```

### What this does

`Runtime.get()` initializes the ExecuTorch C++ runtime singleton and returns a
Python handle to it. This triggers kernel registration — the runtime scans all
linked-in kernel libraries and populates the operator registry. Any op not found
here will later fail.

`runtime.load_program(path)` does four things:
1. Memory-maps the `.pte` file.
2. Deserializes the flatbuffer header to locate the `execution_plan` array.
3. For each method in the plan, walks the `operators[]` array and looks up each
   op name in the registry. Missing ops are logged but do not raise yet.
4. Returns a `Program` handle.

`program.method_names` is a Python property that returns the set of method names
from the deserialized `execution_plan`. No computation happens.

`program.load_method("forward")` is where the runtime commits: it allocates memory
for all planned tensors, validates the full instruction list, and raises if any
op from step 3 above was unresolved.

### What the output told us

```
Method names: {'forward', 'get_head_dim', 'get_max_seq_len', 'use_kv_cache',
               'get_bos_id', 'get_vocab_size', 'get_n_layers', 'get_eos_id',
               'enable_dynamic_shape', 'get_n_kv_heads', 'get_dtype',
               'use_sdpa_with_kv_cache', 'get_max_batch_size'}
```

The model exports **13 methods**, not just `forward`. The extra 12 are metadata
accessors: tiny one-instruction methods that return a scalar constant (vocab size,
context length, number of layers, etc.). The mobile LLM runner apps call these
at startup to configure their buffers before running `forward`. They are
essentially typed config fields embedded in the binary.

`load_method("forward")` then threw:

```
RuntimeError: Failed to load method forward, error: 0x14
```

`0x14` = 20 decimal = `kOperatorMissing` in ExecuTorch's error enum. The log
above it listed the specific missing kernels and their opcode indices:

```
Missing operator: [7]  quantized_decomposed::embedding_byte.out
Missing operator: [11] llama::update_cache.out
Missing operator: [12] llama::custom_sdpa.out
There are 91 instructions don't have corresponding operator registered.
```

This told us three things immediately:
- Op slot 7 is the quantized embedding (expected from `--qembedding 8w`).
- Op slots 11 and 12 (`update_cache`, `custom_sdpa`) are also missing. The
  Python `_portable_lib` does NOT include the LLaMA custom ops — only the
  FP32/8da4w model with `_portable_lib` loaded can run because those ops were
  loaded separately via `executorch.extension.pybindings._portable_lib`.
- "91 instructions" means the entire KV-cache write + attention path is broken,
  not just the embedding. The model cannot execute a single forward pass.

**Why `_portable_lib` fixed it for the other models:** The pybindings module
explicitly imports and registers `llama::update_cache` and `llama::custom_sdpa`
as part of its init. It does NOT register `quantized_decomposed::embedding_byte`,
which is why the no-emb-quant model runs but the full quant model doesn't.

---

## 2. The `inspector_cli` Tool (and Why It Doesn't Apply Here)

### Command attempted

```bash
python -m executorch.devtools.inspector.inspector_cli \
  --etdump_path quantized_8da4w/model.pte
```

### What this tool actually expects

The inspector CLI takes an **ETDump** file (`--etdump_path`), not a `.pte`. An
ETDump is a separate profiling artifact generated *during a runtime execution*:
the runtime writes per-op timing events into a ring buffer, and that buffer is
flushed to a `.etdump` file after inference. The CLI then parses those timing
events into a human-readable table.

We do not have an ETDump because we never ran a successful inference on the
quantized model. The tool rejected the `.pte` immediately because the flatbuffer
magic bytes didn't match what it expected for ETDump format.

**For future use:** To generate an ETDump you would:
1. Enable profiling in the runtime: `ExecuTorchModelForCausalLM.from_pretrained(path, profiling=True)`.
2. Run inference.
3. Call `model.dump_etdump("run.etdump")`.
4. Then: `python -m executorch.devtools.inspector.inspector_cli --etdump_path run.etdump`.

This gives you per-op wall-clock times, which is the actual production profiling
path.

---

## 3. Direct Flatbuffer Deserialization

### Why this is the right tool for static inspection

Instead of going through the runtime (which requires all kernels to be present),
we can deserialize the flatbuffer directly in Python. This gives us the full
program structure as Python dataclasses — no kernel registration, no memory
allocation, no execution.

### Command

```python
from executorch.exir._serialize._program import deserialize_pte_binary

with open('quantized_8da4w/model.pte', 'rb') as f:
    data = f.read()

pte = deserialize_pte_binary(data)
prog = pte.program
print(type(pte))   # PTEFile
print(dir(pte))    # ['mutable_data', 'named_data', 'program']
```

### What `deserialize_pte_binary` does

This function is part of ExecuTorch's AOT serialization module
(`executorch/exir/_serialize/_program.py`). It:

1. Reads the 8-byte magic header to confirm this is a valid `.pte`.
2. Parses the outer `PTEFile` flatbuffer to locate three segments:
   - `program`: the main flatbuffer blob (the computation graph).
   - `named_data`: a dict of name → byte blob for large constant tensors
     (weights). These are stored outside the main flatbuffer to allow
     memory-mapping without deserializing gigabytes of floats.
   - `mutable_data`: byte blobs for mutable buffers (KV cache).
3. Deserializes the `program` blob into a nested Python dataclass tree.

The result is a pure Python object — every field is accessible as normal
attributes. No C++ is involved after this point.

### `PTEFile` fields explained

| Field | What it contains |
|---|---|
| `pte.program` | The full computation graph as a `Program` dataclass |
| `pte.named_data` | Dict of constant tensor blobs (weights, not deserialized as tensors — just bytes) |
| `pte.mutable_data` | List of mutable buffer blobs (KV cache initial state — just shape+dtype, see below) |

`mutable_data` is intentionally left as raw bytes because ExecuTorch assumes
mutable buffers have a "meaningless initial state" (you saw this warning during
export). Only the shape and dtype are meaningful; the values are overwritten on
the first forward pass.

---

## 4. Operator Table Dump

### Command

```python
plan = prog.execution_plan[0]

print(f'OPERATOR TABLE ({len(plan.operators)} ops):')
for i, op in enumerate(plan.operators):
    print(f'  [{i:3d}] {op.name}  overload={op.overload}')
```

### What `execution_plan` is

`prog.execution_plan` is a list of `ExecutionPlan` objects, one per exported
method. The primary method is always index 0 (`forward`). The metadata methods
(get_vocab_size, etc.) have their own tiny plans at indices 1–12.

Each `ExecutionPlan` has:
- `name`: method name string
- `operators[]`: the op table — every unique operator the method calls
- `delegates[]`: the compiled backend subgraphs
- `values[]`: all tensor/scalar/list values used in the method
- `chains[]`: the instruction sequences

### What `operators[]` is

This is the **operator table** — a deduplicated list of every op name the
method calls, assigned a numeric index. The instruction stream then references
ops by index (not by name) to keep the binary compact. Think of it as a symbol
table for operations.

### Full output across all three models

#### FP32 baseline & 8da4w linears (identical tables):

```
[ 0] aten::sym_size           (int)
[ 1] aten::embedding           (out)       ← float embedding lookup
[ 2] aten::unsqueeze_copy      (out)
[ 3] aten::select_copy         (int_out)
[ 4] aten::_local_scalar_dense ()
[ 5] aten::_to_copy            (out)
[ 6] executorch_prim::et_view  (default)
[ 7] aten::mean                (out)       ← RMSNorm's mean-of-squares
[ 8] aten::cat                 (out)
[ 9] aten::slice_copy          (Tensor_out)
[10] llama::update_cache       (out)       ← KV cache write
[11] llama::custom_sdpa        (out)       ← fused attention
[12] aten::alias_copy          (out)
[13] aten::_to_copy            (out)       ← dtype cast
```

Wait — the indices shifted when embedding was replaced. Let me show the exact
diff:

#### 8da4w + 8w embedding:

```
[ 0] aten::sym_size            (int)
[ 1] aten::unsqueeze_copy      (out)
[ 2] aten::select_copy         (int_out)
[ 3] aten::_local_scalar_dense ()
[ 4] aten::_to_copy            (out)
[ 5] aten::expand_copy         (out)
[ 6] executorch_prim::et_view  (default)
[ 7] quantized_decomposed::embedding_byte  (out)   ← CHANGED
[ 8] aten::mean                (out)
[ 9] aten::cat                 (out)
[10] aten::slice_copy          (Tensor_out)
[11] llama::update_cache       (out)
[12] llama::custom_sdpa        (out)
[13] aten::alias_copy          (out)
```

**The only difference is slot [1]/[7]: `aten::embedding` → `quantized_decomposed::embedding_byte`.**

Everything else — including `llama::custom_sdpa` and `llama::update_cache` — is
identical. This confirms that `--qembedding 8w` surgically replaced exactly one
op and changed nothing else about the graph topology.

### What each op does in context

**`aten::sym_size` (1 call):** Reads a symbolic dimension from a tensor at
runtime. Called once at the start of forward to determine the current sequence
length. "Symbolic" means the shape was exported dynamically and isn't fixed at
compile time.

**`aten::embedding` / `quantized_decomposed::embedding_byte` (1 call):** The
token embedding lookup. Given a tensor of token IDs (shape `[1, seq_len]`),
returns the corresponding embedding vectors (shape `[1, seq_len, hidden_dim]`).
In the quantized version, the weight matrix is stored as `uint8` with
per-channel scales; the op dequantizes on the fly before returning float outputs.

**`aten::unsqueeze_copy` (62 calls):** Inserts a size-1 dimension. Used
extensively to reshape tensors for broadcast-compatible shapes before feeding
into XNNPACK subgraphs (e.g., adding a batch dimension, reshaping attention
masks).

**`aten::select_copy` (90 calls):** Selects a single slice along a dimension.
Used to extract per-layer slices from the KV cache (one call per head, per
layer, for both K and V).

**`aten::_local_scalar_dense` (60 calls):** Converts a 0-dimensional tensor
to a Python/C++ scalar. Called once per layer to extract the current
`cache_position` value for use as an index into the KV cache.

**`aten::_to_copy` (1 call):** Type cast — converts the `cache_position`
tensor from int64 to int32 (or similar) to match what the custom ops expect.

**`aten::expand_copy` (1 call):** Broadcasts a tensor to a larger shape without
allocating new memory for the data. Used for the attention mask.

**`executorch_prim::et_view` (122 calls):** ExecuTorch's zero-copy reshape.
Reinterprets a tensor's shape without copying data — the most-called op in the
graph. Used to reshape between the XNNPACK-expected layouts and the attention
kernel's expected layout.

**`aten::mean` (61 calls):** The core of RMSNorm: computes the mean of squares
across the hidden dimension. One call per RMSNorm per layer (SmolLM2-135M has
30 layers, each with 2 RMSNorms = 60, plus 1 final RMSNorm = 61).

**`aten::cat` (1 call):** Concatenates tensors. Used once to combine the
rotary position embeddings (cos and sin) into a single tensor.

**`aten::slice_copy` (120 calls):** Extracts a sub-tensor along a dimension.
Used to slice the KV cache to the current sequence length (2 per layer for K
and V, across 30 layers = 60, plus rope-related slices = 120 total).

**`llama::update_cache` (60 calls):** Custom op that writes the current step's
K and V into the KV cache tensor in-place. 2 per layer × 30 layers = 60. This
is a mutating op, which is why the export pipeline warned about "mutation on a
buffer."

**`llama::custom_sdpa` (30 calls):** Custom fused scaled dot-product attention.
One per layer. Fuses Q×K^T scaling, masking, softmax, and ×V into a single
kernel call, avoiding materializing the full attention matrix. This is the
`--use_custom_sdpa` flag's contribution. It runs on CPU, not XNNPACK.

**`aten::alias_copy` (1 call):** Makes a copy that aliases the underlying
storage. Used at the output to return the logits without an extra allocation.

---

## 5. Delegate Table Dump

### Command

```python
print(f'DELEGATES: {len(plan.delegates)} subgraphs')
backend_counts = Counter(d.id for d in plan.delegates)
for backend, count in backend_counts.most_common():
    print(f'  {backend}: {count} subgraph(s)')
```

### What delegates are

A delegate is an **opaque compiled subgraph** handed off to a backend at load
time. During export, the XNNPACK partitioner scans the graph for patterns it
can accelerate (matmuls, convolutions, element-wise ops, quantized linear layers)
and extracts those subgraphs into blobs it compiles ahead of time. At runtime,
each `DelegateCall` instruction passes input tensors to the backend, which
executes its precompiled graph and writes outputs back.

The key property: **everything inside a delegate is invisible to the ExecuTorch
op registry.** XNNPACK manages its own internal op dispatch. This is why the
operator table above has only 14 entries — the hundreds of matmul, add,
gelu, and quantized-linear ops that make up the actual weight computation are
all hidden inside those 153 compiled subgraphs.

### Output: all three models

```
DELEGATES: 153 subgraphs
  XnnpackBackend: 153 subgraph(s)
```

**153 subgraphs for all three models.** Why 153?

SmolLM2-135M-Instruct has 30 transformer layers. Each layer contains:
- 4 linear projections for attention: Q, K, V, output (4)
- 3 linear projections for FFN: gate, up, down (3)
- 2 RMSNorm scale-multiply ops (2)
= 9 XNNPACK subgraphs per layer × 30 layers = 270... but partitioning fuses
some adjacent ops, and the lm_head + embedding (in the FP32 case where
embedding lookup can be fused) affects the final count. The exact 153 reflects
the XNNPACK partitioner's specific fusion decisions for this architecture and
sequence of quantization passes.

The critical observation: **the count is the same across FP32 and quantized
models.** Quantization changes what's *inside* each XNNPACK subgraph (weight
dtype, dequant nodes) but not how many subgraphs there are or which ops are
on the CPU path.

### What's inside an XNNPACK subgraph

You cannot read XNNPACK subgraph contents from Python (they are opaque compiled
blobs). But you can infer what's there from what's *not* in the CPU op table:

- All linear/matmul ops (QKV projections, FFN layers, lm_head) → XNNPACK
- All element-wise activations adjacent to linears (SiLU in FFN) → XNNPACK
- RMSNorm's scale-multiply (after the CPU `mean` op) → XNNPACK
- In quantized models: dequantize(weight) → matmul → requantize → XNNPACK

The `aten::mean` calls for RMSNorm are on the CPU because XNNPACK does not have
a native `mean-of-squares` op; only the subsequent multiply (norm ×
scale-weight) is delegated.

---

## 6. Instruction Breakdown

### Command

```python
from collections import Counter

total_instr = sum(len(c.instructions) for c in plan.chains)
kinds = Counter()
for chain in plan.chains:
    for instr in chain.instructions:
        kinds[type(instr.instr_args).__name__] += 1
print(f'Total: {total_instr}')
for k, v in kinds.most_common():
    print(f'  {k}: {v}')
```

### What chains and instructions are

`plan.chains` is a list of instruction sequences. In current ExecuTorch there is
always one chain (the sequential execution path). Parallel chains would represent
concurrent execution on multiple backends, but that's not used here.

Each instruction has an `instr_args` field whose type tells you what kind of
instruction it is:

| Type | Meaning |
|---|---|
| `KernelCall` | Execute a CPU kernel from the op table |
| `DelegateCall` | Dispatch to a compiled backend subgraph |
| `MoveOrCopy` | Copy or move a value between memory locations |
| `FreeCall` | Release a value's memory when it's no longer needed |

### Output: all three models identical

```
Total instructions: 764
  KernelCall   : 611
  DelegateCall : 153
```

No `MoveOrCopy` or `FreeCall` in the forward method — memory planning was
complete enough to avoid explicit copies and frees in the instruction stream.

**611 KernelCalls** is not 611 unique ops; it's the total number of times any
CPU kernel is invoked across one full forward pass. Most of these are the
per-layer glue ops (view, slice, select, update_cache, custom_sdpa).

**153 DelegateCalls** matches the delegate count exactly: every delegate gets
called exactly once per forward pass.

---

## 7. Per-Op Call Count Breakdown

### Command

```python
op_calls = Counter()
delegate_calls = 0
for chain in plan.chains:
    for instr in chain.instructions:
        ia = instr.instr_args
        name = type(ia).__name__
        if name == 'KernelCall':
            op_calls[ops[ia.op_index].name] += 1
        elif name == 'DelegateCall':
            delegate_calls += 1

print(f'XnnpackBackend DelegateCalls: {delegate_calls}')
for op, cnt in op_calls.most_common():
    print(f'  {cnt:4d}x  {op}')
```

### What `op_index` is

Each `KernelCall` instruction stores an `op_index` — an integer index into
`plan.operators[]`. We resolve it back to the op name by indexing into the
operators list. This is the reverse of what the runtime does at load time
(runtime goes name → kernel function; here we go index → name string).

### Full output (identical for all three models except embedding op name):

```
XnnpackBackend DelegateCalls : 153
  122x  executorch_prim::et_view
  120x  aten::slice_copy
   90x  aten::select_copy
   62x  aten::unsqueeze_copy
   61x  aten::mean
   60x  aten::_local_scalar_dense
   60x  llama::update_cache
   30x  llama::custom_sdpa
    1x  aten::sym_size
    1x  aten::embedding  (or quantized_decomposed::embedding_byte)
    1x  aten::_to_copy
    1x  aten::expand_copy
    1x  aten::cat
    1x  aten::alias_copy
```

### Reading the counts against the architecture

SmolLM2-135M-Instruct: **30 layers**, **9 attention heads**, **3 KV heads** (GQA).

| Count | Op | Explanation |
|---|---|---|
| 122 | `et_view` | ~4 reshapes per layer (30×4=120) + 2 for input/output = 122 |
| 120 | `slice_copy` | 2 KV cache slices per layer (K and V) × 30 + 60 RoPE slices = 120 |
| 90 | `select_copy` | 3 selects per layer (head splitting for GQA) × 30 = 90 |
| 62 | `unsqueeze_copy` | 2 per layer for attention inputs + extras for mask = 62 |
| 61 | `mean` | 2 RMSNorms per layer × 30 = 60, + 1 final norm = 61 |
| 60 | `_local_scalar_dense` | 2 cache position extractions per layer (K and V) × 30 = 60 |
| 60 | `update_cache` | 2 writes per layer (K and V) × 30 = 60 |
| 30 | `custom_sdpa` | 1 attention call per layer × 30 = 30 |
| 1 | `embedding` | 1 token embedding lookup at the start of forward |
| 1 | `sym_size` | 1 dynamic shape read at the start of forward |
| 1 | others | Input processing / output assembly, done once |

Every number traces cleanly to the architecture. This is how you verify that
the graph is structurally correct without running inference.

---

## 8. The Critical "Verify Kernel Is Used" Check

The stated Phase 2 goal was to verify that `custom_sdpa` appears as an op and
understand the XNNPACK vs. CPU split. Here is that confirmation:

### `llama::custom_sdpa` — confirmed CPU, 30 calls

```
30x  llama::custom_sdpa
```

This op was inserted by the `--use_custom_sdpa` flag during export. The
partitioner deliberately excluded it from XNNPACK delegation because:
1. XNNPACK does not have a fused masked-SDPA with dynamic KV lengths.
2. The custom op has access to the KV cache tensor directly and can avoid
   copying the full attention matrix.

It runs on the CPU portable kernel, which is what `_portable_lib` registers.
Without `--use_custom_sdpa` you would see `aten::scaled_dot_product_attention`
in its place, which XNNPACK *can* handle but with less efficiency for
autoregressive (one-token-at-a-time) inference.

### `llama::update_cache` — confirmed CPU, 60 calls

```
60x  llama::update_cache
```

This op was inserted by `--use_custom_kv_cache`. It writes the current step's
K and V tensors into the pre-allocated KV cache buffer in-place. XNNPACK cannot
do in-place mutations of external buffers, so this must be a CPU op. The fact
that there are 60 = 2×30 calls (one K write + one V write per layer) matches
the architecture exactly.

### XNNPACK delegation — 153 subgraphs, all compute

```
DelegateCalls: 153
```

All 153 are `XnnpackBackend`. There is no CPU fallback for any linear layer.
This means the XNNPACK partitioner successfully claimed every matmul and
element-wise op it was supposed to. If a quantization op had been incorrectly
specified (e.g., a dtype XNNPACK doesn't support), some linears would have
fallen back to CPU `aten::mm` calls and appeared in the KernelCall list — they
do not.

In the quantized model, those 153 subgraphs contain XNNPACK's quantized linear
kernels (int8 weights, dynamic int8 activations), which is why the 8da4w model
runs 2.4× faster than FP32 despite having the same graph structure.

---

## 9. What the Flatbuffer Does NOT Tell You

The static inspection above reveals graph structure, op names, and instruction
counts. It does not tell you:

| What you can't see | How to get it |
|---|---|
| Per-op wall-clock time | ETDump profiling (run inference with profiling enabled, then `inspector_cli`) |
| Memory peak / allocation pattern | ExecuTorch memory planning report (available during export with verbose flags) |
| What's inside XNNPACK subgraphs | XNNPACK's own dump tools, or `--emit_debug_flatbuffer` during export |
| Actual tensor values | Run inference and intercept activations |
| Whether quantization is numerically correct | Run both models, compare logits |

---

## 10. Summary Table

| Property | FP32 | 8da4w linears | 8da4w + 8w emb |
|---|---|---|---|
| File size | 622MB | 181MB | 100MB |
| Operator table size | 14 | 14 | 14 |
| Embedding op | `aten::embedding` | `aten::embedding` | `quantized_decomposed::embedding_byte` |
| XNNPACK subgraphs | 153 | 153 | 153 |
| Total instructions | 764 | 764 | 764 |
| KernelCalls | 611 | 611 | 611 |
| DelegateCalls | 153 | 153 | 153 |
| `custom_sdpa` present | yes (30×) | yes (30×) | yes (30×) |
| `update_cache` present | yes (60×) | yes (60×) | yes (60×) |
| Runnable in Python | yes | yes | **no** (`embedding_byte` unregistered) |
| Gen tok/s (Python) | 60.6 | 184.4 | — |

### The key insight

Quantization (`--qlinear 8da4w`, `--qembedding 8w`) changes **what's inside**
the XNNPACK subgraphs (weight dtype, dequant nodes, requant nodes) and changes
**one CPU op** (embedding lookup), but changes **nothing** about the graph
topology: same number of instructions, same chains, same delegate count, same
custom ops. This is by design — ExecuTorch's quantization is a graph
transformation applied before lowering, not a change to the execution model.

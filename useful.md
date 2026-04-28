# Why `quantized_decomposed::embedding_byte.out` Fails in Python

## The Short Answer

When you export a model with `--qembedding 8w`, ExecuTorch compiles the embedding
lookup into a quantized op called `quantized_decomposed::embedding_byte.out`. That op
is not registered in the Python-side ExecuTorch runtime. The `.pte` file loads fine,
but the moment you call `forward()` the runtime cannot find the kernel and returns
error `0x14` (`kOperatorMissing`).

---

## Background: How ExecuTorch Executes a `.pte` File

A `.pte` file is a self-contained flatbuffer that encodes:

1. The **computation graph** — a list of instructions (operator calls, memory moves, etc.)
2. **Tensor metadata** — shapes, dtypes, memory plans
3. **Constant data** — weights serialized as blobs

At load time ExecuTorch walks the instruction list and, for every operator call,
looks up a matching kernel in its **operator registry**. The registry is a static
map built at compile time from the set of kernels that were linked into the binary
(or, in Python, dynamically loaded into the process).

If even one kernel is missing the method is marked broken (`kOperatorMissing`) and
any attempt to run it returns `0x14`. There is no fallback or lazy-loading — every
op must resolve before the first token is generated.

---

## Two Separate Registries: AOT vs. Runtime

ExecuTorch has a hard split between the **export path** and the **runtime path**.

### Export path (AOT)
`optimum-cli export executorch` runs entirely in Python using PyTorch's
`torch.export` and ExecuTorch's ahead-of-time (AOT) lowering stack. The quantized
embedding op is registered here through `torchao` / `torch.ops.quantized_decomposed`.
This is why export succeeds — the AOT stack knows exactly what `embedding_byte` means
and can serialize it into the flatbuffer.

### Runtime path (inference)
The Python runtime used by `ExecuTorchModelForCausalLM` is the `_portable_lib`
pybinding (`executorch/extension/pybindings/_portable_lib.so`). This is a thin
Python wrapper around ExecuTorch's portable kernel library. The portable library
ships a curated set of kernels focused on correctness and broad coverage of
standard floating-point and integer ops. It does **not** include
`quantized_decomposed` kernels — those live in a separate shared library:
`executorch/kernels/quantized/libquantized_ops_aot_lib.so`.

The design intent is that `libquantized_ops_aot_lib.so` is meant for AOT use
(running quantized graphs inside the Python export pipeline for verification),
not for being bundled into shipping runtimes where size and portability matter.
The actual production kernels for embedded/mobile targets are compiled into the
platform-specific runtime (the Android `.aar` or the iOS `.xcframework`).

---

## Why Loading `libquantized_ops_aot_lib.so` Also Fails

Even if you try to manually load `libquantized_ops_aot_lib.so` at runtime, you
hit an ABI mismatch:

```
OSError: undefined symbol: _ZNK2at10TensorBase14const_data_ptrIfLi0EEEPKT_v
```

This mangled symbol is `at::TensorBase::const_data_ptr<float>()` — a method on
PyTorch's core tensor class. The ExecuTorch pip wheel ships a version of
`libquantized_ops_aot_lib.so` that was compiled against a specific PyTorch
nightly. In this environment that wheel was built against a different nightly than
the one installed (`torch 2.12.0.dev20260317+cpu`). Because PyTorch's C++ ABI
changes frequently between nightlies, the symbol layout no longer matches and the
dynamic linker refuses to load the library.

This is not a bug that can be patched in Python — it requires rebuilding
`libquantized_ops_aot_lib.so` from source against the exact installed torch
version.

---

## Why the Native Runtime Works Fine

On Android and iOS the entire stack is compiled together from source:

- `libexecutorch.a` (core runtime)
- `libquantized_ops_lib.a` (quantized kernels, the *runtime* variant)
- `libxnnpack_backend.a` (XNNPACK delegate)

All three are compiled with the same toolchain, against the same headers, and
linked into a single binary. There is no dynamic symbol resolution at inference
time — the kernel table is resolved at link time. `embedding_byte.out` is
included because it is explicitly listed in the ops registration for quantized
builds (`CMakeLists.txt` in `kernels/quantized/`).

This is also why ExecuTorch prints the disclaimer at the top of every Python
inference run:

> Python-based perf measurements are approximate … For end-to-end,
> platform-accurate benchmarks, please use the official ExecuTorch apps.

---

## The `_portable_lib` Import Does Not Help

Importing `executorch.extension.pybindings._portable_lib` before loading the
model does register a number of custom ops (custom SDPA, custom KV cache update,
etc.) that the LLM recipe needs. That is why the FP32 and `8da4w`-linears-only
models work after that import. But `_portable_lib` is intentionally built without
`quantized_decomposed` ops — they were never part of its scope.

---

## Concrete Operator Chain for Embedding Quantization

When you pass `--qembedding 8w` the export pipeline:

1. Replaces `torch.nn.Embedding` with a quantized version that stores weights as
   `uint8` (8-bit) with a per-channel or per-tensor float scale + zero-point.
2. The forward op becomes `quantized_decomposed::embedding_byte(weight_uint8,
   weight_scale, weight_zero_point, indices) -> float_tensor`.
3. This op is serialized into the `.pte` instruction stream as opcode `7`
   (as seen in the `operator_registry.cpp` log: `Missing operator: [7]`).
4. At load time the runtime scans opcode 7, finds no matching kernel, marks the
   method broken, and `forward()` throws `0x14`.

---

## Workarounds

| Goal | Approach |
|------|----------|
| Benchmark all configs in Python | Re-export without `--qembedding` (`--qlinear 8da4w` only). Linears dominate latency anyway. |
| Test the full 100MB model | Use the ExecuTorch Android benchmark app or build a local runner from source with `-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON`. |
| Fix the ABI mismatch | Build ExecuTorch from source pinned to the exact installed torch nightly and reinstall. |
| Register the op in Python | Open a feature request upstream — `_portable_lib` could optionally link the quantized kernel set for testing purposes. |

---

## Summary

The failure is the result of three deliberate architectural decisions in ExecuTorch:

1. **Ops are statically registered** — no runtime discovery, no lazy loading.
2. **The portable Python runtime is intentionally minimal** — quantized kernels
   are a separate library meant for AOT verification and native-device use, not
   for Python inference.
3. **The AOT `.so` is pinned to a specific torch ABI** — loading it against a
   different torch nightly causes undefined symbol errors.

The 100MB model (`8da4w` linears + `8w` embeddings) is perfectly valid and will
run correctly on Android/iOS. The limitation is entirely in the Python evaluation
environment.

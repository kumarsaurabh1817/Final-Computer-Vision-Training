# Day 2 Notes — Export to ONNX and Accelerate

## Core ideas

**ONNX** is a framework-neutral graph format: operators (Conv, Relu, Gemm, …) plus weights. Export once, run with any ONNX runtime, with no PyTorch needed at serving time.

**Export:** `torch.onnx.export(model, example_input, path, input_names, output_names, dynamic_axes={"input": {0: "batch"}}, opset_version=17)`.
- The classic exporter *traces* the model with the example input. Data-dependent Python control flow is frozen.
- `dynamic_axes` keeps the batch dimension free.
- Always run `onnx.checker.check_model` and a **parity test** (same batch through PyTorch and ORT: max |Δlogit| ~1e-5, identical top-1).

**ONNX Runtime (ORT)** loads the graph, applies **graph optimizations** (constant folding, Conv+BN fusion, redundant node removal) and dispatches to an **execution provider**:

| Provider | Hardware | Notes |
|---|---|---|
| CPU | any | MLAS kernels; INT8 QDQ supported |
| CUDA | NVIDIA GPU | drop-in GPU speed-up |
| TensorRT | NVIDIA GPU | builds an optimized engine (FP16/INT8), slow first run; cache the engine |

**ONNX INT8 (static, QDQ):** a `CalibrationDataReader` feeds representative inputs; ORT inserts QuantizeLinear/DequantizeLinear pairs and runs INT8 kernels where supported. `per_channel=True` quantizes each conv filter separately, which usually costs less accuracy.

**Throughput vs latency:** batch 1 measures what one user waits; batch 32 measures what one GPU can serve. GPUs win big on throughput and less on batch-1 latency, where kernel-launch overhead dominates.

## What to check in your run
- Parity numbers. A large difference means an export bug (often a layer that behaves differently in training mode: call `model.eval()`).
- Node count before/after ORT optimizations.
- Batch-1 latency: PyTorch CPU vs ORT CPU vs ORT CUDA vs TensorRT FP16.
- INT8 ONNX: size, worst-class accuracy, CPU latency.

## Pitfalls
- Exporting in `train()` mode bakes dropout and batch-statistics behaviour into the graph.
- ONNX opset and ORT version must match. A too-new opset fails to load on an old runtime.
- `onnxruntime` and `onnxruntime-gpu` conflict when both are installed. Install one.
- The TensorRT engine is specific to the GPU and TensorRT version. Rebuild it per deployment target.

## Materials
- [ONNX Runtime docs](https://onnxruntime.ai/docs/) · [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [PyTorch → ONNX (PyImageSearch)](https://pyimagesearch.com/) · [torch.onnx](https://pytorch.org/docs/stable/onnx.html)

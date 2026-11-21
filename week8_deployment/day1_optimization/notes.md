# Day 1 Notes — Model Optimization

## Data and model
- **Intel Image Classification** (Kaggle `puneet6060/intel-image-classification`), read from `My Drive/datasets/intel-image-classification/`: 6 scene classes (buildings, forest, glacier, mountain, sea, street). Fixed subset: 400 train and 300 test images per class.
- **Week 2 model:** ResNet-18, `fc = Dropout(0.3) + Linear(512, 6)`, 224 px, ImageNet normalization. Put `pretrained_aug.pt` in `My Drive/checkpoints/week2/` to use yours. Otherwise the notebook trains one with the Week 2 recipe and saves it to `My Drive/cv_course_artifacts/week8/resnet18_week2.pt`.

## Core ideas

**Quantization** stores numbers in fewer bits: FP32 → INT8 means a 4× smaller model and faster integer kernels.

| Kind | Weights | Activations | Needs | Runs on |
|---|---|---|---|---|
| Dynamic | INT8 | quantized at run time per batch | nothing | CPU; helps Linear/LSTM-heavy models |
| Static PTQ | INT8 | INT8 with ranges fixed by **calibration** | ~100–1000 representative images | CPU (fbgemm/x86, qnnpack), ORT, TensorRT |
| QAT | INT8 | INT8, fake-quantized during training | training | same as PTQ, best accuracy |

FX graph mode traces the model, **fuses** Conv+BN+ReLU, inserts observers, calibrates and converts. `torch.ao.quantization` is being replaced by `torchao`. Day 2's ONNX Runtime INT8 path is the portable alternative.

**Pruning** removes weights:
- *Unstructured* (zero individual weights, global L1): high sparsity at small accuracy cost, but dense kernels are **not** faster. The gain is compression (gzip) or sparse hardware.
- *Structured* (remove whole channels/filters, e.g. torch-pruning with dependency graphs): fewer MACs and params, a genuinely faster network. It needs fine-tuning to recover accuracy.

**Distillation:** a small student learns from the teacher's softened probabilities.
`loss = α·T²·KL(softmax(s/T) ‖ softmax(t/T)) + (1−α)·CE(s, y)`. The soft labels carry similarity structure ("glacier looks a bit like mountain"), which helps small students most when training data is limited.

## Metrics
Accuracy, **worst-class accuracy** and per-class deltas, parameters, size on disk (gzip for sparse models), CPU latency at batch 1 (serving), GPU latency and batch-32 throughput.

## What to check in your run
- INT8 static: size ÷4, CPU speed-up, and which class lost the most.
- Unstructured 70 %: accuracy after one epoch of fine-tuning, gzip size, and latency unchanged.
- Structured 30 %: MACs and latency down, accuracy after 2 epochs.
- KD vs CE-only student at the same epochs.

## Pitfalls
- Quantization can drop one class while the average barely moves. Always check per class.
- Calibrate on representative data (the training distribution), not random noise.
- Measure latency at the batch size you will serve (usually 1), after warm-up, with CUDA synchronized.
- INT8 eager/FX models run on CPU only. Don't benchmark them on GPU.

## Materials
- [PyTorch — Quantization](https://pytorch.org/docs/stable/quantization.html) · [torchao](https://github.com/pytorch/ao)
- [PyTorch — Pruning tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) · [Torch-Pruning](https://github.com/VainF/Torch-Pruning)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

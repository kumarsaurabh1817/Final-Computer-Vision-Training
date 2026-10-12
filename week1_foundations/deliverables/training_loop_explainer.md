# Anatomy of the Training Loop

> **Week 1 Deliverable** — A one-page explainer describing what each stage of the training loop does and why normalization helps.

---

## The Training Loop in Four Steps

Every neural network learns through the same fundamental cycle, repeated thousands of times:

```
┌─────────────────────────────────────────────────────────┐
│                    THE TRAINING LOOP                    │
│                                                         │
│   for each batch of data:                               │
│       1. FORWARD PASS   →  predictions = model(inputs)  │
│       2. COMPUTE LOSS   →  loss = criterion(pred, true) │
│       3. BACKWARD PASS  →  loss.backward()              │
│       4. OPTIMIZER STEP →  optimizer.step()              │
│       0. ZERO GRADIENTS →  optimizer.zero_grad()        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### Step 1: Forward Pass — `predictions = model(inputs)`

_Write your explanation here. Cover:_
- What happens when data passes through each layer?
- How are inputs transformed into predictions?
- What are activation functions and why are they needed?

### Step 2: Compute Loss — `loss = criterion(predictions, labels)`

_Write your explanation here. Cover:_
- What is a loss function? (a single number measuring how wrong we are)
- Why CrossEntropyLoss for classification? What does it combine internally?
- How does loss relate to the gradient direction?

### Step 3: Backward Pass (Backpropagation) — `loss.backward()`

_Write your explanation here. Cover:_
- What is the chain rule and how does it apply?
- How do gradients flow from the loss back through each layer?
- What gets stored in `parameter.grad`?
- Why is this the most computationally expensive step?

### Step 4: Optimizer Step — `optimizer.step()`

_Write your explanation here. Cover:_
- The basic update rule: `param = param - lr * param.grad`
- What is the learning rate and why does it matter?
- How SGD with momentum differs from vanilla SGD
- Why too-large LR causes divergence, too-small LR causes slow convergence

### Step 0: Zero Gradients — `optimizer.zero_grad()`

_Write your explanation here. Cover:_
- Why do gradients accumulate by default in PyTorch?
- What happens if you forget this step?
- When might you intentionally skip it? (gradient accumulation for large effective batch sizes)

---

## Why Normalization Helps

### Input Normalization
_Explain why we normalize input data to zero mean, unit variance (or [-1, 1]):_
- What happens to training if pixel values are [0, 255] vs [0, 1]?
- How does the scale affect gradient magnitudes?

### Batch Normalization
_Explain what BatchNorm does and why it helps:_
- What is internal covariate shift?
- How does normalizing intermediate activations speed up training?
- What changes between `model.train()` and `model.eval()`?

---

## Key Diagrams

_Replace with your own diagrams or descriptions:_

```
Input Image (3x32x32)
       │
       ▼
   Conv Layer 1  ──── Filters slide across image, producing feature maps
       │
   BatchNorm     ──── Normalize activations
       │
   ReLU          ──── Zero out negatives (non-linearity)
       │
   MaxPool       ──── Downsample by taking max in each region
       │
       ▼
   Conv Layer 2  ──── Detect higher-level features
       │
      ...
       │
       ▼
   Flatten       ──── (B, C, H, W) → (B, C*H*W)
       │
   Linear        ──── Map features to class scores
       │
       ▼
   10 class scores (logits)
```

---

## Results Summary

| Metric | MLP (Day 3) | CNN (Day 4) | CNN + BN + Dropout (Day 5) |
|--------|-------------|-------------|---------------------------|
| Test Accuracy | __% | __% | __% |
| Parameters | __ | __ | __ |
| Training Time | __s | __s | __s |

---

_Written by: [Your Name]_
_Date: [Date]_

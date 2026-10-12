# Day 3 — Learning Notes

## The Training Loop (Annotated)

### Step 0: `optimizer.zero_grad()`
- _Why is this needed?_

### Step 1: Forward Pass — `outputs = model(inputs)`
- _What happens here?_

### Step 2: Compute Loss — `loss = criterion(outputs, labels)`
- _What does CrossEntropyLoss do internally?_

### Step 3: Backward Pass — `loss.backward()`
- _What does backpropagation compute? How does the chain rule apply?_

### Step 4: Optimizer Step — `optimizer.step()`
- _How are weights updated? What's the role of the learning rate?_

## MLP Results

| Hyperparameter | Value | Test Accuracy |
|----------------|-------|---------------|
| lr=0.01, hidden=512, bs=64 | Default | __% |
| | | |

## Key Observations
- _What's the gap between train and test accuracy? (Overfitting?)_
- _How does the MLP compare to the CNN you'll build tomorrow?_

## Resources Consumed
- [ ] 3Blue1Brown — Neural Networks (video series)
- [ ] PyTorch — Optimizing Model Parameters

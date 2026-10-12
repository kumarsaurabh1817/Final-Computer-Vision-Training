"""
Day 1: PyTorch Basics — Tensors, Autograd & Basic Operations
=============================================================

Objectives:
- Understand tensor creation, operations, and GPU transfer
- Understand autograd and computational graphs
- Build intuition for how gradients flow

Materials:
- PyTorch Learn the Basics: https://pytorch.org/tutorials/beginner/basics/intro.html
- PyTorch 60-Minute Blitz:  https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
"""

import torch
import numpy as np


# =============================================================================
# PART 1: TENSOR BASICS
# =============================================================================

def tensor_creation():
    """Explore different ways to create tensors."""
    print("=" * 60)
    print("PART 1: Tensor Creation")
    print("=" * 60)

    # From Python list
    data = [[1, 2], [3, 4]]
    t_from_list = torch.tensor(data)
    print(f"From list:\n{t_from_list}\n  dtype: {t_from_list.dtype}, shape: {t_from_list.shape}")

    # From NumPy array
    np_array = np.array([[5, 6], [7, 8]])
    t_from_np = torch.from_numpy(np_array)
    print(f"\nFrom numpy:\n{t_from_np}")

    # Common initialization patterns
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    rand = torch.rand(2, 3)          # Uniform [0, 1)
    randn = torch.randn(2, 3)        # Normal (mean=0, std=1)
    print(f"\nZeros:\n{zeros}")
    print(f"\nOnes:\n{ones}")
    print(f"\nRandom uniform:\n{rand}")
    print(f"\nRandom normal:\n{randn}")

    # Like another tensor (same shape + device)
    like_rand = torch.zeros_like(rand)
    print(f"\nZeros like rand:\n{like_rand}")

    # Specify dtype
    float_tensor = torch.ones(2, 3, dtype=torch.float16)
    print(f"\nFloat16 tensor:\n{float_tensor}, dtype: {float_tensor.dtype}")

    # TODO: Try creating tensors of different shapes and dtypes.
    #       What happens with torch.tensor([1, 2.0])?


def tensor_operations():
    """Basic tensor math and indexing."""
    print("\n" + "=" * 60)
    print("PART 2: Tensor Operations")
    print("=" * 60)

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # Element-wise operations
    print(f"a + b =\n{a + b}")
    print(f"a * b (element-wise) =\n{a * b}")
    print(f"a @ b (matrix multiply) =\n{a @ b}")

    # Useful operations
    print(f"\na.sum() = {a.sum()}")
    print(f"a.mean() = {a.mean()}")
    print(f"a.max() = {a.max()}")
    print(f"a.argmax() = {a.argmax()}")

    # Reshaping
    c = torch.arange(12)
    print(f"\nc = {c}")
    print(f"c.reshape(3, 4) =\n{c.reshape(3, 4)}")
    print(f"c.reshape(2, 2, 3) =\n{c.reshape(2, 2, 3)}")

    # Indexing and slicing
    x = torch.rand(4, 5)
    print(f"\nx shape: {x.shape}")
    print(f"x[0] (first row): {x[0]}")
    print(f"x[:, 0] (first col): {x[:, 0]}")
    print(f"x[1:3, 2:4] (slice):\n{x[1:3, 2:4]}")

    # TODO: Practice matrix multiplication of different-shaped tensors.
    #       What's the difference between torch.mm, torch.matmul, and @?


def gpu_operations():
    """Moving tensors to GPU (if available)."""
    print("\n" + "=" * 60)
    print("PART 3: GPU Operations")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Create tensor on GPU directly
    t = torch.rand(3, 3, device=device)
    print(f"\nTensor on {t.device}:\n{t}")

    # Move between devices
    cpu_tensor = torch.rand(3, 3)
    gpu_tensor = cpu_tensor.to(device)
    back_to_cpu = gpu_tensor.cpu()
    print(f"cpu_tensor device: {cpu_tensor.device}")
    print(f"gpu_tensor device: {gpu_tensor.device}")
    print(f"back_to_cpu device: {back_to_cpu.device}")

    # TODO: Time a large matrix multiply on CPU vs GPU.
    #       Use torch.cuda.synchronize() before timing GPU ops.


# =============================================================================
# PART 2: AUTOGRAD — AUTOMATIC DIFFERENTIATION
# =============================================================================

def autograd_basics():
    """Understand autograd and computational graphs."""
    print("\n" + "=" * 60)
    print("PART 4: Autograd Basics")
    print("=" * 60)

    # requires_grad=True tells PyTorch to track operations for gradient computation
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1
    print(f"x = {x}")
    print(f"y = x^2 + 2x + 1 = {y}")

    # Compute gradients (dy/dx = 2x + 2 = 8 at x=3)
    y.backward()
    print(f"dy/dx = {x.grad}  (expected: 2*3 + 2 = 8)")

    # Multi-variable example
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    z = a * b + a ** 2
    z.backward()
    print(f"\nz = a*b + a^2")
    print(f"dz/da = b + 2a = {a.grad}  (expected: 3 + 4 = 7)")
    print(f"dz/db = a = {b.grad}  (expected: 2)")

    # IMPORTANT: Gradients accumulate! You must zero them between calls.
    print("\n--- Gradient accumulation demo ---")
    w = torch.tensor(1.0, requires_grad=True)
    for i in range(3):
        loss = (w * 2) ** 2
        loss.backward()
        print(f"  Iteration {i}: w.grad = {w.grad}")
        # Uncomment the next line to see correct behavior:
        # w.grad.zero_()

    # TODO: Add w.grad.zero_() after each backward() call and observe the difference.


def autograd_with_tensors():
    """Autograd with multi-dimensional tensors (like real networks)."""
    print("\n" + "=" * 60)
    print("PART 5: Autograd with Tensors")
    print("=" * 60)

    # Simulating a simple linear layer: y = Wx + b
    W = torch.randn(3, 5, requires_grad=True)   # "weights"
    b = torch.randn(3, requires_grad=True)       # "bias"
    x = torch.randn(5)                           # "input"

    # Forward pass
    y = W @ x + b
    print(f"Input shape:  {x.shape}")
    print(f"Weight shape: {W.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output: {y}")

    # We need a scalar loss to call backward()
    loss = y.sum()  # Simple placeholder loss
    loss.backward()

    print(f"\nW.grad shape: {W.grad.shape}")
    print(f"b.grad shape: {b.grad.shape}")
    print(f"W.grad:\n{W.grad}")
    print(f"b.grad: {b.grad}")

    # The no_grad context: disables gradient tracking (used during inference)
    with torch.no_grad():
        inference_result = W @ x + b
        print(f"\nInference (no_grad): {inference_result}")
        print(f"Has grad_fn: {inference_result.grad_fn}")  # None

    # TODO: Build a 2-layer computation (linear -> relu -> linear -> loss)
    #       and verify that gradients flow through both layers.


# =============================================================================
# PART 3: BUILDING A SIMPLE LINEAR MODEL
# =============================================================================

def simple_linear_regression():
    """Train a linear regression model using raw autograd — no nn.Module yet."""
    print("\n" + "=" * 60)
    print("PART 6: Simple Linear Regression with Autograd")
    print("=" * 60)

    # Generate synthetic data: y = 3x + 2 + noise
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10          # 100 data points
    y_true = 3 * X + 2 + torch.randn(100, 1) * 0.5

    # Initialize parameters
    w = torch.randn(1, requires_grad=True)
    b_param = torch.zeros(1, requires_grad=True)

    learning_rate = 0.01
    n_epochs = 100

    for epoch in range(n_epochs):
        # Forward pass
        y_pred = w * X + b_param

        # Compute loss (MSE)
        loss = ((y_pred - y_true) ** 2).mean()

        # Backward pass
        loss.backward()

        # Update parameters (must be inside no_grad!)
        with torch.no_grad():
            w -= learning_rate * w.grad
            b_param -= learning_rate * b_param.grad

        # Zero gradients for next iteration
        w.grad.zero_()
        b_param.grad.zero_()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | w: {w.item():.3f} | b: {b_param.item():.3f}")

    print(f"\nLearned: y = {w.item():.2f}x + {b_param.item():.2f}")
    print(f"Truth:   y = 3.00x + 2.00")

    # TODO: Try different learning rates. What happens with lr=0.1? lr=0.001?


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    tensor_creation()
    tensor_operations()
    gpu_operations()
    autograd_basics()
    autograd_with_tensors()
    simple_linear_regression()

    print("\n" + "=" * 60)
    print("DAY 1 COMPLETE!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Tensors are the fundamental data structure (like numpy arrays, but GPU-capable)")
    print("  2. requires_grad=True enables autograd to track operations")
    print("  3. .backward() computes gradients via the chain rule")
    print("  4. Gradients accumulate — always zero them between steps")
    print("  5. Use torch.no_grad() during inference to save memory")
    print("\nNext: Day 2 — Image fundamentals with OpenCV & Pillow")

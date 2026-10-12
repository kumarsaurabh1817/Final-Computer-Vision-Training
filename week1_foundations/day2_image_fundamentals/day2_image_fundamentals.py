"""
Day 2: Image Fundamentals — Pixels, Channels, Color Spaces & Transformations
==============================================================================

Objectives:
- Understand how images are represented as tensors (H x W x C)
- Load, display, and manipulate images with OpenCV and Pillow
- Explore color spaces (RGB, BGR, HSV, Grayscale)
- Practice resizing, cropping, rotating, and basic filtering

Materials:
- OpenCV — Getting Started with Images
- Image Processing with OpenCV & Pillow
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory for saving results
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_DIR = Path(__file__).parent / "sample_images"
SAMPLE_DIR.mkdir(exist_ok=True)


# =============================================================================
# PART 1: IMAGE BASICS — WHAT IS AN IMAGE?
# =============================================================================

def create_synthetic_images():
    """Create images from scratch to understand the pixel representation."""
    print("=" * 60)
    print("PART 1: Synthetic Images — Understanding Pixels")
    print("=" * 60)

    # A grayscale image is a 2D array of pixel intensities [0, 255]
    gray_img = np.zeros((100, 200), dtype=np.uint8)
    gray_img[20:80, 50:150] = 200  # White rectangle
    print(f"Grayscale image shape: {gray_img.shape}  (H x W)")
    print(f"Pixel value range: [{gray_img.min()}, {gray_img.max()}]")

    # A color image is a 3D array: H x W x C (3 channels)
    color_img = np.zeros((100, 200, 3), dtype=np.uint8)
    color_img[20:80, 50:100] = [255, 0, 0]    # Red rectangle (RGB)
    color_img[20:80, 100:150] = [0, 255, 0]   # Green rectangle
    color_img[40:60, 75:125] = [0, 0, 255]    # Blue rectangle (overlapping)
    print(f"\nColor image shape: {color_img.shape}  (H x W x C)")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title("Grayscale (H x W)")
    axes[1].imshow(color_img)
    axes[1].set_title("Color (H x W x 3)")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_synthetic_images.png", dpi=100)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '01_synthetic_images.png'}")

    # TODO: Create a gradient image (pixel values increase from left to right).
    #       Create a checkerboard pattern.


# =============================================================================
# PART 2: LOADING IMAGES — OPENCV vs PILLOW
# =============================================================================

def load_images_demo():
    """Load images with both libraries and understand the differences."""
    print("\n" + "=" * 60)
    print("PART 2: Loading Images — OpenCV vs Pillow")
    print("=" * 60)

    # Create a sample image if none exists
    sample_path = SAMPLE_DIR / "sample.png"
    if not sample_path.exists():
        print(f"No sample image found. Creating one at {sample_path}")
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        # Draw some shapes to make it more interesting
        cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)
        cv2.circle(img, (128, 128), 60, (255, 0, 0), -1)
        cv2.putText(img, "Sample", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(str(sample_path), img)
        print(f"Created sample image: {sample_path}")

    # --- OpenCV ---
    img_cv = cv2.imread(str(sample_path))
    print(f"\nOpenCV  — shape: {img_cv.shape}, dtype: {img_cv.dtype}")
    print(f"  Channel order: BGR (not RGB!)")
    print(f"  Pixel [0,0]: {img_cv[0, 0]}  (B, G, R)")

    # --- Pillow ---
    img_pil = Image.open(sample_path)
    img_pil_np = np.array(img_pil)
    print(f"\nPillow  — size: {img_pil.size} (W, H), mode: {img_pil.mode}")
    print(f"  As numpy — shape: {img_pil_np.shape}, dtype: {img_pil_np.dtype}")
    print(f"  Channel order: RGB")
    print(f"  Pixel [0,0]: {img_pil_np[0, 0]}  (R, G, B)")

    # ⚠️ KEY DIFFERENCE: OpenCV uses BGR, Pillow uses RGB
    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_cv)  # Will look wrong — BGR shown as RGB
    axes[0].set_title("OpenCV (BGR displayed as RGB)\n⚠️ Colors wrong!")
    axes[1].imshow(img_cv_rgb)
    axes[1].set_title("OpenCV (converted to RGB)\n✅ Colors correct")
    axes[2].imshow(img_pil)
    axes[2].set_title("Pillow (native RGB)\n✅ Colors correct")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_opencv_vs_pillow.png", dpi=100)
    plt.close()

    # TODO: Load a photo from your phone / web. Compare the pixel values
    #       in both libraries at the same coordinates.


# =============================================================================
# PART 3: COLOR SPACES
# =============================================================================

def color_spaces_demo():
    """Explore different color space representations."""
    print("\n" + "=" * 60)
    print("PART 3: Color Spaces")
    print("=" * 60)

    sample_path = SAMPLE_DIR / "sample.png"
    img_bgr = cv2.imread(str(sample_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to different color spaces
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    print(f"RGB shape:  {img_rgb.shape}")
    print(f"Gray shape: {img_gray.shape}")
    print(f"HSV shape:  {img_hsv.shape}")
    print(f"\nHSV at center pixel: H={img_hsv[128,128,0]}, S={img_hsv[128,128,1]}, V={img_hsv[128,128,2]}")

    # Visualize channels
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # RGB channels
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original (RGB)")
    for i, (ch_name, color) in enumerate(zip(['Red', 'Green', 'Blue'],
                                               ['Reds', 'Greens', 'Blues'])):
        axes[0, i + 1].imshow(img_rgb[:, :, i], cmap=color)
        axes[0, i + 1].set_title(f"{ch_name} Channel")

    # Grayscale + HSV channels
    axes[1, 0].imshow(img_gray, cmap='gray')
    axes[1, 0].set_title("Grayscale")
    for i, ch_name in enumerate(['Hue', 'Saturation', 'Value']):
        axes[1, i + 1].imshow(img_hsv[:, :, i], cmap='gray')
        axes[1, i + 1].set_title(f"HSV — {ch_name}")

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_color_spaces.png", dpi=100)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '03_color_spaces.png'}")

    # TODO: Convert an image to HSV, shift the Hue channel by 60,
    #       convert back to RGB, and see what happens to the colors.


# =============================================================================
# PART 4: GEOMETRIC TRANSFORMATIONS
# =============================================================================

def transformations_demo():
    """Resize, crop, rotate, flip — the operations you'll use in data pipelines."""
    print("\n" + "=" * 60)
    print("PART 4: Geometric Transformations")
    print("=" * 60)

    sample_path = SAMPLE_DIR / "sample.png"
    img = cv2.imread(str(sample_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    print(f"Original size: {w}x{h}")

    # Resize
    resized = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
    print(f"Resized to: {resized.shape[1]}x{resized.shape[0]}")

    # Crop (just array slicing)
    crop = img_rgb[50:200, 50:200]
    print(f"Cropped to: {crop.shape[1]}x{crop.shape[0]}")

    # Rotate
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
    rotated = cv2.warpAffine(img_rgb, M, (w, h))

    # Flip
    flipped_h = cv2.flip(img_rgb, 1)  # Horizontal
    flipped_v = cv2.flip(img_rgb, 0)  # Vertical

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    titles_imgs = [
        ("Original", img_rgb), ("Resized (128x128)", resized),
        ("Cropped", crop), ("Rotated 45°", rotated),
        ("Flipped Horizontal", flipped_h), ("Flipped Vertical", flipped_v),
    ]
    for ax, (title, im) in zip(axes.flat, titles_imgs):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_transformations.png", dpi=100)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '04_transformations.png'}")

    # TODO: Resize an image with different interpolation methods
    #       (NEAREST, LINEAR, CUBIC, AREA) and compare quality.


# =============================================================================
# PART 5: BASIC FILTERING
# =============================================================================

def filtering_demo():
    """Blur, sharpen, edge detection — understand convolution visually."""
    print("\n" + "=" * 60)
    print("PART 5: Basic Filtering (Preview of Day 4 Convolutions)")
    print("=" * 60)

    sample_path = SAMPLE_DIR / "sample.png"
    img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)

    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (11, 11), 0)

    # Edge detection (Canny)
    edges = cv2.Canny(img, 50, 150)

    # Sobel gradients (x and y)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    imgs_titles = [
        (img, "Original (Gray)"), (blurred, "Gaussian Blur"),
        (edges, "Canny Edges"), (np.abs(sobel_x), "Sobel X"),
        (np.abs(sobel_y), "Sobel Y"), (sobel_mag, "Sobel Magnitude"),
    ]
    for ax, (im, title) in zip(axes.flat, imgs_titles):
        ax.imshow(im, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_filtering.png", dpi=100)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '05_filtering.png'}")

    # NOTE: These filters are CONVOLUTIONS — exactly what CNNs learn automatically!
    # TODO: Try different kernel sizes for GaussianBlur and Canny thresholds.


# =============================================================================
# PART 6: IMAGE AS A PYTORCH TENSOR
# =============================================================================

def image_as_tensor():
    """Convert images to PyTorch tensors — the bridge to Day 3."""
    import torch
    import torchvision.transforms as T

    print("\n" + "=" * 60)
    print("PART 6: Image as a PyTorch Tensor")
    print("=" * 60)

    sample_path = SAMPLE_DIR / "sample.png"
    img_pil = Image.open(sample_path).convert("RGB")

    # Manual conversion
    img_np = np.array(img_pil)                       # (H, W, C) uint8 [0, 255]
    img_tensor = torch.from_numpy(img_np).float()    # (H, W, C) float32
    img_tensor = img_tensor.permute(2, 0, 1)         # (C, H, W) — PyTorch convention
    img_tensor = img_tensor / 255.0                   # Normalize to [0, 1]

    print(f"PIL image size: {img_pil.size}")
    print(f"NumPy shape: {img_np.shape}  (H, W, C)")
    print(f"Tensor shape: {img_tensor.shape}  (C, H, W)")
    print(f"Tensor range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

    # Using torchvision transforms (the standard way)
    transform = T.Compose([
        T.Resize((224, 224)),          # Standard input size for many models
        T.ToTensor(),                   # (H,W,C) uint8 → (C,H,W) float [0,1]
        T.Normalize(                    # ImageNet normalization
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img_transformed = transform(img_pil)
    print(f"\nTransformed tensor shape: {img_transformed.shape}")
    print(f"Transformed range: [{img_transformed.min():.2f}, {img_transformed.max():.2f}]")
    print(f"(Range is wider than [0,1] because of normalization)")

    # Batch dimension: models expect (B, C, H, W)
    batch = img_transformed.unsqueeze(0)  # Add batch dim
    print(f"\nBatch shape: {batch.shape}  (B, C, H, W)")

    # TODO: Apply different transforms (RandomCrop, RandomHorizontalFlip)
    #       and visualize the augmented images side by side.


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    create_synthetic_images()
    load_images_demo()
    color_spaces_demo()
    transformations_demo()
    filtering_demo()
    image_as_tensor()

    print("\n" + "=" * 60)
    print("DAY 2 COMPLETE!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Images are just arrays of numbers: (H, W) gray or (H, W, 3) color")
    print("  2. OpenCV uses BGR, Pillow/matplotlib use RGB — always convert!")
    print("  3. PyTorch expects tensors in (C, H, W) format, normalized to [0, 1]")
    print("  4. torchvision.transforms handles the conversion pipeline")
    print("  5. Filters (blur, edge detection) ARE convolutions — CNNs learn them!")
    print("\nNext: Day 3 — The training loop by hand (MLP on CIFAR-10)")

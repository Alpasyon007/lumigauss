"""
Generate sky masks for a scene using DeepLabV3 semantic segmentation.
Outputs binary masks where black (0) = sky, white (255) = not sky.
This matches the convention expected by load_sky_masks() in dataset_readers.py.

Usage:
    python generate_sky_masks.py --input_dir ./data/lk2_colmap/undistorted/images --output_dir ./data/lk2_colmap/dynamic_masks/sky
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from tqdm import tqdm


def generate_sky_masks(input_dir: str, output_dir: str, threshold: float = 0.5,
                       device: str = "cuda"):
    """Generate sky masks for all images in input_dir using DeepLabV3."""
    os.makedirs(output_dir, exist_ok=True)

    # Load pretrained DeepLabV3 model
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).to(device).eval()

    # COCO-stuff class index for "sky" is not in the 21 VOC classes.
    # In Pascal VOC (which DeepLabV3 uses), there's no explicit "sky" class.
    # However, sky pixels typically get classified as "background" (class 0).
    # We'll use a different approach: anything not classified as a known object
    # AND in the upper portion of the image with high brightness is likely sky.
    #
    # Actually, let's use the COCO-pretrained model which HAS a sky class.
    # DeepLabV3_ResNet101_Weights.DEFAULT is trained on COCO with 21 classes (Pascal VOC).
    # Sky is not a VOC class. Let's use a simpler approach instead.

    preprocess = weights.transforms()

    # Collect image files
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images from {input_dir}")
    print(f"Output directory: {output_dir}")

    # Pascal VOC classes used by DeepLabV3:
    # 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle,
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog,
    # 13=horse, 14=motorbike, 15=person, 16=pottedplant, 17=sheep,
    # 18=sofa, 19=train, 20=tvmonitor
    #
    # Sky gets classified as "background" (0). But so does ground, walls, etc.
    # Strategy: use background class + spatial prior (upper image) + color prior (blue/bright)

    for fname in tqdm(image_files, desc="Generating sky masks"):
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Run segmentation
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)["out"]  # [1, 21, H', W']
            probs = torch.softmax(output, dim=1)  # [1, 21, H', W']

        # Background probability (class 0)
        bg_prob = probs[0, 0]  # [H', W']

        # Resize to original image size
        bg_prob = torch.nn.functional.interpolate(
            bg_prob.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze()  # [H, W]

        # Color-based sky prior: sky tends to be bright and blue-ish
        img_np = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
        brightness = img_np.mean(axis=2)  # [H, W]
        # Blue channel dominance: sky is often more blue than red
        blue_ratio = img_np[:, :, 2] / (img_np.mean(axis=2) + 1e-6)

        # Spatial prior: sky is more likely in the upper part of the image
        y_coords = np.linspace(1.0, 0.0, H)[:, None]  # 1 at top, 0 at bottom
        spatial_prior = np.clip(y_coords * 1.5 - 0.2, 0, 1)  # Strong at top, weak at bottom

        # Combine: background prob × (brightness + blue_ratio + spatial) / 3
        sky_score = bg_prob.cpu().numpy() * (
            0.4 * brightness +
            0.3 * np.clip(blue_ratio - 0.8, 0, 1) * 3.0 +
            0.3 * spatial_prior.squeeze()
        )

        # Threshold to binary mask
        sky_binary = (sky_score > threshold).astype(np.uint8)

        # Post-process: fill small holes and remove small regions
        from scipy import ndimage
        # Remove small sky regions (noise)
        sky_labeled, n_features = ndimage.label(sky_binary)
        if n_features > 0:
            # Keep only large connected components
            min_size = H * W * 0.01  # At least 1% of image
            for i in range(1, n_features + 1):
                if (sky_labeled == i).sum() < min_size:
                    sky_binary[sky_labeled == i] = 0

        # Fill small holes in non-sky areas within sky
        non_sky = 1 - sky_binary
        non_sky_labeled, n_features_ns = ndimage.label(non_sky)
        if n_features_ns > 0:
            min_hole = H * W * 0.005
            for i in range(1, n_features_ns + 1):
                region = non_sky_labeled == i
                if region.sum() < min_hole:
                    # Check if this hole is surrounded by sky (i.e., in upper portion)
                    ys = np.where(region)[0]
                    if ys.mean() < H * 0.6:  # Only fill holes in upper part
                        sky_binary[region] = 1

        # Create output mask: 0=sky (black), 255=not-sky (white)
        mask = ((1 - sky_binary) * 255).astype(np.uint8)
        # Wait, convention is black=sky, white=not sky
        # sky_binary: 1=sky, 0=not-sky
        # Output: 0 where sky (black), 255 where not sky (white)
        mask = ((1 - sky_binary) * 255).astype(np.uint8)

        # Save as PNG with same base name
        base_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(output_dir, base_name)
        Image.fromarray(mask).save(out_path)

    print(f"\nDone! Generated {len(image_files)} sky masks in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sky masks using DeepLabV3 segmentation")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save sky masks")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Sky detection threshold (lower = more sky detected)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run model on")
    args = parser.parse_args()

    generate_sky_masks(args.input_dir, args.output_dir, args.threshold, args.device)

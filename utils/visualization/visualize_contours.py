"""
File: scripts/visualization/visualize_contours.py
Description: Script to visualize 2D contours from a 3D NIfTI image and its corresponding mask.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
import argparse
import os

def visualize_contours(image_path, mask_path, output_file):
    """
    Visualize 2D contours on a slice of a 3D NIfTI image and save the output.

    Args:
        image_path (str): Path to the 3D NIfTI image file.
        mask_path (str): Path to the 3D NIfTI mask file.
        slice_index (int): Index of the slice to visualize.
        output_file (str): Path to save the output image.
    """

    # Check if the specified files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Load the NIfTI image and extract the specified slice
    slice_index = img.shape[2] // 2
    img = nib.load(image_path).get_fdata()
    img = img[:, :, slice_index]

    # Load the NIfTI mask and extract the specified slice
    mask = nib.load(mask_path).get_fdata()
    mask = mask[:, :, slice_index]

    # Generate contours from the mask at different levels
    contours1 = measure.find_contours(mask, level=0.5)
    contours2 = measure.find_contours(mask, level=1.0)

    # Plot the image and overlay the contours
    plt.figure(figsize=(12, 8))
    plt.imshow(img, cmap='gray')
    for contour in contours1:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, label='Level 0.5 Contour')

    for contour in contours2:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, label='Level 1 Contour')

    # Add legend, title, and save the figure
    plt.legend()
    plt.title(f"Slice {slice_index} with Overlaid Contours")
    plt.axis('off')
    plt.savefig(output_file)
    plt.close()
    print(f"Visualization saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 2D contours from a 3D NIfTI image and mask."
    )
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to the 3D NIfTI image file.")
    parser.add_argument("--mask", "-m", type=str, required=True, help="Path to the 3D NIfTI mask file.")
    parser.add_argument("--output", "-o", type=str, default="output.png", help="Path to save the output image. Default is 'output.png'.")
    args = parser.parse_args()

    # Check if the specified files exist
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask file not found: {args.mask}")

    # Run the visualization function
    visualize_contours(args.image, args.mask, args.output)

if __name__ == "__main__":
    main()

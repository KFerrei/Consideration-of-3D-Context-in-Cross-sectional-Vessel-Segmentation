"""
File: utils/visualization/image_transformations.py
Description: Apply different image transformations (both spatial and intensity) to 3D medical images.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import torchio as tio
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import argparse
import os
from skimage import measure
import copy

def load_image_and_label(image_path, mask_path):
    """
    Loads a 3D image and its corresponding label (mask) from the given paths.

    Args:
        image_path (str): Path to the input image file (.nii.gz).
        mask_path (str): Path to the corresponding label file (.nii.gz).

    Returns:
        subject (torchio.Subject): A Subject containing the image and label as TorchIO ScalarImages.
    """
    # Check if the specified files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    image = nib.load(image_path)
    image_data = image.get_fdata()
    image_tensor = torch.tensor(image_data, dtype=torch.float32).squeeze().unsqueeze(0)

    label = nib.load(mask_path)
    label_data = label.get_fdata()
    label_tensor = torch.tensor(label_data, dtype=torch.int64).squeeze().unsqueeze(0)

    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image_tensor),
        label=tio.ScalarImage(tensor=label_tensor)
    )

    return subject

def apply_transformations(subject):
    """
    Applies a set of spatial and intensity transformations to the image and its label.

    Args:
        subject (torchio.Subject): The subject containing the image and label to transform.

    Returns:
        transformed_subjects (list of tuples): A list of tuples containing the transformation name and the transformed subject.
    """
    transformations_space = [
        ('Original', None),
        ('Flip', tio.RandomFlip(axes=0, flip_probability=1)),
        ('Affine', tio.RandomAffine(scales=[0.1, 0.1, 0], degrees=[0, 0, 120], translation=[25, 25, 0], p=1.0)),
        ('Elastic', tio.RandomElasticDeformation(6, (10, 10, 0), p=1.0)),
    ]

    transformations_intensity = [
        ('Motion', tio.RandomMotion(degrees=10, translation=10, num_transforms=2, p=1.0)),
        ('Ghosting', tio.RandomGhosting(num_ghosts=5, axes=(1, 2), intensity=0.8, p=1.0)),
        ('Bias Field', tio.RandomBiasField(coefficients=0.9, order=3, p=1.0)),
        ('Noise', tio.RandomNoise(mean=0.1, std=1, p=1.0))
    ]
    
    # Apply spatial transformations
    transformed_subjects_space = []
    for title, transform in transformations_space:
        if transform is not None:
            transformed_subject = transform(subject)
        else:
            transformed_subject = subject
        transformed_subjects_space.append((title, transformed_subject))

    # Apply intensity transformations
    transformed_subjects_intensity = []
    for title, transform in transformations_intensity:
        transformed_subject = copy.deepcopy(subject)
        transformed_image = transform(transformed_subject.image.data)
        transformed_subject.image.set_data(transformed_image)
        transformed_subjects_intensity.append((title, transformed_subject))

    return transformed_subjects_space, transformed_subjects_intensity

def main(args):
    """
    Main function to load an image and its corresponding label, apply transformations, and visualize results.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Load image and label
    image_path = os.path.join(args.data_dir, 'images', args.file)
    mask_path = os.path.join(args.data_dir, 'masks', args.file)
    subject = load_image_and_label(image_path, mask_path)

    # Apply transformations
    transformed_subjects_space, transformed_subjects_intensity = apply_transformations(subject)

    # Visualization setup
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    
    # Loop and plot space transformation
    for j, transformed_subjects in enumerate([transformed_subjects_space, transformed_subjects_intensity]):
        for i, (title, tss) in enumerate(transformed_subjects):
            transformed_image = tss.image.data[0, :, :, 4].cpu().numpy()
            transformed_mask = tss.label.data[0, :, :, 4].cpu().numpy()
            contours1 = measure.find_contours(transformed_mask, level=0.5)
            contours2 = measure.find_contours(transformed_mask, level=1)
            axs[j, i].imshow(transformed_image, cmap='gray')
            for contour in [contours1, contours2]:
                for c in contour:
                    axs[j, i].plot(c[:, 1], c[:, 0], linewidth=2)
            axs[j, i].axis('off')
            axs[j, i].set_title(f"{title}")
    # Save the result
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/transformations.png")
    plt.close()
    print(f"Visualization saved to {args.output_dir}/transformations.png")

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Apply transformations to 3D medical images.")
    parser.add_argument('--data_dir', '-dd',type=str, required=True, help='Directory where the dataset is located.')
    parser.add_argument('--file', '-f', type=str, required=True, help='The image file to process (e.g., "image.nii.gz").')
    parser.add_argument('--output_dir', '-od', type=str, default='.', help='Directory to save the output image (default: current directory).')
    args = parser.parse_args()
    main(args)    

"""
File: utils/visualization/plot_bland_altman.py
Description: Data plotting for analyzing the agreement between two different lists.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import nibabel as nib
from evalutils.stats import hausdorff_distance
import torch
from tqdm import tqdm
from config import DEVICE
from utils.helpers import get_args
from scripts import init_model_test, init_datasets_test

def create_wall_dist_dicts(pairs_folder, model, slice_of_interest):
    """
    Compute Hausdorff distance between ground truth and model prediction for each sample.

    Args:
        pairs_folder (list): List of dictionaries containing image and label paths.
        model (torch.nn.Module): Trained model to make predictions.
        slice_of_interest (int): The slice index to be used for evaluation.

    Returns:
        tuple: Two dictionaries with file names and corresponding max wall thickness distances
               for ground truth and model predictions.
    """
    file_names = []
    distances_gt = []
    distances_model = []

    # Loop through each sample in the dataset
    for idx in tqdm(range(len(pairs_folder))):
        # Load the image and label, convert to tensors and move to device
        image = nib.load(pairs_folder[idx]["image"]).get_fdata().squeeze()
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        label = nib.load(pairs_folder[idx]["label"]).get_fdata().squeeze()
        label = label[:, :, slice_of_interest]  # Extract the specific slice of interest

        # Compute Hausdorff distance for ground truth
        distance_gt = hausdorff_distance(label == 1, label == 2, nib.load(pairs_folder[idx]["label"]).header["pixdim"][1:3])

        # Perform model inference
        model.eval()
        pred = model(image)
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        # Compute Hausdorff distance for model prediction
        try:
            distance_model = hausdorff_distance(pred == 1, pred == 2, nib.load(pairs_folder[idx]["label"]).header["pixdim"][1:3])
        except ValueError:
            distance_model = None

        # Store valid results
        if distance_model is not None:
            file_names.append(os.path.basename(pairs_folder[idx]["label"]))
            distances_gt.append(distance_gt)
            distances_model.append(distance_model)

    return {"file_name": file_names, "max_wall_thickness": distances_gt}, {"file_name": file_names, "max_wall_thickness": distances_model}


def plot_bland_altman(by_slice, all_gt, all_model, saving_dir):
    """
    Plot Bland-Altman plot for the given distances.

    Args:
        by_slice (dict): Dictionary containing distances by slice.
        all_gt (np.array): Ground truth distances.
        all_model (np.array): Model prediction distances.
        saving_dir (str): Directory where the plot will be saved.
    """
    colors = plt.cm.tab10(range(8))
    labels = ["Plane 1", "Plane 2", "Plane 3", "Plane 4", "Plane 5", "Plane 6", "Plane 7", "Plane 8"]

    for idx, distances in enumerate(by_slice.values()):
        distances = np.array(distances)
        mean_value_of_two_ratings = np.mean(distances, axis=0)
        difference_between_two_ratings = distances[0] - distances[1]
        plt.scatter(mean_value_of_two_ratings, difference_between_two_ratings, color=colors[idx], label=labels[idx])

    plt.legend()

    difference_between_two_ratings = all_gt - all_model
    std = np.std(difference_between_two_ratings)
    mean = np.mean(difference_between_two_ratings)

    # Add lines for mean and 95% confidence intervals
    plt.axhline(mean, color='gray', linestyle='--', lw=0.4)
    plt.axhline(mean + 1.96 * std, color='gray', linestyle='--', lw=0.4)
    plt.axhline(mean - 1.96 * std, color='gray', linestyle='--', lw=0.4)

    plt.xlabel("Mean $max(VWT)$ [mm]")
    plt.ylabel(r'Ground truth $max(VWT)$ - $M_R$ $max(VWT)$ [mm]')
    plt.savefig(os.path.join(saving_dir, "bland_altman.png"))
    plt.show()


def calculate_icc(all_gt, all_model, saving_dir):
    """
    Calculate the Intraclass Correlation Coefficient (ICC) for the ground truth and model predictions.

    Args:
        all_gt (np.array): Ground truth distances.
        all_model (np.array): Model prediction distances.
        saving_dir (str): Directory where the ICC results will be saved.

    Returns:
        None
    """
    num_slices = len(all_gt)
    exam = list(range(num_slices)) * 2
    judge = ["A"] * num_slices + ["B"] * num_slices
    rating = list(all_gt) + list(all_model)

    df = pd.DataFrame({'exam': exam, 'judge': judge, 'rating': rating})
    icc = pg.intraclass_corr(data=df, targets='exam', raters='judge', ratings='rating')
    icc.set_index('Type')

    # Save ICC results
    icc.to_csv(os.path.join(saving_dir, 'icc_results.csv'), index=True)


def save_difference_statistics(difference_between_two_ratings, saving_dir):
    """
    Save the statistics of the difference between ground truth and model predictions.

    Args:
        difference_between_two_ratings (np.array): Differences between ground truth and model predictions.
        saving_dir (str): Directory where the statistics will be saved.

    Returns:
        None
    """
    std = np.std(difference_between_two_ratings)
    mean = np.mean(difference_between_two_ratings)

    with open(os.path.join(saving_dir, 'difference_between_two_ratings.txt'), 'w') as f:
        f.write(f"Mean: {mean} \nSTD: {std}")

if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    
    test_dataset = init_datasets_test(args.folder, args.slices, args.sides, args.slice_of_interest, 
                                      args.num_classes, args.normalize)

    model = init_model_test(args.name_model, args.skip_connection_type, args.bottleneck_type, 
                             args.num_classes, args.slice_of_interest, args.filter_1, args.depth, 
                             args.dropout, args.n_slices, args.model_to_load)

    # Compute distances for the test set
    distances_gt, distances_model = create_wall_dist_dicts(test_pairs, model, args.slice_interest)

    # Convert results to NumPy arrays for easier manipulation
    all_gt = np.array(distances_gt["max_wall_thickness"])
    all_model = np.array(distances_model["max_wall_thickness"])

    # Organize distances by slice for plotting
    by_slice = {str(i): [[], []] for i in range(1, 9)}

    for filename, d_gt, d_model in zip(distances_gt["file_name"], distances_gt["max_wall_thickness"], distances_model["max_wall_thickness"]):
        for key in by_slice.keys():
            if key in filename:
                by_slice[key][0].append(d_gt)
                by_slice[key][1].append(d_model)
    
    # Set up saving directory
    saving_dir = os.path.join(args.saving_dir, f"{model.checkpoint['name']}")
    os.makedirs(saving_dir, exist_ok=True)

    # Plot Bland-Altman plot
    plot_bland_altman(by_slice, all_gt, all_model, saving_dir)

    # Calculate ICC
    calculate_icc(all_gt, all_model, saving_dir)

    # Save difference statistics
    difference_between_two_ratings = all_gt - all_model
    save_difference_statistics(difference_between_two_ratings, saving_dir)
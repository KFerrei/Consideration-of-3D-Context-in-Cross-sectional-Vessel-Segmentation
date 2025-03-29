"""
File: utils/visualization/train.py
Description: Visualisation of a comparison between the ground truth and model predictions for a given set of image slices.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import matplotlib.pyplot as plt
import torch
from skimage import measure
import os

def make_inference(data, model, slice_of_interest):
    """
    Makes inferences of the model.

    Args:
        data (list): A list of tuples containing images and labels.
        model (torch.nn.Module): The trained model used for inference.
        slice_of_interest (int): Index of the slice to be visualized.

    Returns:
        image (torch): The image of interest
        label (torch): The label of interest
        pred  (torch): The prediction mask made by the model
    """
    image, label = data
    device = next(model.parameters()).device
    image = image.to(device).unsqueeze(0)
    pred = model(image)
    pred = torch.argmax(pred, dim = 1)
    return image[:,:,:, :, slice_of_interest], label, pred

def plot_inferences(dataset, model, saving_dir, slice_of_interest, title = "inferences.png", list_ind = [0, 1, 2]):
    """
    Generates and saves a comparison of ground truth and model predictions for a given set of image slices.

    Args:
        dataset (list): A list of tuples containing images and labels.
        model (torch.nn.Module): The trained model used for inference.
        saving_dir (str): Directory where the comparison images will be saved.
        slice_of_interest (int): Index of the slice to be visualized.
        title (str): The filename for saving the plot. Defaults to "inferences.png".
        list_ind (list): List of indices specifying which dataset samples to visualize. Defaults to [0, 1, 2].

    Returns:
        None
    """
    n_inferences = len(list_ind)
    fig, axs = plt.subplots(n_inferences, 3, figsize = (10, n_inferences*3))

    for i, k in enumerate(list_ind):
        image2D, label, pred = make_inference(dataset[k], model, slice_of_interest)
        # Show and save image
        axs[i, 0].imshow(image2D.detach().cpu().squeeze(), cmap = 'gray')
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image')

        contours1 = measure.find_contours(label.numpy(), level=0.5)
        contours2 = measure.find_contours(label.numpy(), level=1)

        axs[i, 1].imshow(image2D.detach().cpu().squeeze(), cmap = 'gray')  
        for contour in contours1:
            axs[i, 1].plot(contour[:, 1], contour[:, 0], linewidth=2)
        for contour in contours2:
            axs[i, 1].plot(contour[:, 1], contour[:, 0], linewidth=2)     
        axs[i, 1].axis('off')
        axs[i, 1].set_title(f'Ground truth') 

        axs[i, 2].imshow(pred.detach().cpu().squeeze(), cmap = 'gray')
        for contour in contours1:
            axs[i, 2].plot(contour[:, 1], contour[:, 0], linewidth=2)
        for contour in contours2:
            axs[i, 2].plot(contour[:, 1], contour[:, 0], linewidth=2)    
        axs[i, 2].axis('off')
        axs[i, 2].set_title(f'Prediction') 
    plt.tight_layout()     
    plt.savefig(os.path.join(saving_dir, title))
    plt.close()
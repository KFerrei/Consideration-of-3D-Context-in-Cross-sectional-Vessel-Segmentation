"""
File: scripts/test_model.py
Description: Test script for 2D and 3D deep learning models, including model training, loss function selection, 
             and evaluation.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import os
import torch
import numpy as np
import pandas as pd
from utils.datasets import Custom3DDataSet, get_data_pairs
from models import init_model
from tqdm import tqdm
from utils.helpers import get_args
from utils.metrics import SegmentationEvaluator
from config import DEVICE
from utils.visualization import plot_box, plot_inferences

def test_model(test_dataset, model, pixel_dim, saving_dir = None):
    """
    This function evaluates a segmentation model on a test dataset.

    Args:
        test_dataset (Dataset): The test dataset, where each item is a tuple (image, label).
        model (torch.nn.Module): The segmentation model to be tested.
        saving_dir (str): The directory to save the results.
        pixel_dim (int): The pixel dimension used for calculating metrics like MCD and HD.
        to_save (bool): If True, the results will be saved as CSV files. Default is True.

    Returns:
        global_stats (pandas.DataFrame): A DataFrame containing global performance statistics for each class.
        image_stats (pandas.DataFrame): A DataFrame containing per-image statistics for each class.
    """
    model.eval()
    evaluator = SegmentationEvaluator([0, 1, 2])
    r_dc, r_hd, r_mcd = np.zeros((3, len(test_dataset))), np.zeros((3, len(test_dataset))), np.zeros((3, len(test_dataset)))
    fail = 0
    for i, (image, label) in enumerate(tqdm(test_dataset)):
        image       = image.to(DEVICE).unsqueeze(0)
        label       = label.numpy()
        pred        = model(image)
        pred        = torch.argmax(pred, dim = 1).squeeze().cpu().numpy()    
        dc, hd, mcd = evaluator.evaluate(label, pred, pixel_dim)
        dc_values   = np.array(list(dc.values()), dtype=float)
        hd_values   = np.array(list(hd.values()), dtype=float)
        mcd_values  = np.array(list(mcd.values()), dtype=float)
        
        if np.isinf(dc_values).any() or np.isinf(hd_values).any() or np.isinf(mcd_values).any():
            fail += 1 
        else:
            r_dc[:, i]  = np.array([dc[0], dc[1], dc[2]])
            r_hd[:, i]  = np.array([hd[0], hd[1], hd[2]])
            r_mcd[:, i] = np.array([mcd[0], mcd[1], mcd[2]])

    mean_dc, median_dc   = np.mean(r_dc, axis=1), np.median(r_dc, axis=1)
    mean_hd, median_hd   = np.mean(r_hd, axis=1), np.median(r_hd, axis=1)
    mean_mcd, median_mcd = np.mean(r_mcd, axis=1), np.median(r_mcd, axis=1)
    
    global_stats = pd.DataFrame({
        'Class': ['Background', 'Wall', 'Lumen'],
        'Mean_DC' :  mean_dc, 'Median_DC' : median_dc,
        'Mean_HD' :  mean_hd, 'Median_HD' : median_hd,
        'Mean_MCD': mean_mcd, 'Median_MCD': median_mcd,
        'Fail': fail
    })    

    image_stats = pd.DataFrame({
        'Image_Index': np.arange(len(test_dataset)),
        'DC_Class_0' :  r_dc[0],  'DC_Class_1':  r_dc[1], 'DC_Class_2': r_dc[2],
        'HD_Class_0' :  r_hd[0],  'HD_Class_1':  r_hd[1], 'HD_Class_2': r_hd[2],
        'MCD_Class_0': r_mcd[0], 'MCD_Class_1': r_mcd[1],'MCD_Class_2': r_mcd[2]
    })

    if saving_dir is not None:
        global_stats.to_csv(os.path.join(saving_dir, "global_statistics.csv"), index=False)
        image_stats.to_csv(os.path.join(saving_dir, "image_statistics.csv"), index=False)

    return global_stats, image_stats
    
def init_model_test(name_model, skip_connection_type, bottleneck_type, num_classes, slice_of_interest, 
                     filter_1, depth, dropout, n_slices, model_to_load):
    """
    Initialize the model for training.
    
    Args:
        name_model (str): Name of the model architecture.
        skip_connection_type (str): Type of skip connections.
        bottleneck_type (str): Type of bottleneck.
        num_classes (int): Number of classes in the dataset.
        slice_of_interest (int): Slice of interest for 3D data.
        filter_1 (int): Number of filters in the first layer.
        depth (int): Depth of the model.
        dropout (float): Dropout rate.
        n_slices (int): Number of input slices.
        model_to_load (str): Path to a pre-trained model.
    
    Returns:
        nn.Module: Initialized model.
    """
    print(f"{'='*5} Start loading model {'='*5}")
    model = init_model(name_model, skip_connection_type, bottleneck_type, num_classes, slice_of_interest, 
                       filter_1, depth, dropout, n_slices)
    model.to(device = DEVICE)

    if not os.path.exists(model_to_load):
        raise FileNotFoundError(f"Image file not found: {model_to_load}")
    else:
        loaded_checkpoint = torch.load(model_to_load, map_location=torch.device(DEVICE), weights_only=False)
        model.load_state_dict(loaded_checkpoint['model_state_dict'])
        model.checkpoint = loaded_checkpoint
    return model

def init_datasets_test(folder, slices, sides, slice_interest, num_classes, normalize):
    """
    Initialize training and validation datasets and loaders.
    
    Args:
        folder (str): Path to the dataset folder.
        slices (int): Number of slices for 3D data.
        sides (int): Number of sides for 3D data.
        slice_interest (int): Slice of interest for 3D data.
        num_classes (int): Number of classes in the dataset.
        batch_size (int): Batch size for training and validation.
        val_size (float): Proportion of validation data.
        data_augmentation (bool): Whether to apply data augmentation.
        normalize (str): Normalization type ('minmax' or 'zscore').
    
    Returns:
        tuple: Training and validation DataLoaders.
    """
    print(f"{'='*5} Start creating datasets {'='*5}")
    test_pairs = get_data_pairs(folder, slices, sides, slice_interest, num_classes)
    test_dataset = Custom3DDataSet(test_pairs, slice_interest, False, normalize)
    return test_dataset


if __name__ == '__main__':
    args = get_args()

    test_dataset = init_datasets_test(args.folder, args.slices, args.sides, args.slice_of_interest, 
                                      args.num_classes, args.normalize)

    model = init_model_test(args.name_model, args.skip_connection_type, args.bottleneck_type, 
                             args.num_classes, args.slice_of_interest, args.filter_1, args.depth, 
                             args.dropout, args.n_slices, args.model_to_load)
    
    print(f"{'='*5} Start testing model {'='*5}")
    print(f"     Model {model.checkpoint['name']} on {DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"     Number of parameters: {total_params}")
    print(f"     Size test: {len(test_dataset)}")
    
    saving_dir = os.path.join(args.saving_dir, model.checkpoint['name'])  
    os.makedirs(saving_dir, exist_ok=True)

    global_stats, image_stats = test_model(test_dataset, model, args.pixel_dim, saving_dir)
    
    plot_box(image_stats, saving_dir)

    worst_indices_class_1 = np.argsort(image_stats['DC_Class_1'])[:2]
    worst_indices_class_2 = np.argsort(image_stats['DC_Class_2'])[:2]
    worst_indices_class_3 = np.argsort(image_stats['HD_Class_1'])[-2:][::-1]
    worst_indices_class_4 = np.argsort(image_stats['HD_Class_2'])[-2:][::-1]
    worst_indices_class_5 = np.argsort(image_stats['MCD_Class_1'])[-2:][::-1]
    worst_indices_class_6 = np.argsort(image_stats['MCD_Class_2'])[-2:][::-1]

    best_indices_class_1 = np.argsort(image_stats['DC_Class_1'])[-2:][::-1]
    best_indices_class_2 = np.argsort(image_stats['DC_Class_2'])[-2:][::-1]
    best_indices_class_3 = np.argsort(image_stats['HD_Class_1'])[:2]
    best_indices_class_4 = np.argsort(image_stats['HD_Class_2'])[:2]
    best_indices_class_5 = np.argsort(image_stats['MCD_Class_1'])[:2]
    best_indices_class_6 = np.argsort(image_stats['MCD_Class_2'])[:2]

    worst_indices = np.unique(np.concatenate((worst_indices_class_1, worst_indices_class_2,
                                              worst_indices_class_3, worst_indices_class_4,
                                              worst_indices_class_5, worst_indices_class_6)))
    best_indices = np.unique(np.concatenate((best_indices_class_1, best_indices_class_2,
                                             best_indices_class_3, best_indices_class_4,
                                             best_indices_class_5, best_indices_class_6)))
    plot_inferences(test_dataset, model, saving_dir, args.slice_interest,  
                    title = "best_inferences.png", list_ind = best_indices)
    plot_inferences(test_dataset, model, saving_dir, args.slice_interest,  
                    title = "worst_inferences.png", list_ind = worst_indices)
"""
File: scripts/cross_validation.py
Description: Perform a cross validation training
Author: Kevin Ferreira
Date: 18 December 2024
"""

import os
from utils.helpers import get_args
from utils.datasets import Custom3DDataSet, get_data_splits_cross_val
from torch.utils.data import DataLoader
from config import DEVICE, SEED 
import pandas as pd
from .train import *
from .test_model import *

def cross_validation(nb_cross_val, data_pairs, slice_of_interest, data_augmentation, normalize, batch_size,
                     name_model, skip_connection_type, bottleneck_type, num_classes, 
                     filter_1, depth, dropout, n_slices, epochs, learning_rate, weight_decay, milestones, 
                     gamma, criterion, pixel_dim, saving_dir, n_test, model_to_load):
    """
    Function to perform cross-validation training on the provided dataset.

    Args:
        nb_cross_val (int): Number of cross-validation folds.
        data_pairs (list): List containing pairs of train/test data for each fold.
        slice_of_interest (int): Slice index to focus on during training.
        data_augmentation (bool): Flag to apply data augmentation.
        normalize (bool): Flag to normalize the data.
        batch_size (int): Size of each batch during training.
        name_model (str): The name of the model to use.
        skip_connection_type (str): Type of skip connection to use in the model.
        bottleneck_type (str): Type of bottleneck to use in the model.
        num_classes (int): Number of classes for classification.
        filter_1 (int): Filter size for the first convolutional layer.
        depth (int): Depth of the model.
        dropout (float): Dropout rate.
        n_slices (int): Number of slices to be used.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        milestones (list): List of milestones for the learning rate scheduler.
        gamma (float): Gamma for the learning rate scheduler.
        criterion (str): Loss function to use.
        pixel_dim (tuple): Dimensions of the pixels (for evaluation).
        saving_dir (str): Directory to save the model and results.
        n_test (int): The test number for saving model results.
        model_to_load (string): path of the model to load

    Returns:
        pd.DataFrame: A concatenated dataframe of all cross-validation results.
        str: Path where results are saved.
    """
    global_stats_concat = pd.DataFrame({})
    for i in range(nb_cross_val):
        train_loader, test_dataset = init_dataset_i(i, data_pairs, slice_of_interest, data_augmentation, normalize, batch_size)
        model = init_model_i(i, name_model, skip_connection_type, bottleneck_type, num_classes, slice_of_interest, 
                             filter_1, depth, dropout, n_slices, [learning_rate, 0, batch_size], model_to_load)
        print(f"{'='*5} Start training model {i} {'='*5}")
        print(f"     Model {model.checkpoint['name']} on {DEVICE}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"     Number of parameters: {total_params}")
        print(f"     Size train: {len(train_loader)}")
        print(f"     Size test: {len(test_dataset)}")
        train_losses, _ = train_model(model, train_loader, None, epochs, 
                                      learning_rate, weight_decay, milestones, 
                                      gamma, saving_dir, criterion_name = criterion)

        global_stats, _ = test_model(test_dataset, model, pixel_dim = pixel_dim, saving_dir = None)
        global_stats_concat = pd.concat([global_stats_concat, global_stats], ignore_index=True)

        saving_p = os.path.join(saving_dir, model.checkpoint['name'], f"params_{n_test}")  
        os.makedirs(saving_p, exist_ok=True)
        global_stats_concat.to_csv(os.path.join(saving_p, f"global_statistics.csv"), index=False)
        model.checkpoint['model_state_dict'] = model.state_dict()
        model.checkpoint['results']['train_losses'] = train_losses
        torch.save(model.checkpoint, os.path.join(saving_p, f"model_{i}.pt"))

    return global_stats_concat, saving_p

def init_dataset_i(i, data_pairs, slice_of_interest, data_augmentation, normalize, batch_size):
    print(f"{'='*5} Start loading dataset {i} {'='*5}")
    train_pairs  = data_pairs[i][0]
    train_dataset = Custom3DDataSet(train_pairs, slice_of_interest, data_augmentation, normalize)
    train_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    
    test_pairs    = data_pairs[i][1]
    test_dataset  = Custom3DDataSet(test_pairs, slice_of_interest, False, normalize)
    return train_loader, test_dataset

def init_model_i(i, name_model, skip_connection_type, bottleneck_type, num_classes, slice_of_interest, 
                 filter_1, depth, dropout, n_slices, params, model_to_load = None):
    print(f"{'='*5} Start creating model {i} {'='*5}")
    init_seed(SEED)
    model = init_model(name_model, skip_connection_type, bottleneck_type, num_classes, slice_of_interest, 
                       filter_1, depth, dropout, n_slices)
    model.to(device = DEVICE)
    if model_to_load is not None:
        model_i_path = os.path.join(model_to_load, f"model_{i}.pt")  
        if not os.path.exists(model_i_path):
            raise FileNotFoundError(f"Image file not found: {model_i_path}")
        else:
            loaded_checkpoint = torch.load(model_i_path, map_location=torch.device(DEVICE), weights_only=False)
            model.load_state_dict(loaded_checkpoint['model_state_dict'])
            model.checkpoint = loaded_checkpoint
            print(f"Model loaded from {model_i_path}")
    else:
        model.checkpoint['LR'] = params[0]
        model.checkpoint['VAL_SIZE'] = params[1]
        model.checkpoint['BATCH_SIZE'] = params[2]
    return model

def save_results(global_stats_concat, saving_dir, args):
    background_rows = global_stats_concat[global_stats_concat["Class"] == "Background"]
    background_mean = background_rows[["Mean_DC", "Median_DC", "Mean_HD", "Median_HD", "Mean_MCD", "Median_MCD", "Fail"]].mean()
    lumen_rows = global_stats_concat[global_stats_concat["Class"] == "Lumen"]
    lumen_mean = lumen_rows[["Mean_DC", "Median_DC", "Mean_HD", "Median_HD", "Mean_MCD", "Median_MCD", "Fail"]].mean()
    wall_rows = global_stats_concat[global_stats_concat["Class"] == "Wall"]
    wall_mean = wall_rows[["Mean_DC", "Median_DC", "Mean_HD", "Median_HD", "Mean_MCD", "Median_MCD", "Fail"]].mean()

    globals_rows =  pd.DataFrame({
        'Class': ['Background (Mean)','Wall (Mean)','Lumen (Mean)'],
        'Mean_DC': [background_mean['Mean_DC'],wall_mean['Mean_DC'],lumen_mean['Mean_DC']],
        'Median_DC': [background_mean['Median_DC'],wall_mean['Median_DC'],lumen_mean['Median_DC']],
        'Mean_HD': [background_mean['Mean_HD'],wall_mean['Mean_HD'],lumen_mean['Mean_HD']],
        'Median_HD': [background_mean['Median_HD'],wall_mean['Median_HD'],lumen_mean['Median_HD']],
        'Mean_MCD': [background_mean['Mean_MCD'],wall_mean['Mean_MCD'],lumen_mean['Mean_MCD']],
        'Median_MCD': [background_mean['Median_MCD'],wall_mean['Median_MCD'],lumen_mean['Median_MCD']],
        'Fail': [background_mean['Fail'], wall_mean['Fail'], lumen_mean['Fail']]
    })
    global_stats_concat = pd.concat([global_stats_concat, globals_rows], ignore_index=True)
    global_stats_concat.to_csv(os.path.join(saving_dir, f"global_statistics.csv"), index=False)
    
    with open(os.path.join(saving_dir, f"params.txt"), 'w') as f:
        f.write("Params tested:\n")
        f.write(f"Dataset: {args.folder}\n")
        f.write(f"Nb slices: {args.n_slices}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Filter: {args.filter_1}\n")
        f.write(f"Depth: {args.depth}\n")
        f.write(f"Skip Connection: Trainable Weighted Mean\n")
        f.write(f"Data Augmentation: {args.data_augmentation}\n")
        f.write(f"Learning Rate: Scheduler starting at {args.lr}\n")
        f.write(f"Weight decay: {args.wd}\n")
        f.write(f"Batch size: {args.batch_size}\n")


if __name__ == '__main__':
    args = get_args()
    data_pairs = get_data_splits_cross_val(args.folder, args.slices, args.sides, args.nb_cross_val, SEED, is_test = False)
    
    global_stats_concat, saving_dir = cross_validation(args.nb_cross_val, data_pairs, args.slice_of_interest, args.data_augmentation, args.normalize, 
                                                       args.batch_size, args.name_model, args.skip_connection_type, args.bottleneck_type, 
                                                       args.num_classes, args.filter_1, args.depth, args.dropout, args.n_slices, 
                                                       args.epochs, args.learning_rate, args.weight_decay, args.milestones, 
                                                       args.gamma, args.criterion, args.pixel_dim, args.saving_dir, args.n_test, args.model_to_load)

    save_results(global_stats_concat, saving_dir, args)
    

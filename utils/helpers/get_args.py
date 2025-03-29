"""
File: helpers/get_args.py
Description: Helper function to get the arguments for the scripts
Author: Kevin Ferreira
Date: 18 December 2024
"""

import argparse
import config

def get_args():
    """
    Parse command-line arguments.
    
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Training script.")
    
    # Dataset and training parameters
    parser.add_argument('-f', '--folder', default=config.DATA_FOLDER, type=str, 
                        help="Path to the dataset folder.")
    parser.add_argument('-sf', '--saving_dir', default=config.SAVING_FOLDER, type=str, 
                        help="Directory to save models and plots.")
    parser.add_argument('-s', '--slices', default=config.SLICES, nargs='+', type=int, 
                        help="Number of slices per volume.")
    parser.add_argument('-sd', '--sides', default=config.SIDES, type=int, 
                        help="Number of sides for 3D slices.")
    parser.add_argument('-si', '--slice_of_interest', default=config.SLICE_OF_INTEREST, nargs='+', type=int, 
                        help="Slice index of interest.")
    parser.add_argument('-nc', '--num_classes', default=config.NUM_CLASSES, type=int, 
                        help="Number of output classes.")
    parser.add_argument('-b', '--batch_size', default=config.BATCH_SIZE, type=int, 
                        help="Batch size for training.")
    parser.add_argument('-v', '--val_size', default=config.VAL_SIZE, type=float, 
                        help="Validation set proportion. (0 to 1)")
    parser.add_argument('-da', '--data_augmentation', default=config.DATA_AUGMENTATION, action='store_true', 
                        help="Enable data augmentation. (True or False)")
    parser.add_argument('-no', '--normalize', default=config.NORMALIZATION, 
                        help="Normalization type of the data. (None, minmax or zscore)")
    parser.add_argument('-pd', '--pixel_dim', default=config.PIXEL_DIM, type=float, 
                        help="Pixel dimension of the images")
    
    # Model parameters
    parser.add_argument('-nm', '--name_model', default=config.NAME_MODEL, type=str, 
                        help="Model name. (unet_2d, unet_3d, unet_2_5d or unet_3d_to_2d)")
    parser.add_argument('-sc', '--skip_connection_type', default=config.SKIP_CONNECTION_TYPE, type=str, 
                        help="Type of skip connections. (none, attention_softmax, attention_sigmoid, mean or mid)")
    parser.add_argument('-bt', '--bottleneck_type', default=config.BOTTLENECK_TYPE, type=str, 
                        help="Type of bottleneck. (none, attention or transformer)")
    parser.add_argument('-f1', '--filter_1', default=config.FILTER_1, type=int, 
                        help="Number of filters in the first layer.")
    parser.add_argument('-d', '--depth', default=config.DEPTH, type=int, 
                        help="Depth of the model.")
    parser.add_argument('-dr', '--dropout', default=config.DROPOUT, nargs='+', type=float, 
                        help="Dropout rate. (list of int)")
    parser.add_argument('-ns', '--n_slices', default=config.N_SLICES, type=int, 
                        help="Number of slices to process.")
    parser.add_argument('-ml', '--model_to_load', default=config.MODEL_TO_LOAD, type=str, 
                        help="Path to the pre-trained model checkpoint to load.")

    # Training parameters
    parser.add_argument('-e', '--epochs', default=config.EPOCHS, type=int, 
                        help="Number of training epochs.")
    parser.add_argument('-lr', '--learning_rate', default=config.LEARNING_RATE, type=float, 
                        help="Initial learning rate.")
    parser.add_argument('-wd', '--weight_decay', default=config.WEIGHT_DECAY, type=float, 
                        help="Weight decay (L2 regularization).")
    parser.add_argument('-m', '--milestones', default=config.MILESTONES, nargs='+', type=int, 
                        help="List of epoch milestones for learning rate decay.")
    parser.add_argument('-g', '--gamma', default=config.GAMMA, type=float, 
                        help="Gamma value for learning rate decay.")
    parser.add_argument('-cr', '--criterion', default=config.CRITERION, type=str, 
                        help="Loss function to use. (cross_entropy, dice_loss, focal_loss, mix_loss, dice_topo_loss, ce_topo_loss)")
    parser.add_argument('-ncv', '--nb_cross_val', default=config.N_CROSS_VAL, type=int, 
                        help="Number of cross validation splits.")
    parser.add_argument('-nt', '--n_test', default=1000, type=int, 
                        help="Number of cross validation splits.")
    return parser.parse_args()
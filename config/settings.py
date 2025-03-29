"""
File: config/settings.py
Description: Settings of the project
Author: Kevin Ferreira
Date: 18 December 2024
"""
import torch

# General Configuration
PROJECT_NAME = "ThesisProject"
VERSION = "1.0"
DEBUG = False

# Dataset parameters
DATA_FOLDER       = '/home/kefe11/ThesisProject/data/Dataset_25mm_128p_17s'
SAVING_FOLDER     = '/home/kefe11/ThesisProject/results'
SLICES            = [f'Slice{i}' for i in range(1, 9)]
SIDES             = ['LEFT', 'RIGHT']
SLICE_OF_INTEREST = 8
NUM_CLASSES       = 3
BATCH_SIZE        = 32
VAL_SIZE          = 0.15
DATA_AUGMENTATION = True
NORMALIZATION     = "minmax"
PIXEL_DIM          = 25/128 #nib.load(file).header["pixdim"][1:3]

# Model Configuration
NAME_MODEL           = 'unet_3d_to_2d'
SKIP_CONNECTION_TYPE = 'mid'
BOTTLENECK_TYPE      = 'transformer'
MODEL_TO_LOAD        = None
DROPOUT              = [0.1, 0.3]
FILTER_1             = 32
DEPTH                = 6
N_SLICES             = 17

# Training Hyperparameters
EPOCHS        = 300              
LEARNING_RATE = 0.01
WEIGHT_DECAY  = 1e-5 
MILESTONES    = [20, 60, 120, 200]
GAMMA         = 0.1
CRITERION     = "cross_entropy" 
N_CROSS_VAL   = 5

# Resource Configuration
SEED   = 14
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() 
                       else "cpu")







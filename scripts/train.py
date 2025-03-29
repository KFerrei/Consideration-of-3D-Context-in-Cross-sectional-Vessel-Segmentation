"""
File: scripts/train.py
Description: Training script for 2D and 3D deep learning models, including model training, loss function selection, 
             and evaluation.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import os
import torch
from utils.losses import init_loss
from utils.datasets import Custom3DDataSet, get_data_pairs
from models import init_model
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.helpers import get_args
from config import DEVICE, SEED 

def train_model_epoch(train_loader, model, optimizer, criterion):
    """
    Train the model for one epoch.
    
    Args:
        train_loader (DataLoader): DataLoader for the training data.
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (nn.Module): Loss function.
    
    Returns:
        float: Average training loss for the epoch.
    """
    total_loss = 0.0
    model.train()
    for images, labels in tqdm(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        preds  = model(images)
        loss   = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def init_training(model, learning_rate, weight_decay, milestones, gamma, criterion_name):
    """
    Initialize the optimizer, learning rate scheduler, and loss function.
    
    Args:
        model (nn.Module): Model to be trained.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for regularization.
        milestones (list): Epochs at which to reduce learning rate.
        gamma (float): Learning rate reduction factor.
        criterion_name (str): Name of the loss function.
    
    Returns:
        tuple: Optimizer, learning rate scheduler, criterion, and best validation loss.
    """
    optimizer    = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma)
    criterion    = init_loss(criterion_name)

    if len(model.checkpoint['results']['val_losses'])>0:
        best_val_loss = model.checkpoint['results']['val_losses'][-1]
        print(f"\033[1m\033[92m Epoch {0} Train Loss {model.checkpoint['results']['train_losses'][-1]} Val Loss {best_val_loss}\033[0m") 
    else:
        best_val_loss = 100
    return optimizer, lr_scheduler, criterion, best_val_loss

def train_model(model, train_loader, val_loader, 
                epochs, learning_rate, weight_decay, milestones, gamma, 
                saving_dir, criterion_name):
    """
    Train the model across multiple epochs and save the best model.
    
    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for regularization.
        milestones (list): Epochs at which to reduce learning rate.
        gamma (float): Learning rate reduction factor.
        saving_dir (str): Directory to save the trained model and plots.
        criterion_name (str): Name of the loss function.
    
    Returns:
        tuple: Training and validation losses per epoch.
    """
    optimizer, lr_scheduler, criterion, best_val_loss = init_training(model, learning_rate, weight_decay, 
                                                                      milestones, gamma, criterion_name)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        train_loss = train_model_epoch(train_loader, model, optimizer, criterion)
        lr_scheduler.step()
        train_losses.append(train_loss)
        if val_loader is not None:
            val_loss = eval_model(val_loader, model, criterion)
            val_losses.append(val_loss)
            print(f"\033[1m\033[92m Epoch {epoch+1}/{epochs} Train Loss {train_loss} Val Loss {val_loss}\033[0m") 
            if val_loss < best_val_loss:
                # Save the best model
                print("Saving model")
                model.checkpoint['model_state_dict'] = model.state_dict()
                model.checkpoint['EPOCHS'] = epoch
                model.checkpoint['results']['train_losses'] = train_losses
                model.checkpoint['results']['val_losses'] = val_losses
                torch.save(model.checkpoint, os.path.join(saving_dir, "best_model.pt"))
                plot_loss(train_losses, val_losses, os.path.join(saving_dir, "losses.png") )
                best_val_loss = val_loss
        else:
            print(f"\033[1m\033[92m Epoch {epoch+1}/{epochs} Train Loss {train_loss}\033[0m")

        if hasattr(criterion, 'compute_topo') and epoch==200:
            print("Adding topological loss")
            criterion.compute_topo = True
    return train_losses, val_losses

def plot_loss(train_losses, val_losses, title):
    """
    Plot training and validation losses and save the plot.
    
    Args:
        train_losses (list): Training losses per epoch.
        val_losses (list): Validation losses per epoch.
        title (str): File path to save the plot.
    """
    plt.plot(train_losses, label = 'Train')
    plt.plot(val_losses, label = 'Val')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Cross Entropy Loss {model.checkpoint['name']}")
    plt.savefig(title)
    plt.close()

def init_seed(seed_value):
    """
    Initialize random seed for reproducibility.
    
    Args:
        seed_value (int): Seed value.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_model(val_loader, model, criterion):
    """
    Evaluate the model on validation data.
    
    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): Model to evaluate.
        criterion (nn.Module): Loss function.
    
    Returns:
        float: Average validation loss.
    """
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images3D, labels in tqdm(val_loader):
            images3D = images3D.to(DEVICE)
            labels = labels.to(DEVICE)
            preds = model(images3D)
            loss = criterion(preds, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def init_datasets_train(folder, slices, sides, slice_interest, num_classes, batch_size, val_size, data_augmentation, normalize):
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
    data_pairs = get_data_pairs(folder, slices, sides, slice_interest, num_classes)
    generator  = torch.Generator().manual_seed(SEED)
    
    train_pairs, val_pairs = random_split(data_pairs, [1-val_size, val_size], 
                                          generator=generator)
    train_dataset = Custom3DDataSet(train_pairs, slice_interest, data_augmentation, normalize)
    val_dataset   = Custom3DDataSet(val_pairs, slice_interest, False, normalize)
    train_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size = batch_size)
    return train_loader, val_loader

def init_model_train(name_model, skip_connection_type, bottleneck_type, num_classes, slice_of_interest, 
                     filter_1, depth, dropout, n_slices, params, model_to_load):
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
        params (list): Training parameters such as learning rate, validation size, and batch size.
        model_to_load (str): Path to a pre-trained model.
    
    Returns:
        nn.Module: Initialized model.
    """
    print(f"{'='*5} Start creating model {'='*5}")
    model = init_model(name_model, skip_connection_type, bottleneck_type, num_classes, slice_of_interest, 
                       filter_1, depth, dropout, n_slices)
    model.to(device = DEVICE)

    if model_to_load is not None:
        if not os.path.exists(model_to_load):
            raise FileNotFoundError(f"Image file not found: {model_to_load}")
        else:
            loaded_checkpoint = torch.load(args.model_path, map_location=torch.device(DEVICE), weights_only=False)
            model.load_state_dict(loaded_checkpoint['model_state_dict'])
            model.checkpoint = loaded_checkpoint
    else:
        model.checkpoint['LR'] = params[0]
        model.checkpoint['VAL_SIZE'] = params[1]
        model.checkpoint['BATCH_SIZE'] = params[2]
    return model

if __name__ == '__main__':
    args = get_args()
    init_seed(SEED)

    # Initialize datasets and loaders
    train_loader, val_loader = init_datasets_train(
        args.folder, args.slices, args.sides, args.slice_of_interest, args.num_classes,
        args.batch_size, args.val_size, args.data_augmentation, args.normalize
    )

    # Initialize model to train 
    model = init_model_train(args.name_model, args.skip_connection_type, args.bottleneck_type, 
                             args.num_classes, args.slice_of_interest, args.filter_1, args.depth, 
                             args.dropout, args.n_slices, [args.learning_rate, args.val_size, args.batch_size], 
                             args.model_to_load)
    
    print(f"{'='*5} Start training model {'='*5}")
    print(f"     Model {model.checkpoint['name']} on {DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"     Number of parameters: {total_params}")
    print(f"     Size train: {len(train_loader)}")
    print(f"     Size val: {len(val_loader)}")

    saving_dir = os.path.join(args.saving_dir, model.checkpoint['name'])  
    os.makedirs(saving_dir, exist_ok=True)

    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                           args.epochs, args.learning_rate, args.weight_decay, args.milestones, args.gamma, 
                                           saving_dir, criterion_name =args.criterion)
    plot_loss(train_losses, val_losses, os.path.join(saving_dir, "losses.png"))  


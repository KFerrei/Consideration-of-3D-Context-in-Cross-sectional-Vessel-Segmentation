"""
File: utils/datasets/custom_dataset.py
Description: Custom Dataset for loading 3D medical images, applying preprocessing, 
             augmentations, and generating train/validation splits.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import os
import nibabel as nib
import torch
import torchio as tio
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

def normalize_min_max(image):
    """
    Normalize the image using Min-Max scaling.

    Args:
        image (np.ndarray or torch.Tensor): Input image.

    Returns:
        torch.Tensor: Normalized image.
    """
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)
    return image

def normalize_zscore(image):
    """
    Normalize the image using Z-score normalization.

    Args:
        image (np.ndarray or torch.Tensor): Input image.

    Returns:
        torch.Tensor: Normalized image.
    """
    mean = image.mean()
    std = image.std()
    image = (image - mean) / std
    return image

class Custom3DDataSet(Dataset):
    """
    Custom Dataset for loading 3D medical images and labels, with optional augmentations 
    and normalization.

    Args:
        files (list): List of dictionaries containing paths to images and labels.
        slice_of_interest (int): The slice index of interest.
        data_augmentation (bool): Whether to apply data augmentation.
        is_test (bool): Whether the dataset is for testing (affects output structure).
        normalize (str): Normalization type ('minmax' or 'zscore').
    """
    def __init__(self, files, slice_of_interest, data_augmentation = True, normalize = None, is_test = False):
        
        # Check that 'normalize' is either None or a valid option
        if normalize is not None:
            assert normalize in ["minmax", "zscore"], \
                "'normalize' must be 'minmax', 'zscore', or None."
            
        self.files             = files
        self.slice_of_interest = slice_of_interest
        self.data_augmentation = data_augmentation
        self.is_test           = is_test
        self.normalize         = normalize
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Load and process an image-label pair.

        Args:
            idx (int): Index of the data pair.

        Returns:
            tuple: (image, label) for training/validation,
                   (image, label, filename) for testing.
        """
        # Check that the index is valid
        assert 0 <= idx <= len(self.files), f"Index {idx} is out of range for 'files'."
        
        # Check that the file paths exist
        image_path = self.files[idx]["image"]
        label_path = self.files[idx]["label"]
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' does not exist.")
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label file '{label_path}' does not exist.")
    
        # Load image and label
        image_path = self.files[idx]["image"]        
        image = nib.load(image_path)
        image = image.get_fdata()
        image = torch.tensor(image, dtype =torch.float32).squeeze().unsqueeze(0)
        
        label_path = self.files[idx]["label"]
        label = nib.load(label_path)
        label = label.get_fdata()
        label = torch.tensor(label, dtype =torch.int64).squeeze().unsqueeze(0)
        
        if self.data_augmentation:
            # Apply augmentations
            transformations_space = tio.OneOf([
                tio.RandomFlip(axes=0, flip_probability=0.5),
                tio.RandomAffine(scales=[0.1, 0.1, 0], degrees=[0, 0, 60], translation=[5, 5, 0], p=0.75),
            ])

            transformations_intensity = tio.OneOf([
                tio.RandomMotion(degrees = 10, translation = 5, num_transforms = 2, p=0.5),
                tio.RandomGhosting(num_ghosts = 3, axes = (1, 2), intensity = 0.4, p=0.5),
                tio.RandomBiasField(coefficients = 0.3, order = 2, p=0.5),
                tio.RandomNoise(mean = 0, std = 5, p=0.5)
            ])
            
            subject = tio.Subject(image = tio.ScalarImage(tensor=image), 
                                  label = tio.ScalarImage(tensor=label))
            subject_transform = transformations_space(subject)
            image = subject_transform.image.data
            image = transformations_intensity(image)
            label = subject_transform.label.data
            
        label    = label[0,:,:, self.slice_of_interest].long()
        
        # Normalize if specified
        if self.normalize == "zscore":
            image = normalize_zscore(image)
        elif self.normalize == "minmax":
            image = normalize_min_max(image)
        if self.is_test:
            return image, label, image_path.split('/')[-1]
        return image, label

def get_data_pairs(working_folder, slices, sides, slice_of_interest, num_classes, is_test = False):
    """
    Generate data pairs for training/validation/testing.

    Args:
        working_folder (str): Path to the dataset folder.
        slices (list): List of slice indices of interest.
        sides (list): List of sides of interest ('LEFT', 'RIGHT').
        slice_of_interest (int): Slice index of interest.
        num_classes (int): Number of segmentation classes.
        is_test (bool): Whether the data is for testing.

    Returns:
        list: List of dictionaries containing image and label/prediction paths.
    """
    assert os.path.isdir(working_folder), f"The directory '{working_folder}' does not exist."

    data_pairs = []  

    if is_test:
        images_folder = os.path.join(working_folder, "test/images")
        predictions_folder  = os.path.join(working_folder,  "test/predictions")
    else:
        images_folder = os.path.join(working_folder, "train/images")
        masks_folder  = os.path.join(working_folder,  "train/masks")
    
    # Ensure image and mask directories exist
    assert os.path.isdir(images_folder), f"Images folder '{images_folder}' does not exist."
    if not is_test:
        assert os.path.isdir(masks_folder), f"Masks folder '{masks_folder}' does not exist."
    
    
    for filename in os.listdir(images_folder):
        try:
            _, _, slice_i, side = filename.split('_')
        except ValueError:
            raise ValueError(f"Filename '{filename}' does not follow the expected format.")

        if slice_i in slices and side.split('.')[0] in sides:
            image_path = os.path.join(images_folder, filename)
            if is_test:
                prediction_path = os.path.join(predictions_folder, filename)
                data_pairs.append({"image": image_path, "prediction": prediction_path})
            else:
                mask_path  = os.path.join(masks_folder,  filename)
                if os.path.isfile(image_path) and os.path.isfile(mask_path):
                    data_pairs.append({"image": image_path, "label": mask_path})
                else:
                    raise FileNotFoundError(f"Missing file: {image_path} or {mask_path}")
    return data_pairs

def get_data_splits_cross_val(working_folder, slices, sides, n_splits, SEED, is_test = False):
    """
    Create cross-validation splits based on patient IDs.

    Args:
        working_folder (str): Path to the dataset folder.
        slices (list): List of slice indices of interest.
        sides (list): List of sides of interest ('LEFT', 'RIGHT').
        n_splits (int): Number of cross-validation splits.
        SEED (int): Random seed for reproducibility.
        is_test (bool): Whether the data is for testing.

    Returns:
        list: List of train/validation splits, each as a tuple of data pairs.
    """
    assert os.path.isdir(working_folder), f"The directory '{working_folder}' does not exist."
    
    # Validate the number of splits
    assert n_splits > 1, "'n_splits' must be greater than 1."
    
    if is_test:
        images_folder = os.path.join(working_folder, "test/images")
        predictions_folder  = os.path.join(working_folder,  "test/predictions")
    else:
        images_folder = os.path.join(working_folder, "train/images")
        masks_folder  = os.path.join(working_folder,  "train/masks")
    
    # Ensure image and mask directories exist
    assert os.path.isdir(images_folder), f"Images folder '{images_folder}' does not exist."
    if not is_test:
        assert os.path.isdir(masks_folder), f"Masks folder '{masks_folder}' does not exist."
    
    # Step 1: Group data by patient ID
    patient_data = {}
    for filename in os.listdir(images_folder):
        id_p, exam_i, slice_i, side = filename.split('_')
        if slice_i in slices and side.split('.')[0] in sides:
            image_path = os.path.join(images_folder, filename)
            if is_test:
                prediction_path = os.path.join(predictions_folder, filename)
                if id_p not in patient_data:
                    patient_data[id_p] = []
                patient_data[id_p].append({"image": image_path, "prediction": prediction_path})
            else:
                mask_path = os.path.join(masks_folder, filename)
                if os.path.isfile(image_path) and os.path.isfile(mask_path):
                    if id_p not in patient_data:
                        patient_data[id_p] = []
                    patient_data[id_p].append({"image": image_path, "label": mask_path})
    
    if not patient_data:
        raise ValueError("No data found for the specified parameters.")
    # Step 2: Create patient IDs list and initialize KFold
    patient_ids = list(patient_data.keys())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    print(kf)

    # Step 3: Split into folds
    data_splits = []
    for train_index, val_index in kf.split(patient_ids):
        train_data, val_data = [], []
        for idx in train_index:
            train_data.extend(patient_data[patient_ids[idx]])
        for idx in val_index:
            val_data.extend(patient_data[patient_ids[idx]])
        data_splits.append((train_data, val_data))
    return data_splits
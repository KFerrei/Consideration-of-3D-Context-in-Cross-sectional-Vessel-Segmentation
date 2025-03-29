"""
File: tests/test_topo_loss.py
Description:  Test the losses implementation
Author: Kevin Ferreira
Date: 8 January 2025
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import time
from utils.losses import init_loss
from models import init_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.losses.topological_loss import TopologicalLoss

PATH = "/home/kefe11/ThesisProject/data/Dataset_Test"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVING_PATH = '/home/kefe11/ThesisProject/results/figures/topo_loss'
###################### LOAD IMAGES & LABEL ######################

def load_image(image_path, is_label = False):
    image = np.array(Image.open(image_path))
    nx, ny = image.shape[0], image.shape[1]
    new_image = np.zeros((nx, ny)) if is_label else np.zeros((nx, ny, 3))
    for i in range(nx):
        for j in range(ny):
            if np.mean(image[i, j]) >= 170:
                new_image[i, j] = 2 if is_label else [0, 0, 100]
            elif np.mean(image[i, j]) >= 85:
                new_image[i, j] = 1 if is_label else [0, 100, 0]
            else:
                new_image[i, j] = 0 if is_label else [100, 0, 0]        
    if is_label:
        new_image = torch.from_numpy(new_image).long()
    else: 
        new_image = torch.from_numpy(new_image).permute(2, 0, 1)  
        new_image.requires_grad = True
    new_image = new_image.to(DEVICE)
    return new_image

images = []
n_image = 5
image_path = PATH + f'/{n_image}.png'
image = load_image(image_path)
images.append(image)
batch_image = torch.stack(images)
labels = [load_image(f"{PATH}/{n_image}.png", is_label = True) for i in range(1,7)]
batch_label = torch.stack(labels)

max_iter = 5
criterion    = TopologicalLoss(max_iter=max_iter, kernel_size=5, reduction="else")
losses, barcodes_pred, imagesTransf   = criterion(batch_image[0].unsqueeze(0), batch_label[0].unsqueeze(0))
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
for b, bb in barcodes_pred.items():
    for c, bc in bb.items():
        axs[c].set_title(f"Barcode classe {c}")
        for i_c, comp in bc.items():
            axs[c].plot([i - max_iter//2 for i in range(max_iter)], comp.detach().cpu().numpy(),label = f"Comp {i_c}")
            axs[c].legend()
            axs[c].set_xlabel("Filtration Step")
            axs[c].set_ylabel("Size")
plt.tight_layout()
plt.savefig(f'{SAVING_PATH}/barcodes_maxpool.png')
plt.close()

fig, axs = plt.subplots(3, max_iter, figsize=(10, 5))
for i in range(max_iter):
    im = imagesTransf[i]
    im = im.squeeze(0).cpu().numpy()  
    for j in range(3):
        axs[j, i].imshow(im[j], cmap='gray')  
        axs[j, i].set_title(f"Iteration {i+1}")
        axs[j, i].axis('off') 
plt.tight_layout()
plt.savefig(f'{SAVING_PATH}/images_barcode.png')
plt.close()



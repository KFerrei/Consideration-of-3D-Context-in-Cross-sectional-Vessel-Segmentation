"""
File: tests/test_losses.py
Description:  Test the losses implementation
Author: Kevin Ferreira
Date: 18 December 2024
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import time
from utils.losses import init_loss

PATH = "/home/kefe11/ThesisProject/data/Dataset_Test"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVING_PATH = '/home/kefe11/ThesisProject/results/figures'
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
n_image = 6
for i in range(n_image):
    image_path = PATH + f'/{i+1}.png'
    image = load_image(image_path)
    images.append(image)
batch_image = torch.stack(images)
labels = [load_image(f"{PATH}/1.png", is_label = True) for i in range(1,7)]
batch_label = torch.stack(labels)

######################### COMPUTE LOSS ##########################
# losses_names = ['cross_entropy', 'dice_loss', 'focal_loss', 'mix_loss', 'topo_loss', 'dice_topo_loss', 'ce_topo_loss', 'ce_dice_loss']
# results = ["" for _ in range(n_image)]
# for i in range(n_image):
#     for loss_name in losses_names:
#         loss    = init_loss(loss_name, reduction = "mean")
#         if hasattr(loss, 'compute_topo') :
#             print("Adding topological loss")
#             loss.compute_topo = True
#         t_start = time.time()
#         l       = loss(batch_image[i].unsqueeze(0), batch_label[i].unsqueeze(0))
#         t_end   = time.time()
#         print(f"{loss_name}: {l.item()} in {t_end-t_start:.4f} s")
#         results[i] = results[i] + f"{loss_name}: {l.item():.2f} in {t_end-t_start:.2f} s\n"

# fig, axs = plt.subplots(1, n_image, figsize=(25, 8))
# for i in range(n_image):
#     image_to_plot = batch_image[i, 1, :, :] * 0.5 + batch_image[i, 2, :, :]
#     axs[i].imshow(image_to_plot.detach().cpu().numpy(), cmap='gray')
#     axs[i].axis('off')
#     axs[i].set_title(results[i])
# plt.tight_layout()
# plt.savefig(f'{SAVING_PATH}/test_losses.png')
# plt.close()

# ##################### PLOT BARCODES ######################
# topo_loss    = init_loss('topo_loss', reduction = "mean")
# l  = topo_loss(batch_image, batch_label)
# max_iter = 10
# print(f"Topo Loss: {l}")
# l.backward()

topo_loss    = init_loss('topo_loss_paper', reduction = "mean")
l  = topo_loss(batch_image, batch_label)
max_iter = 10
print(f"Topo Loss: {l}")
l.backward()

# topo_loss    = init_loss('topo_loss', reduction = "all")
# l, barcodes  = topo_loss(batch_image, batch_label)
# max_iter = 5
# fig, axs = plt.subplots(6, 4, figsize=(10, 15), gridspec_kw={'width_ratios': [1, 3, 3, 3]})
# for i in range(n_image):
#     image_to_plot = batch_image[i, 1, :, :] * 0.5 + batch_image[i, 2, :, :]
#     axs[i, 0].imshow(image_to_plot.detach().cpu().numpy(), cmap='gray')
#     axs[i, 0].axis('off')
# for b, bb in barcodes.items():
#     for c, bc in bb.items():
#         axs[b, c+1].set_title(f"Barcode classe {c}")
#         for i_c, comp in bc.items():
#             axs[b, c+1].plot([i - max_iter//2 for i in range(max_iter)], comp.detach().cpu().numpy(),label = f"Comp {i_c}")
#             axs[b, c+1].legend()
#             axs[b, c+1].set_xlabel("Filtration Step")
#             axs[b, c+1].set_ylabel("Size")
# plt.tight_layout()
# plt.savefig(f'{SAVING_PATH}/barcodes_maxpool.png')
# plt.close()


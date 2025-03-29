import re
import torch
import torch.nn as nn
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


working_folder = '/home/kefe11/Dataset_25mm_128p'
data_pairs = []  

images_folder = os.path.join(working_folder, "train/images")
masks_folder  = os.path.join(working_folder,  "train/masks")
SLICES_TRAIN       = [f'Slice{i}' for i in range(1, 9)]

for k, filename in enumerate(os.listdir(images_folder)):
    _, _, slice_i, side = filename.split('_')
    if slice_i in SLICES_TRAIN and side.split('.')[0] in ['LEFT', 'RIGHT']:
        image_path = os.path.join(images_folder, filename)
        mask_path  = os.path.join(masks_folder,  filename)       
        mask = nib.load(mask_path).get_fdata()
        n_slice = []
        count = []
        for i in range(9):
            slice_i = mask[:, :, i]
            if not np.all(slice_i == 0):
                count.append(np.sum(slice_i == 1) + np.sum(slice_i == 2))
                n_slice.append(i)
        if len(n_slice) == 0 or 4 not in n_slice or len(n_slice) >1:
            print(mask_path, n_slice, count)
            # image = nib.load(image_path).get_fdata()
            # fig, ax = plt.subplots(1, 3, figsize=(12, 6))
            # ax[0].imshow(mask[:, :, 3, 0, 0, 0], cmap='gray')
            # ax[1].imshow(mask[:, :, 4, 0, 0, 0], cmap='gray')
            # ax[2].imshow(image[:, :, 4, 0, 0, 0], cmap='gray')
            # plt.savefig(f'{k}_{n_slice}.png', bbox_inches='tight', pad_inches=0)
print(nib.load(image_path).header["pixdim"][1:3])

# model0 = torch.load("/home/kefe11/Results_50mm_128p_2/UNet3DTo2D/params_9/model_0.pt", map_location=torch.device('cpu'))
# model1 = torch.load("/home/kefe11/Results_50mm_128p_2/UNet3DTo2D/params_9/model_1.pt", map_location=torch.device('cpu'))
# #model2 = torch.load("/home/kefe11/Results_50mm_128p_2/UNet3DTo2D/params_7/model_2.pt", map_location=torch.device('cpu'))
# #model3 = torch.load("/home/kefe11/Results_50mm_128p_2/UNet3DTo2D/params_7/model_3.pt", map_location=torch.device('cpu'))

# models = [model0['model_state_dict'], model1['model_state_dict']] #, model2['model_state_dict'], model3['model_state_dict']]
# for key in models[0].keys():
#     if "weighted_means" in key:
#         mean = np.zeros(np.shape(models[0][key]))
#         for model in models:
#             mean += np.array(model[key])
#         print(key, mean/len(models))



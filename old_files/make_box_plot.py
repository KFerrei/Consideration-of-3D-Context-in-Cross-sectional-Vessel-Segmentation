import torch
from constants import *
from test import *
import pandas as pd 
from torch.utils.data import DataLoader, random_split, ConcatDataset

############################################################

data_folder = '/home/kefe11/Dataset_25mm_128p_17'
result_folder = '/home/kefe11/Results_25mm_128p_norm/UNet3DTo2D/params_36'
saving_dir = '/home/kefe11/Results_25mm_128p_norm/UNet3DTo2D/params_36/figures'
slice_interset, n_slices = 8, 17 # SLICE_INTEREST
model_name = "unet_3d_to_2d_mid_btl_att_tra" 
filter_1, depth = 32, 6
pixel_dim = 25/128
norm = "minmax"

############################################################

data_pairs = get_data_pairs(data_folder, SLICES_TRAIN, SIDES, slice_interset, NUM_CLASSES)
generator = torch.Generator().manual_seed(SEED)
splits_perc = [1/(N_CROSS_VAL) for i in range(N_CROSS_VAL)]
splits = random_split(data_pairs, splits_perc, generator=generator)

image_stats = pd.DataFrame({})
for i in range(1):
    model = init_model(model_name, NUM_CLASSES, slice_interset, filter_1 = filter_1, depth = depth, 
                        dropout = [0.1, 0.3], n_slices = n_slices)
    model.to(device = DEVICE)
    loaded_checkpoint = torch.load(f"{result_folder}/model_{i}.pt", map_location=DEVICE)
    model.load_state_dict(loaded_checkpoint['model_state_dict'])

    test_pairs = splits[i]
    print(test_pairs)
    test_dataset = Custom3DDataSet(test_pairs, slice_interset, False, normalize = norm)
    _, im_stats = test_model(test_dataset, model, saving_dir, False, pixel_dim = pixel_dim)
    image_stats = pd.concat([image_stats, im_stats], ignore_index=True)

    worst_indices_class_1 = np.argsort(im_stats['DC_Class_1'])[:2]
    worst_indices_class_2 = np.argsort(im_stats['DC_Class_2'])[:2]
    worst_indices_class_3 = np.argsort(im_stats['HD_Class_1'])[-2:][::-1]
    worst_indices_class_4 = np.argsort(im_stats['HD_Class_2'])[-2:][::-1]
    worst_indices_class_5 = np.argsort(im_stats['MCD_Class_1'])[-2:][::-1]
    worst_indices_class_6 = np.argsort(im_stats['MCD_Class_2'])[-2:][::-1]

    best_indices_class_1 = np.argsort(im_stats['DC_Class_1'])[-2:][::-1]
    best_indices_class_2 = np.argsort(im_stats['DC_Class_2'])[-2:][::-1]
    best_indices_class_3 = np.argsort(im_stats['HD_Class_1'])[:2]
    best_indices_class_4 = np.argsort(im_stats['HD_Class_2'])[:2]
    best_indices_class_5 = np.argsort(im_stats['MCD_Class_1'])[:2]
    best_indices_class_6 = np.argsort(im_stats['MCD_Class_2'])[:2]

    worst_indices = np.unique(np.concatenate((worst_indices_class_1, worst_indices_class_2,
                                                worst_indices_class_3, worst_indices_class_4,
                                                worst_indices_class_5, worst_indices_class_6)))
    best_indices = np.unique(np.concatenate((best_indices_class_1, best_indices_class_2,
                                                best_indices_class_3, best_indices_class_4,
                                                best_indices_class_5, best_indices_class_6)))

    plot_inferences(test_dataset, model, saving_dir, slice_interset,  
                    title = f"best_inferences_{i}.png", list_ind = best_indices)
    plot_inferences(test_dataset, model, saving_dir, slice_interset,  
                    title = f"worst_inferences_{i}.png", list_ind = worst_indices)
    
plot_box(image_stats, saving_dir)
# # image_stats.to_csv(os.path.join(saving_dir, "image_statistics.csv"), index=False)
import os
from sklearn.model_selection import KFold
from custom_dataset import get_data_splits_cross_val
def cal(data):
    sqr = 0
    for i in range(len(data)):
        sqr += (len(data[i][1])-530)**2
    sqr = sqr**0.5
    return sqr

folder = '/home/kefe11/Dataset_25mm_128p'
slices = [f'Slice{i}' for i in range(1, 9)]
sides = ['LEFT', 'RIGHT']
n_splits = 5

images_folder = os.path.join(folder, "train/images")
masks_folder  = os.path.join(folder,  "train/masks")

# Step 1: Group data by patient ID
patient_data = {}
for filename in os.listdir(images_folder):
    id_p, exam_i, slice_i, side = filename.split('_')
    if slice_i in slices and side.split('.')[0] in sides:
        image_path = os.path.join(images_folder, filename)
        mask_path = os.path.join(masks_folder, filename)
        if os.path.isfile(image_path) and os.path.isfile(mask_path):
            if id_p not in patient_data:
                patient_data[id_p] = []
            patient_data[id_p].append({"image": image_path, "label": mask_path})

# Step 2: Create patient IDs list and initialize KFold
best_seed = (0, 100)
for s in range(20):
    patient_ids = list(patient_data.keys())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=s)
    # Step 3: Split into folds
    data_splits = []
    for train_index, val_index in kf.split(patient_ids):
        train_data, val_data = [], []
        for idx in train_index:
            train_data.extend(patient_data[patient_ids[idx]])
        for idx in val_index:
            val_data.extend(patient_data[patient_ids[idx]])
        data_splits.append((train_data, val_data))
    err = cal(data_splits)
    print(f"Seed {s}: {err}")
    if err < best_seed[1]:
        best_seed = (s, err)
        print("  ")
print("Best seed:", best_seed)
patient_ids = list(patient_data.keys())
kf = KFold(n_splits=n_splits, shuffle=True, random_state=best_seed[0])

data_splits = get_data_splits_cross_val(folder, slices, sides, n_splits, best_seed[0], is_test = False)
for i in range(len(data_splits)):
    print(len(data_splits[i][1]))



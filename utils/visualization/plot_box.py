"""
File: utils/visualization/box_plot.py
Description: Plots boxplots for the segmentation model
Author: Kevin Ferreira
Date: 18 December 2024
"""

import os
import matplotlib.pyplot as plt

def plot_box(image_stats, saving_dir):
    """
    Plots boxplots for the Dice Coefficient (DC), Hausdorff Distance (HD), and Mean Contour Distance (MCD) metrics 
    for two classes: Wall and Lumen, based on per-image statistics.

    Args:
        image_stats (pandas.DataFrame): A DataFrame containing per-image statistics. 
                                        It must have columns like 'DC_Class_1', 'DC_Class_2', 'HD_Class_1', 
                                        'HD_Class_2', 'MCD_Class_1', 'MCD_Class_2'.
        saving_dir (str): The directory where the generated boxplot image will be saved.

    Returns:
        None
    """
    fig, axs = plt.subplots(1,3, figsize=(15, 6))
    labels =[[ 'DC_Class_1',  'DC_Class_2'], 
                [ 'HD_Class_1',  'HD_Class_2'], 
                ['MCD_Class_1', 'MCD_Class_2']]
    for i, label in enumerate(labels):
        data = [image_stats[label[0]], image_stats[label[1]]]
        axs[i].boxplot(data)
        axs[i].set_xticks(ticks=np.arange(1, len(data) + 1), 
                            labels=['Wall', 'Lumen'], fontsize=20)
        axs[i].tick_params(axis='y', labelsize=20)
        axs[i].set_ylabel(f'{label[0].split("_")[0]}', fontsize=20)
    axs[0].set_ylim(0.5, 1)
    axs[1].set_ylim(0, 6)
    axs[2].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(saving_dir, 'box_plot.png'))
    plt.close()
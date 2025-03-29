"""
File: utils/helpers/file_organizer.py
Description: Script to organize and copy .nii.gz files from a source directory to a destination directory 
             based on file type (image or mask). It organizes them into train and test directories.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import os
import shutil
import argparse

def organize_files(source_folder, destination_folder):
    """
    Organize and copy .nii.gz files from the source folder to the destination folder.
    
    Args:
        source_folder (str): The path to the source folder containing the .nii.gz files.
        destination_folder (str): The path to the destination folder where files will be copied.
    """
     # Create destination directories if they don't exist
    os.makedirs(destination_folder, exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "train/masks"), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "test/images"), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "test/predictions"), exist_ok=True)
    # Walk through the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.nii.gz'):  
                source_file = os.path.join(root, file)
                splited_source = source_file.split('/')
                exam = splited_source[4].split('_')[1]
                id_p = splited_source[5]
                type_f, slice_n, side = splited_source[6].split('_')
                dest_file_name = f"{id_p}_{exam}_Slice{slice_n}_{side}.nii.gz"
                if type_f == "image":
                    destination_file = os.path.join(destination_folder, "train/images", dest_file_name)
                elif type_f == "mask":
                    destination_file = os.path.join(destination_folder, "train/masks", dest_file_name)
                else:
                    continue  
                shutil.copy2(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")

def main():
    parser = argparse.ArgumentParser(description="Organize and copy .nii.gz files")
    parser.add_argument('source_folder', '-sf', type=str, help="Path to the source folder")
    parser.add_argument('destination_folder', '-df', type=str, help="Path to the destination folder")
    args = parser.parse_args()
    organize_files(args.source_folder, args.destination_folder)

if __name__ == "__main__":
    main()
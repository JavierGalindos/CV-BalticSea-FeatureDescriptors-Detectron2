# Task 1: Preprocessing
# Collect a set of images suitable for the tasks below of at least 3 species. 
# Write code to preprocess the
# images of plants into a uniform size of your choice, e.g. 1024x1024 pixels.

# The current task is about automated detection of benthic species of plants in the Baltic Sea:
# • Charophyte
# • Fucus
# • Furcellaria lubricalis
# • Mytilus
# • Zostera marina

#https://laji.fi/

import os
import cv2

print(os.getcwd)
reshape_size = (1024,1024)
raw_img_path = './Raw images/'
processed_img_path = './Processed images/'

for folder in os.listdir('./Raw images'):
    folder_path_raw = os.path.join(raw_img_path, folder)
    folder_path_processed = os.path.join(processed_img_path, folder)
    for file in os.listdir(folder_path_raw):
        full_path_raw = os.path.join(folder_path_raw, file)
        print(full_path_raw)
        #print(full_path)
        img = cv2.imread(full_path_raw)
        # print(img.shape[:2])
        # print(reshape_size)
        if img.shape[:2] != reshape_size:
            img = cv2.resize(img, reshape_size)
        
        full_path_processed = os.path.join(folder_path_processed, file)
        status = cv2.imwrite(full_path_processed, img)
        if not status:
            raise("Failed writing the image.")
    




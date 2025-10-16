import cv2
import os
import numpy as np

def HE(image):
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  v = hsv_image[:, :, 2]
  equalized_v = cv2.equalizeHist(v)
  hsv_image[:, :, 2] = equalized_v
  enhanced_image  = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
  return enhanced_image


def process_images_in_folder(folder_path, output_folder=None, gamma=0.7):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error reading file {image_path}")
            continue

        HE_corrected = HE(image)

        if output_folder:
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, HE_corrected)
            print(f"Processed and saved {output_path}")
        else:
            cv2.imshow('Gamma Corrected Image', HE_corrected)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# train
folder_path = './data/lol/train/input/'
output_folder = './data/lol/train/he/'

# test
# folder_path = './data/val/input/'
# output_folder = './data/val/he/'

process_images_in_folder(folder_path, output_folder, gamma=3)
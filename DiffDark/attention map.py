import cv2
import numpy as np
import os
import torch

# train
input_folder = './data/lol/train/input/'
input_folder1 ='./data/lol/train/gt/'
output_folder = './data/lol/train/at/'

# test
# 注意测试的时候并不会去用这个注意力图，或者你换成任意单通道图像也可以
# input_folder = './data/val/input/'
# input_folder1 ='./data/val/gt/'
# output_folder = './data/val/at/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
image_files1 = [f for f in os.listdir(input_folder1) if f.endswith('.png')]
def gen_att(haze,clear):
    r = torch.tensor(haze[0, :, :]).unsqueeze(0)
    g = torch.tensor(haze[1, :, :]).unsqueeze(0)
    b = torch.tensor(haze[2, :, :]).unsqueeze(0)
    Y = 0.299 * r + 0.587 * g + 0.144 * b
    r_clear = torch.tensor(clear[0, :, :]).unsqueeze(0)
    g_clear = torch.tensor(clear[1, :, :]).unsqueeze(0)
    b_clear = torch.tensor(clear[2, :, :]).unsqueeze(0)
    Y_clear = 0.299 * r_clear + 0.587 * g_clear + 0.144 * b_clear
    m_g = Y_clear - Y
    m_g_max = torch.max(torch.max(m_g,1).values,1).values.unsqueeze(-1).unsqueeze(-1)+1e-6
    m_g_min = torch.min(torch.min(m_g,1).values,1).values.unsqueeze(-1).unsqueeze(-1)
    m_g_l = (m_g- m_g_min)/(m_g_max-m_g_min)
    return m_g_l

for image_file in image_files:
    input_image_path = os.path.join(input_folder, image_file)
    input_image_path1 = os.path.join(input_folder1, image_file)
    image = cv2.imread(input_image_path).transpose((2, 0, 1))
    image1 = cv2.imread(input_image_path1).transpose((2, 0, 1))
    attention_map = gen_att(image1,image)
    attention_map = attention_map.permute(1, 2, 0)
    attention_map_scaled = (attention_map * 255).cpu().numpy().astype(np.uint8)
    output_image_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_image_path, attention_map_scaled)

    print(f'Processed: {image_file} -> Saved as: {output_image_path}')
print('Processing completed.')



from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np
import argparse
import os
import torch
from PIL import Image

# Content Loss

def MSE_Loss(image_1, image_2):
    return mse(image_1, image_2)

def SSIM_Loss(image_1, image_2):
    return ssim(image_1, image_2, full = True)

# Style Loss

def Gram_Loss(image_1, image_2):
    H, W, C = image_1.shape

    F1 = image_1.reshape(-1, C).T 
    F2 = image_2.reshape(-1, C).T
    
    G1 = F1 @ F1.T / (H * W)
    G2 = F2 @ F2.T / (H * W)

    return mse(G1, G2)

Loss_Function_Dict = {
    'SSIM': SSIM_Loss,
    'Gram': Gram_Loss,
    'MSE': MSE_Loss,
}

def compute_loss(config):
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    device = torch.device(device = device)

    image_1_dir = config.image_1
    image_2_dir = config.image_2
    loss_function = Loss_Function_Dict[config.desired_loss]

    # Assumption Here is That We Will Be Comparing Corresponding Images (e.g. noised vs. denoised in00.png)
    image_1_dirs = list(os.listdir(config.image_1))
    image_2_dirs = list(os.listdir(config.image_2))

    losses = list()
    
    for i in range(len(image_1_dirs)):
        curr_im_1_dirr, curr_im_2_dirr = os.path.join(image_1_dir, image_1_dirs[i]), os.path.join(image_2_dir, image_2_dirs[i])
        img_1, img_2 = np.array(Image.open(curr_im_1_dirr).convert("RGB")), np.array(Image.open(curr_im_2_dirr).convert("RGB"))

        curr_loss = loss_function(img_1, img_2)
        losses.append(curr_loss)
        
    print(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_1', type = str, default = './examples/content')
    parser.add_argument('--image_2', type = str, default = './examples/denoised_content')
    parser.add_argument('--desired_loss', type = str, default = 'MSE')
    
    config = parser.parse_args()
    
    compute_loss(config = config)
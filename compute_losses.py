from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import torch
from PIL import Image
import sys
# Content Loss

def MSE_Loss(image_1, image_2):
  return mse(image_1, image_2)

def SSIM_Loss(image_1, image_2):
  return ssim(image_1, image_2, full = True, win_size = 7, channel_axis = -1)[0]

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
    image_1_dir = config.image_1
    image_2_dir = config.image_2
    loss_fn     = Loss_Function_Dict[config.desired_loss]

    # List files in the order the filesystem provides
    files1 = [f for f in os.listdir(image_1_dir)
              if f.lower().endswith(('.png','.jpg','.jpeg'))]
    files2 = [f for f in os.listdir(image_2_dir)
              if f.lower().endswith(('.png','.jpg','.jpeg'))]

    losses = []
    names  = []

    for f1, f2 in zip(files1, files2):
        path1 = os.path.join(image_1_dir, f1)
        path2 = os.path.join(image_2_dir, f2)
        img1 = np.array(Image.open(path1).convert("RGB"))
        img2 = np.array(Image.open(path2).convert("RGB"))

        val = loss_fn(img1, img2)
        losses.append(val)
        if (config.include_both_names):
          names.append(f1 + " & " + f2)
        else:
          names.append(f1)

    os.makedirs(config.plot_location, exist_ok=True)
    out_path = os.path.join(config.plot_location, config.file_name)

    fig, ax = plt.subplots(figsize=(max(8, len(names)*0.5), 12))
    ax.bar(range(len(names)), losses, color='C0')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(config.desired_loss)
    ax.set_title(f"{config.desired_loss} per image")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

    return losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_1', type = str, default = './examples/content')
    parser.add_argument('--image_2', type = str, default = './examples/denoised_content')
    parser.add_argument('--plot_location', type = str, default = './plots')
    parser.add_argument('--file_name', type = str, default = './noised_performance')
    parser.add_argument('--desired_loss', type = str, default = 'MSE')
    parser.add_argument('--include_both_names', type = bool, default = True)
    
    config = parser.parse_args()
    
    compute_loss(config = config)
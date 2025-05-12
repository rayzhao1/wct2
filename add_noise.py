import numpy as np
import argparse
import os
import torch
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def add_noise(image, loc = 0, scale = 1):
    noise = np.random.normal(loc = loc, scale = scale, size = image.shape)
    noised_signal = image + noise.astype(int)
    noised_signal = np.clip(noised_signal, 0.0, 1.0)
    return noised_signal

def noise_images(config):

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    device = torch.device(device = device)

    # Creating Desired Directory if Needed
    destination_path = config.desired_dir

    try:
        os.mkdir(destination_path)
    except:
        pass

    # Artificially Noising Images

    image_dirs = set(os.listdir(config.images))
    for image_dir in image_dirs:
        full_dir = os.path.join(config.images, image_dir)
        image = mpimg.imread(full_dir)
        
        noised_image = add_noise(image = image)
        noised_path = os.path.join(config.desired_dir, image_dir)
        mpimg.imsave(noised_path, noised_image)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type = str, default = './examples/content')
    parser.add_argument('--desired_dir', type = str, default = './examples/noised_content')
    config = parser.parse_args()
    noise_images(config = config)
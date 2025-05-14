import numpy as np
import argparse
import os
import matplotlib.image as mpimg
import torch
from PIL import Image

def add_gaussian_noise(image, loc=0.0, scale=1.0, strength=0.75):
    img = image.astype(np.float32)
    noise = strength * np.random.normal(loc=loc, scale=scale, size=img.shape).astype(np.float32)
    noised = img + noise
    print(noised.shape)
    print()
    return np.clip(noised, 0.0, 1.0)

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
        image = Image.open(full_dir).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        noised_image = (add_gaussian_noise(image = image) * 255.0).astype('uint8')
        noised_path = os.path.join(config.desired_dir, image_dir)
        Image.fromarray(noised_image).save(noised_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type = str, default='./examples/content')
    parser.add_argument('--desired_dir', type = str, default='./examples/noised_content')
    config = parser.parse_args()
    noise_images(config)

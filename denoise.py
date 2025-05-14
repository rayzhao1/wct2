import numpy as np
import argparse
import os
import torch
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from io import BytesIO

from bm3d import bm3d
import numpy as np
from skimage.restoration import denoise_wavelet


from scipy import ndimage as nd
import cv2

# Non-Local Means Based Denoising
def nl(config, destination_path, image_dir):
    full_dir = os.path.join(config.images, image_dir)
    img = np.array(Image.open(full_dir).convert("RGB"))
    out = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return out

# Wavelet-Based Denoising
def wavelet_denoise(config, destination_path, image_dir):
    full_dir = os.path.join(config.images, image_dir)
    img = np.array(Image.open(full_dir).convert("RGB"))
    out = denoise_wavelet(img, mode = 'soft', wavelet_levels = 2, wavelet = 'haar') * 255
    return out.astype('uint8')

# FFT-Based Denoising
def fft_denoise(config, destination_path, image_dir):
    full_dir = os.path.join(config.images, image_dir)
    img = np.array(Image.open(full_dir).convert("RGB"))
    
    H, W, C = img.shape
    out = np.zeros_like(img, dtype=float)
    
    for c in range(C):
        F = np.fft.fft2(img[:, :, c] / 255.0)
        Fshift = np.fft.fftshift(F)
        mag, phase = np.abs(Fshift), np.angle(Fshift)
        
        # Adjustable Percentile-Based Filter
        percentile_threshold = config.image_percentile_threshold  # Use the config value
        mask = mag >= np.percentile(mag, percentile_threshold)
        
        # Apply a Gaussian filter to smooth the mask
        mask = nd.gaussian_filter(mask.astype(float), sigma=2)
        filtered = Fshift * mask
        
        F_ishift = np.fft.ifftshift(filtered)
        recon = np.fft.ifft2(F_ishift).real
        
        out[:, :, c] = recon
    
    out = (out * 255).astype('uint8')
    return out
    

# Median Filter
def median_filter_denoise(config, destination_path, image_dir):
    full_dir = os.path.join(config.images, image_dir)
    img = np.array(Image.open(full_dir).convert("RGB"))
    out = cv2.medianBlur(img, ksize = 3)
    return out

denoise_scheme_dict = {
    'FFT': fft_denoise,
    'Median': median_filter_denoise, 
    'NL': nl,
    'Wavelet': wavelet_denoise,
}


def denoise(config):
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    device = torch.device(device = device)

    # Creating Desired Directory if Needed
    destination_path = config.desired_dir
    denoise_scheme = denoise_scheme_dict[config.denoise_scheme]
    image_dirs = set(os.listdir(config.images))

    try:
      os.makedirs(destination_path, exist_ok = True)
    except:
      pass 

    for image_dir in image_dirs:

        output = denoise_scheme(config = config, destination_path = destination_path, image_dir = image_dir)
        # Save the denoised image
        denoised_path = os.path.join(destination_path, image_dir)   
        Image.fromarray(output).save(denoised_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type = str, default = './examples/noised_content')
    parser.add_argument('--desired_dir', type = str, default = './examples/denoised_content')
    parser.add_argument('--image_quality', type = str, default = 50)
    parser.add_argument('--image_threshold_lb', type = str, default = 40000)
    parser.add_argument('--image_threshold_ub', type = str, default = 100000)
    parser.add_argument('--image_percentile_threshold', type = str, default = 98)
    parser.add_argument('--denoise_scheme', type = str, default = "Wavelet")
    config = parser.parse_args()
    denoise(config = config)
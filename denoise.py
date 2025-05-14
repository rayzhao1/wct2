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
import pywt

# JPEG-Style Compression
def jpeg_compression_denoise(config, destination_path, image_dirs):
    quality = config.image_quality

    image_dirs = set(os.listdir(config.images))

    for image_dir in image_dirs:
        full_dir = os.path.join(config.images, image_dir)
        img = Image.open(full_dir).convert("YCbCr") # Getting rid of Alpha Channel; RGBA -> RGB

        buffer = BytesIO()
        img.save(buffer, "JPEG", quality = quality)
        buffer.seek(0)

        destination_path = os.path.join(config.desired_dir, image_dir)

        base, _ = os.path.splitext(image_dir)
        out_fname = base + ".jpg"
        out_path  = os.path.join(config.desired_dir, out_fname)
        with open(out_path, "wb") as f:
            f.write(buffer.getvalue())

# Non-Local Means Based Denoising
def nl(config, destination_path, image_dir):
    full_dir = os.path.join(config.images, image_dir)
    img = np.array(Image.open(full_dir).convert("RGB"))
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    denoised_path = os.path.join(destination_path, image_dir)
    mpimg.imsave(denoised_path, dst)

# Wavelet-Based Denoising
def wavelet_denoise(config, destination_path, image_dir):
    full_dir = os.path.join(config.images, image_dir)
    img = np.array(Image.open(full_dir).convert("RGB"))
    denoised_image = denoise_wavelet(img, mode = 'soft', wavelet_levels = 2, wavelet = 'haar')
    denoised_path = os.path.join(destination_path, image_dir)
    mpimg.imsave(denoised_path, denoised_image)

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
        
        # Percentile-Based Filter
        mask = mag >= np.percentile(mag, 99)
        filtered = Fshift * mask
        
        F_ishift = np.fft.ifftshift(filtered)
        recon = np.fft.ifft2(F_ishift).real
        
        out[:, :, c] = recon
    
    out = np.clip(out, 0.0, 1.0)
    
    # Save the denoised image
    denoised_path = os.path.join(destination_path, image_dir)
    mpimg.imsave(denoised_path, out)

# Median Filter
def median_filter_denoise(config, destination_path, image_dir):
    full_dir = os.path.join(config.images, image_dir)
    img = np.array(Image.open(full_dir).convert("RGB"))
    dst = cv2.medianBlur(img, ksize = 3)
    denoised_path = os.path.join(destination_path, image_dir)
    mpimg.imsave(denoised_path, dst)
        
# BM3D Filter        
def bm3d_filter(config, destination_path, image_dir):
    full_dir = os.path.join(config.images, image_dir)
    noisy = Image.open(full_dir).convert("RGB")
    sigma = 0.1  # your known noise sigma
    denoised = bm3d(noisy, sigma_psd = sigma)
    denoised_path = os.path.join(destination_path, image_dir)
    mpimg.imsave(denoised_path, denoised)

denoise_scheme_dict = {
    'FFT': fft_denoise,
    'Median': median_filter_denoise, 
    'bm3d': bm3d_filter,
    'nl': nl,
    'Wavelet': wavelet_denoise,
}


def denoise(config):
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    device = torch.device(device = device)

    # Creating Desired Directory if Needed
    destination_path = config.desired_dir

    try:
        os.mkdir(destination_path)
    except:
        pass    

    denoise_scheme = config.denoise_scheme
    denoise_scheme = denoise_scheme_dict[denoise_scheme]

    image_dirs = set(os.listdir(config.images))
    for image_dir in image_dirs:
        print('new image')
        denoise_scheme(config = config, destination_path = destination_path, image_dir = image_dir)

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
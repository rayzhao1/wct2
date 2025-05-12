import numpy as np
import argparse
import os
import torch
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import io
from scipy.fftpack import dct, idct

def jpeg_compress_decompress(img, quality = 50):
    # Convert to uint8 [0,255] if needed
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img = img
    
    # Encode to JPEG in memory
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    # Decode back to array
    img_decoded = np.array(Image.open(buffer))
    
    # Convert to float in [0,1]
    img_out = img_decoded.astype(np.float32) / 255.0
    return img_out

# Example usage:
# content = plt.imread("path/to/content.png")
# content = content.astype(np.float32) / 255.0
# denoised = jpeg_compress_decompress(content, quality=40)
# plt.imshow(denoised); plt.axis("off")


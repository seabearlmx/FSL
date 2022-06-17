import torch
import numpy as np


def distance_np(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize[0]/2) ** 2 + (j - imageSize[1]/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial_np(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance_np(i, j, imageSize=(rows, cols), r=r)
    return mask


def FDA_low_np(img, r):
    img_np = img  # .cpu().numpy()
    Images_freq_low = []
    tmp = np.zeros([img_np.shape[0], img_np.shape[1], 3])
    r = np.amin((img_np.shape[0], img_np.shape[1])) - r
    low_mask = mask_radial_np(np.zeros([img_np.shape[0], img_np.shape[1]]), r)
    for i in range(3):
        a_ = np.fft.fftshift(np.fft.fft2(img_np[:, :, i]))
        low_img = a_ * low_mask
        low_img = np.fft.ifft2(np.fft.ifftshift(low_img))
        tmp[:, :, i] = np.real(low_img)
    Images_freq_low.append(tmp)
    return np.array(Images_freq_low, dtype='float32')


def FDA_high_np(img, r):
    img_np = img  # .cpu().numpy()
    Images_freq_high = []
    tmp = np.zeros([img_np.shape[0], img_np.shape[1], 3])
    low_mask = mask_radial_np(np.zeros([img_np.shape[0], img_np.shape[1]]), r)
    for i in range(3):
        a_ = np.fft.fftshift(np.fft.fft2(img_np[:, :, i]))
        high_img = a_ * (1 - low_mask)
        high_img = np.fft.ifft2(np.fft.ifftshift(high_img))
        tmp[:, :, i] = np.real(high_img)
    Images_freq_high.append(tmp)
    return np.array(Images_freq_high, dtype='float32')

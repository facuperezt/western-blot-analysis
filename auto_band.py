#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
from typing import List
from PIL import Image
from blot_math import CompareBands

path = "img.jpeg"

img_PIL = Image.open(path).convert('L')
# read the image
img = cv2.imread(path)
# convert to gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# blur
smooth = cv2.GaussianBlur(gray, (95,95), 0)
# divide gray by morphology image
division = cv2.divide(gray, smooth, scale=192)

# img_PIL = Image.fromarray(np.uint8(division), mode= 'L')
full_strip = np.asarray(img_PIL)

def gaussian_smooth(x, kernel_percentage, s = 1):
    filter_length = int(max(len(x)//(1/kernel_percentage), 3))
    filter_length += ((filter_length%2) + 1) % 2 
    mid = filter_length//2 - 1
    kernel_x = [(1/(s*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*s**2)))) for i in range(-mid,mid+1)]
    kernel_x = np.array(kernel_x)
    kernel_x /= kernel_x.sum()

    ret = np.convolve(x, kernel_x, 'valid')
    length_diff = len(x) - len(ret)
    div, mod = divmod(length_diff, 2)
    padded = np.pad(ret, (div, div+mod), 'edge')
    padded[:div] = x[:div]
    padded[-(div+mod):] = x[-(div+mod):]

    return padded

def detect_nr_bands(full_strip : np.ndarray):
    plt.figure()
    x = full_strip.mean(axis=0)
    x += 255 - x.max()
    x /= x.min()
    x -= 1
    dx = x[1:] - x[:-1]
    smooth_dx = gaussian_smooth(dx, 0.01, 5)
    peaks, _ = find_peaks(np.diff(smooth_dx, 1), height=0.01, distance=len(x)/20)
    peaks_dist = peaks[1:] - peaks[:-1]
    plt.plot(x)
    for (peak1, peak2) in zip(peaks[1:], peaks[:-1]):
        _p = [peak1, peak2]
        _c = 'green' if peak1 - peak2 > peaks_dist.mean() else 'red'
        plt.plot(_p, x[_p], '--x', color=_c)
    

    return (peaks_dist > peaks_dist.min()).sum()

def detect_bands(full_strip : np.ndarray, nr_bands = None, kernel_percentage = 0) -> List[np.ndarray]:
    """
    Gets the full strip image as a numpy array and returns it divided into the separate bands to analyze.
    """
    if nr_bands is None:
        nr_bands = detect_nr_bands(full_strip)

    x = full_strip.mean(axis=0)
    if kernel_percentage > 0:
        x = gaussian_smooth(x, kernel_percentage)
    peaks, _ = find_peaks(x, distance=len(x)//nr_bands)

    return [full_strip[:, start:end] for start, end in zip([0] + peaks.tolist(), peaks.tolist() + [None])], [0] + peaks.tolist()
    

bands = CompareBands()

detected_bands, leftmost_pixel = detect_bands(full_strip)
for band,left_pixel in zip(detected_bands, leftmost_pixel):
    plt.figure()
    plt.imshow(band)
    bands.add_band(band, left_pixel)

bands.math()

# %%

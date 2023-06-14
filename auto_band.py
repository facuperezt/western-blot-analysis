#%%
from importlib import reload
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import ball, binary_closing, disk
import matplotlib.pyplot as plt
import blot_math
reload(blot_math)
from blot_math import CompareBands
from argparse import ArgumentParser

def correct_background(image):
    # Apply background subtraction using rolling ball algorithm
    radius = 50  # Adjust this radius value as needed
    selem = disk(radius)
    background = binary_closing(image < threshold_otsu(image), selem)
    corrected_image = np.where(background, image, image - background)

    return corrected_image


def detect_bands(image_path):
    # Load the image
    image = cv2.imread(image_path, 0)  # Read the image as grayscale
    image = crop_image(image)
    # Correct the background
    corrected_image = correct_background(image)

    # Preprocess the image
    blurred = cv2.GaussianBlur(corrected_image, (5, 5), 1, 0)  # Apply Gaussian blur to reduce noise
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply thresholding
    # Invert thresholded because we're working with inverted images
    thresholded = 255 - thresholded

    # Find contours in the image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and get bounding boxes
    areas = []
    bounds = []
    middles = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))
        bound = cv2.boundingRect(contour)
        bounds.append(bound)
        middles.append((bound[0] + bound[2]/2, bound[1] + bound[3]/2))
    min_area = (max(areas) + min(areas))/3 # Minimum area to consider as a band (adjust as needed)
    # fig, ax = plt.subplots()
    # ax.plot([x for x,y in middles], areas, 'x')
    # x_lim = ax.get_xlim()
    # ax.hlines(min_area, -10, 1000, 'gray', 'dashed', 'min_area', alpha=0.4)
    # ax.set_xlim(x_lim)
    bounding_boxes = []
    half_contours = []
    for (x, y, w, h), area in zip(bounds, areas):
        if area > min_area or (area > min_area/2 and w/h > 2.5):
            bounding_boxes.append((x, y, w, h))
        elif area > min_area/3 and w/h > 0.6 and w/h < 2:
            half_contours.append((x, y, w, h))

    # Draw the detected bands on the image
    color_image = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR)
    centers = []
    max_size = [0, 0]
    for x, y, w, h in bounding_boxes:
        # cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        centers.append([x+(w/2), y+(h/2)])
        _w, _h = max_size
        max_size = [max(_w, w), max(_h, h)]
    
    half_centers = []
    for x, y, w, h in half_contours:
        # cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        half_centers.append([x+(w/2), y+(h/2)])
        _w, _h = max_size
        max_size = [max(_w, w), max(_h, h)]

    new_bounding_boxes = combine_centers(half_contours, bounding_boxes)
    for x, y, w, h in new_bounding_boxes:
        _w, _h = max_size
        max_size = [max(_w, w), max(_h, h)]


    w, h = max_size
    new_bounding_boxes = [[int(_x - ((w - _w)/2)), int(_y - ((h - _h)/2)), w + 1, h + 1] for _x, _y, _w, _h in new_bounding_boxes]
    for x,y,w,h in new_bounding_boxes:
        cv2.rectangle(color_image, pt1= (x,y), pt2 = (x+w, y+h), color= (0,255,0), thickness= 3)

    # # Display the results
    # cv2.imshow("Detected Bands", color_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.figure(figsize=(12,9))
    plt.imshow(color_image)

    # Return the bounding boxes
    return [image[y:y+h, x:x+w] for x, y, w, h in new_bounding_boxes], new_bounding_boxes

def combine_centers(half_contours, bounding_boxes):
    if not half_contours:
        final_contours = bounding_boxes 
        final_order = sorted(range(len(final_contours)), key= lambda x: final_contours[x][0])

        return [final_contours[i] for i in final_order]
    half_centers = [[x+(w/2), y+(h/2)] for x,y,w,h in half_contours]
    centers = [[x+(w/2), y+(h/2)] for x,y,w,h in bounding_boxes]
    all = half_centers + centers
    order = sorted(range(len(all)), key= lambda x: all[x][0])
    all : list = [all[k] for k in order]
    half_centers = sorted(half_centers)
    centers = sorted(centers)

    order_half_contours = sorted(range(len(half_contours)), key= lambda x: half_contours[x][0])
    hc_sorted = [half_contours[k] for k in order_half_contours]

    new_contours = []
    combined_hc_mask = [0]* len(half_centers)
    fancy_sorted_half_centers = [half_centers[0]]
    i = 1
    while True:
        if len(fancy_sorted_half_centers) == len(half_centers): break
        fancy_sorted_half_centers.append(half_centers[-i])
        if len(fancy_sorted_half_centers) == len(half_centers): break
        fancy_sorted_half_centers.append(half_centers[i])
        i += 1

    for j,hc in enumerate(fancy_sorted_half_centers):
        idx = all.index(hc)
        idx_hc = half_centers.index(hc)
        if combined_hc_mask[idx] == 1: continue
        # If its the last and the previous is NOT a half_center, we missed something. Same for the first. Or if it has no half_center neighbours.
        if (hc == all[-1] and all[-2] not in half_centers) or \
            (hc == all[0] and all[1] not in half_centers) or \
            (all[idx-1] not in half_centers and all[idx+1] not in half_centers):
            raise RuntimeError('There was an error predicting the bands, please re-run the program with different parameters.')
        # If its the last, check if the previous also is a half_center, then merge.
        elif hc == all[-1] and all[-2] in half_centers and not (combined_hc_mask[-1] or combined_hc_mask[-2]):
            combined_hc_mask[-1] = 1
            combined_hc_mask[-2] = 1
            new_contours.append(_combine_half_contours(hc_sorted[-2], hc_sorted[-1]))
        # If its the first and the next is also a half_center, merge.
        elif hc == all[0] and all[1] in half_centers and not (combined_hc_mask[0] or combined_hc_mask[1]):
            combined_hc_mask[0] = 1
            combined_hc_mask[1] = 1
            new_contours.append(_combine_half_contours(hc_sorted[0], hc_sorted[1]))
        # If only left, merge left.
        elif all[idx - 1] in half_centers and all[idx + 1] not in half_centers and not (combined_hc_mask[idx - 1] or combined_hc_mask[idx]):
            combined_hc_mask[idx - 1] = 1
            combined_hc_mask[idx] = 1
            new_contours.append(_combine_half_contours(hc_sorted[idx_hc - 1], hc_sorted[idx_hc]))
        # If only right, merge right.
        elif all[idx - 1] not in half_centers and all[idx + 1] in half_centers and not (combined_hc_mask[idx] or combined_hc_mask[idx + 1]):
            combined_hc_mask[idx] = 1
            combined_hc_mask[idx + 1] = 1
            new_contours.append(_combine_half_contours(hc_sorted[idx_hc], hc_sorted[idx_hc + 1]))
        # If one on each side, we get funky.
        elif all[idx - 1] in half_centers and all[idx + 1] in half_centers:
            print('xdxdxdxdxdxdxdxdxdxdxdxd')

    final_contours = bounding_boxes + new_contours
    final_order = sorted(range(len(final_contours)), key= lambda x: final_contours[x][0])

    return [final_contours[i] for i in final_order]

def _combine_half_contours(bb1, bb2):
    x1, y1, w1, h1 = bb1 # left one
    x2, y2, w2, h2 = bb2 # right one

    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y

    return x, y, w, h

def crop_image(image):
    if image.shape[1]/image.shape[0] < 2:
        image = image[100:, 100:]
        mask = image < 250
        image = image[mask.any(axis=1)][:, mask.any(axis=0)]
    return image

def plot_detected_bands(detected_bands, bounding_boxes):
    fig, axs = plt.subplots(1, len(detected_bands), sharex= True, sharey= True)
    for band, ax in zip(detected_bands, axs):
        print(band.shape)
        ax.imshow(band)
    fig.show()

    bands = CompareBands()

    fig, axs = plt.subplots(2, len(detected_bands), sharex= 'row', sharey= 'row', figsize=(12,9))
    for band,bbox, ax in zip(detected_bands, bounding_boxes, axs[0]):
        ax : plt.Axes
        left_pixel = bbox[0]
        ax.imshow(band)
        ax.axis('off')
        bands.add_band(band, left_pixel)

    bands.plot_bars(axs[1])
    fig.show()

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--image_path', default= 'img.jpeg', type= str, help= 'Path to the blot image to be analyzed.')
    known, unknown = parser.parse_known_args()

    return known

if __name__ == "__main__":
    args = arg_parser()
    # Example usage
    image_path = args.image_path
    # image_path = "wide_pic_bands.jpeg"
    detected_bands, bounding_boxes = detect_bands(image_path)
    plot_detected_bands(detected_bands, bounding_boxes)
    plt.show()
# %%
